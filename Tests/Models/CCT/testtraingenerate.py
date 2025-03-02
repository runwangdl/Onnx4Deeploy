import onnx
import torch
import yaml
import os
import numpy as np
import sys
import subprocess
import io
from torch import nn
from CCT.cct import cct_test  # Ensure correct import based on your CCT module
from onnxruntime.training import artifacts

def make_c_name(name, count=0):
    if name.lower() in ["input", "output"]:
        return name  # Keep 'input' and 'output' as is
    
    name = re.sub(r'input|output', '', name, flags=re.IGNORECASE)  # Remove 'input' and 'output' from other names
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    if name[0].isdigit() or name[0] == '_':
        name = f'node_{count}' + name
    return name

def run_onnx_optimization(onnx_file, embedding_dim, num_heads, input_shape):
    """ Run ONNX Runtime tools to optimize the model """

    batch_size, channels, height, width = input_shape  # Extract input dimensions

    try:
        print("🔹 Fixing dynamic shape...")
        subprocess.run([
            "python", "-m", "onnxruntime.tools.make_dynamic_shape_fixed",
            "--input_name", "input",
            "--input_shape", f"{batch_size},{channels},{height},{width}",
            onnx_file, onnx_file
        ], check=True)

        print("🔹 Running symbolic shape inference...")
        subprocess.run([
            "python", "-m", "onnxruntime.tools.symbolic_shape_infer",
            "--input", onnx_file, "--output", onnx_file, "--verbose", "3"
        ], check=True)

        print("🔹 Optimizing ONNX model for ViT...")
        subprocess.run([
            "python", "-m", "onnxruntime.transformers.optimizer",
            "--input", onnx_file, "--output", onnx_file,
            "--model_type", "vit",
            "--num_heads", str(num_heads),  # Controlled via config
            "--hidden_size", str(embedding_dim),  # Ensures hidden size = embedding_dim
            "--use_multi_head_attention",
            "--disable_bias_skip_layer_norm",
            "--disable_skip_layer_norm",
            "--disable_bias_gelu"
        ], check=True)

        print("✅ ONNX model optimization complete!")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error during ONNX optimization: {e}")

def generate_cct_training_onnx(save_path=None):
    """ Generate ONNX training model for CCT based on config, with optional save path """

    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config.yaml")

    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    model_config = config["cct"]

    # Extract parameters from config
    pretrained = model_config["pretrained"]
    img_size = model_config["img_size"]  
    num_classes = model_config["num_classes"]
    embedding_dim = model_config["embedding_dim"]
    num_heads = model_config["num_heads"]  
    num_layers = model_config["num_layers"]  
    opset_version = model_config.get("opset_version", 12)

    # Ensure input_shape is correctly set based on img_size
    input_shape = (1, 3, img_size, img_size)  

    # Generate a unique folder name based on model parameters
    folder_name = f"CCT_train_{img_size}_{embedding_dim}_{num_heads}_{num_layers}"
    
    # Set default save path if not provided
    base_path = save_path if save_path else os.path.join(script_dir, "onnx", folder_name)
    os.makedirs(base_path, exist_ok=True)  # Ensure directory exists

    # Define filenames
    onnx_infer_file = os.path.join(base_path, "network_infer.onnx")  # Initial PyTorch exported ONNX
    onnx_train_file = os.path.join(base_path, "network_train.onnx")  # Final training model

    # Load CCT model with parameters from config
    model = cct_test(
        pretrained=pretrained, 
        img_size=img_size, 
        num_classes=num_classes, 
        embedding_dim=embedding_dim, 
        num_heads=num_heads,  
        num_layers=num_layers  
    )
    model.train()  # Ensure the model is in training mode

    # Identify only `classifier.fc.weight` for training
    all_param_names = [name for name, _ in model.named_parameters()]
    requires_grad = [name for name in all_param_names if name in ["classifier.fc.weight", "classifier.fc.bias", "/classifier/attention_pool/Transpose_output_0"]]  # Train only `classifier.fc.weight`
    frozen_params = [name for name in all_param_names if name not in requires_grad]  # Freeze all others

    print(f"🔹 Training Only: {requires_grad}")
    print(f"🔹 Frozen Parameters: {frozen_params}")

    # Generate random input data for export
    input_tensor = torch.randn(*input_shape, dtype=torch.float32)

    # Export model to ONNX in training mode
    f = io.BytesIO()
    torch.onnx.export(
        model,
        input_tensor,
        f,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset_version,
        do_constant_folding=False,  # Ensure parameters are not folded into constants
        training=torch.onnx.TrainingMode.TRAINING,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        export_params=True,
        keep_initializers_as_inputs=False,
    )

    # Load ONNX model from buffer and save it as network_infer.onnx
    onnx_model = onnx.load_model_from_string(f.getvalue())
    onnx.save(onnx_model, onnx_infer_file)
    print(f"✅ Inference ONNX model saved to {onnx_infer_file}")

    run_onnx_optimization(onnx_infer_file, embedding_dim, num_heads, input_shape)
    onnx_model = onnx.load(onnx_infer_file)

    # Generate artifacts for training
    artifacts.generate_artifacts(
        onnx_model,
        optimizer=artifacts.OptimType.AdamW,
        loss=artifacts.LossType.CrossEntropyLoss,
        requires_grad=requires_grad,  # Only classifier.fc.weight is trainable
        frozen_params=frozen_params,  # Freeze all others
        artifact_directory=base_path,
        additional_output_names=["output"]
    )

    training_model_path = os.path.join(base_path, "training_model.onnx")
    if os.path.exists(training_model_path):
        os.rename(training_model_path, onnx_train_file)
        print(f"✅ Final Training ONNX model saved as {onnx_train_file}")
    


if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_cct_training_onnx(save_path)
