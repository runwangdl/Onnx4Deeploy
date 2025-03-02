import onnx
import onnxruntime as ort
import numpy as np
import yaml
import sys
import os
import torch
import subprocess
from onnx import TensorProto, helper
from CCT.cct import cct_test  # Ensure correct import based on your CCT module

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

def generate_cct_onnx_and_data(save_path=None):
    """ Generate ONNX model for CCT based on config, with optional save path """

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
    folder_name = f"CCT_infer_{img_size}_{embedding_dim}_{num_heads}_{num_layers}"
    
    # Set default save path if not provided
    base_path = save_path if save_path else os.path.join(script_dir, "onnx", folder_name)

    # Define standard filenames
    onnx_file = os.path.join(base_path, "network.onnx")
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")

    # Ensure the save directory exists
    os.makedirs(base_path, exist_ok=True)

    # Load CCT model with parameters from config
    model = cct_test(
        pretrained=pretrained, 
        img_size=img_size, 
        num_classes=num_classes, 
        embedding_dim=embedding_dim, 
        num_heads=num_heads,  
        num_layers=num_layers  
    )
    model.eval()

    # Generate random input data
    input_data = np.random.randn(*input_shape).astype(np.float32)
    np.savez(input_file, input=input_data)

    # Convert input data to tensor
    input_tensor = torch.tensor(input_data)

    # Export model to ONNX
    torch.onnx.export(
        model,
        input_tensor,
        onnx_file,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"✅ ONNX model saved to {onnx_file}")

    # Run ONNX optimization steps
    run_onnx_optimization(onnx_file, embedding_dim, num_heads, input_shape)

    # Load and run inference with ONNX Runtime
    ort_session = ort.InferenceSession(onnx_file)
    output_data = ort_session.run(None, {"input": input_data})[0]

    # Save output data directly
    np.savez(output_file, output=output_data)
    print(f"✅ Output data saved to {output_file}")

if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_cct_onnx_and_data(save_path)
