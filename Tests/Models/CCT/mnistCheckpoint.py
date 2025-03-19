import numpy as np
import onnxruntime as ort
import yaml
import os
import torch
import torchvision
from torchvision import transforms
from utils.utils import load_config
from utils.utils import run_onnx_optimization, load_config, rename_and_save_onnx, run_train_onnx_optimization, rename_nodes, randomize_layernorm_params
from utils.fixshape import infer_shapes_with_custom_ops, print_onnx_shapes

def preprocess_mnist(batch_size, image_size):
    """
    Preprocess MNIST dataset with configurable image size.
    
    Args:
        batch_size: Number of images to process
        image_size: Size to resize images to (will be used as both height and width)
        
    Returns:
        Tuple of (images, labels) as numpy arrays
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    
    dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    indices = np.random.choice(len(dataset), batch_size, replace=False)
    images = torch.stack([dataset[i][0] for i in indices])
    labels = np.array([dataset[i][1] for i in indices], dtype=np.int64)
    
    return images.numpy(), labels

def run_onnx_model(input_data, labels, model_path):
    """
    Run inference on ONNX model.
    
    Args:
        input_data: Input data for the model
        labels: Labels for the model
        model_path: Path to the ONNX model
        
    Returns:
        Dictionary of model outputs
    """
    ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    
    output_names = [output.name for output in ort_session.get_outputs()]
    print(f"Model has {len(output_names)} outputs: {output_names}")
    
    outputs = ort_session.run(None, {"input": input_data, "labels": labels})
    
    output_dict = {}
    for i, name in enumerate(output_names):
        output_dict[name] = outputs[i]
    
    return output_dict

def create_test_input_output():
    """
    Create test input and output files by running inference on the model.
    """
    # Load config
    config = load_config()
    if isinstance(config, tuple):
        # Handle the case where load_config returns a tuple of values
        pretrained, img_size, num_classes, embedding_dim, num_heads, num_layers, batch_size, opset_version = config
    else:
        # Handle the case where load_config returns a dictionary
        img_size = config.get("img_size", 16)  # Default to 16 if not specified
        batch_size = config.get("batch_size", 8)  # Default to 8 if not specified
        embedding_dim = config.get("embedding_dim", 384)
        num_heads = config.get("num_heads", 6)
        num_layers = config.get("num_layers", 7)
    
    print(f"Using image size: {img_size}, batch size: {batch_size}")
    
    folder_name = f"CCT_train_{img_size}_{embedding_dim}_{num_heads}_{num_layers}"
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_dir, "onnx", folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    network_path = os.path.join(base_dir, "onnx", folder_name, "network_train.onnx")
    input_path = os.path.join(base_dir, "onnx", folder_name, "inputs.npz")
    output_path = os.path.join(base_dir, "onnx", folder_name, "outputs.npz")
    
    print(f"Network path: {network_path}")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    
    if not os.path.exists(network_path):
        raise FileNotFoundError(f"ONNX model file not found: {network_path}")
    
    # Create input data with the specified image size
    input_data, labels = preprocess_mnist(batch_size, img_size)
    np.savez(input_path, input=input_data, labels=labels)
    print(f"✅ Input saved to inputs.npz (image size: {img_size}x{img_size}, batch size: {batch_size})")
    
    # Run the model
    outputs_dict = run_onnx_model(input_data, labels, model_path=network_path)
    
    # Save outputs
    np.savez(output_path, **outputs_dict)
    print(f"✅ Output saved to outputs.npz with {len(outputs_dict)} values")
    
    # Print output shapes
    print("Output shapes:")
    for name, arr in outputs_dict.items():
        print(f"  {name}: {arr.shape}")

def main():
    """
    Main function to run the script with better error handling
    """
    try:
        create_test_input_output()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()