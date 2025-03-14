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

def preprocess_mnist(batch_size, image_size=16):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    
    dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    indices = np.random.choice(len(dataset), batch_size, replace=False)
    images = torch.stack([dataset[i][0] for i in indices])  # [batch_size, 3, 16, 16]
    labels = np.array([dataset[i][1] for i in indices], dtype=np.int64)
    
    return images.numpy(), labels

def run_onnx_model(input_data, labels, model_path):
    ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    
    output_names = [output.name for output in ort_session.get_outputs()]
    print(f"Model has {len(output_names)} output: {output_names}")
    
    outputs = ort_session.run(None, {"input": input_data, "labels": labels})
    
    output_dict = {}
    for i, name in enumerate(output_names):
        output_dict[name] = outputs[i]
    
    return output_dict

def create_test_intput_output():
    pretrained, img_size, num_classes, embedding_dim, num_heads, num_layers, batch_size, opset_version = load_config()
    
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
    
    print(f"Using batch_size={batch_size}")
    
    if not os.path.exists(network_path):
        raise FileNotFoundError(f"ONNX model file not found: {network_path}")
    
    input_data, labels = preprocess_mnist(batch_size)
    np.savez(input_path, input=input_data, labels=labels)
    print("✅ Input Saved to inputs.npz")
    
    outputs_dict = run_onnx_model(input_data, labels, model_path=network_path)
    
    np.savez(output_path, **outputs_dict)
    print(f"✅ Output Saved to outputs.npz with {len(outputs_dict)} values")
    
    print("Output shapes:")
    for name, arr in outputs_dict.items():
        print(f"  {name}: {arr.shape}")

if __name__ == "__main__":
    create_test_intput_output()