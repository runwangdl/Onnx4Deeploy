import numpy as np
import onnxruntime as ort
import yaml
import os
import torch
import torchvision
from torchvision import transforms
from utils.utils import load_config

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

    outputs = ort_session.run(None, {"input": input_data, "labels": labels})

    loss, grad = outputs
    return loss, grad



def main():
  
    pretrained, img_size, num_classes, embedding_dim, num_heads, num_layers, batch_size, opset_version = load_config()
    
    print(f"Using batch_size={batch_size}")

  
    input_data, labels = preprocess_mnist(batch_size)


    np.savez("./onnx/CCT_train_16_8_1_1/inputs.npz", input=input_data, labels=labels)
    print("✅ Input Saved to inputs.npz")

    loss, grad = run_onnx_model(input_data, labels, model_path="./onnx/CCT_train_16_8_1_1/network_train.onnx")

    np.savez("./onnx/CCT_train_16_8_1_1/outputs.npz", loss=loss, grad=grad)
    print("✅ Output Saved to outputs.npz")

if __name__ == "__main__":
    main()
