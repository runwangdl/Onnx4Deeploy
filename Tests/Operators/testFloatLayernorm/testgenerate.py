import onnx
import onnxruntime as ort
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper

def generate_layernorm_onnx_and_data(save_path=None):
    """ Generate ONNX model for LayerNorm operator based on config, with optional save path """
    
    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config.yaml")
    
    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    model_config = config["layernorm"]
        
    input_shape = tuple(model_config["input_shape"])
    axis = model_config.get("axis", -1)  # Default to last dimension
    epsilon = model_config.get("epsilon", 1e-5)  # Default epsilon value
    
    # Set default save path if not provided
    base_path = save_path if save_path else os.path.join(script_dir, "onnx")
    
    # Define standard filenames
    onnx_file = os.path.join(base_path, "network.onnx")
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")
    
    # Ensure the save directory exists
    os.makedirs(base_path, exist_ok=True)
    
    # Convert negative axis to positive
    norm_axis = axis if axis >= 0 else axis + len(input_shape)
    
    # Calculate shapes for scale and bias
    # They should match the normalized axis dimension
    params_shape = [input_shape[norm_axis]]
    
    # Generate random input data
    input_data = np.random.randn(*input_shape).astype(np.float32)
    
    # Generate random scale (gamma) and bias (beta) parameters
    scale_data = np.random.randn(*params_shape).astype(np.float32)
    bias_data = np.random.randn(*params_shape).astype(np.float32)
    
    # Save input data
    np.savez(input_file, input=input_data, scale=scale_data, bias=bias_data)
    
    # Calculate expected output shapes
    # For normalized output, it's the same as input
    output_shape = input_shape
    
    # For mean and invstddev, reduce along the normalization axis
    mean_invstddev_shape = list(input_shape)
    mean_invstddev_shape[norm_axis] = 1
    mean_invstddev_shape = tuple(mean_invstddev_shape)
    
    # Define ONNX tensors
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
    scale_tensor = helper.make_tensor_value_info("scale", TensorProto.FLOAT, params_shape)
    bias_tensor = helper.make_tensor_value_info("bias", TensorProto.FLOAT, params_shape)
    
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
    mean_tensor = helper.make_tensor_value_info("mean", TensorProto.FLOAT, mean_invstddev_shape)
    invstddev_tensor = helper.make_tensor_value_info("invstddev", TensorProto.FLOAT, mean_invstddev_shape)
    
    # Create a single combined LayerNorm node
    layernorm_node = helper.make_node(
        "LayerNormalization",  # Using the ONNX standard name for this op
        inputs=["input", "scale", "bias"],
        outputs=["output", "mean", "invstddev"],
        axis=axis,
        epsilon=epsilon,
        name="layernorm_node"
    )
    
    # Collect all nodes
    nodes = [layernorm_node]
    
    # Create the graph
    graph_def = helper.make_graph(
        nodes, 
        "layernorm_graph", 
        [input_tensor, scale_tensor, bias_tensor], 
        [output_tensor, mean_tensor, invstddev_tensor]
    )
    
    model_def = helper.make_model(
        graph_def, 
        producer_name="layernorm_model", 
        opset_imports=[helper.make_opsetid("", 17)]  # Using ONNX opset 17 which supports LayerNormalization
    )
    
    # Save ONNX model
    onnx.save(model_def, onnx_file)
    print(f"✅ ONNX model saved to {onnx_file}")
    
    # Use ONNX Runtime to directly generate the reference outputs
    # This ensures the outputs match the ONNX standard exactly
    session = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
    
    # Run inference with ONNX Runtime
    outputs = session.run(
        ["output", "mean", "invstddev"],
        {
            "input": input_data,
            "scale": scale_data,
            "bias": bias_data
        }
    )
    
    # Get the results
    output_data, mean_data, invstddev_data = outputs
    
    # Save output data
    np.savez(output_file, output=output_data, mean=mean_data, invstddev=invstddev_data)
    print(f"✅ Output data saved to {output_file}")
    
    # Print additional information
    print(f"📊 Input shape: {input_shape}")
    print(f"📊 Scale shape: {scale_data.shape}")
    print(f"📊 Bias shape: {bias_data.shape}")
    print(f"📊 Output shape: {output_data.shape}")
    print(f"📊 Mean shape: {mean_data.shape}")
    print(f"📊 InvStdDev shape: {invstddev_data.shape}")
    print(f"📊 Normalization axis: {axis}")
    print(f"📊 Epsilon: {epsilon}")
    
    print("\n✅ Reference outputs generated using ONNX Runtime with the official")
    print("  'LayerNormalization' operator from opset 17. The third output is the")
    print("  inverse standard deviation, as per the ONNX standard.")

if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_layernorm_onnx_and_data(save_path)