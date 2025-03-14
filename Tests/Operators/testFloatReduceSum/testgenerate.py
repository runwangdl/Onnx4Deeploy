import onnx
import onnxruntime as ort
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper

def generate_reducesum_onnx_and_data(save_path=None):
    """ Generate ONNX model for ReduceSum operator based on config, with optional save path """
    
    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config.yaml")
    
    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    model_config = config["reducesum"]
        
    input_shape = tuple(model_config["input_shape"])
    axes = model_config.get("axes", None)  # Get the axes to reduce, or None for all axes
    keepdims = model_config.get("keepdims", 1)  # Default to keeping dims
    
    # Set default save path if not provided
    base_path = save_path if save_path else os.path.join(script_dir, "onnx")
    
    # Define standard filenames
    onnx_file = os.path.join(base_path, "network.onnx")
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")
    
    # Ensure the save directory exists
    os.makedirs(base_path, exist_ok=True)
    
    # Generate random input data and save it directly
    input_data = np.random.randn(*input_shape).astype(np.float32)
    np.savez(input_file, input=input_data)
    
    # Calculate expected output shape
    output_shape = list(input_shape)
    if axes is not None:
        # Convert negative axes to positive
        normalized_axes = [ax if ax >= 0 else ax + len(input_shape) for ax in axes]
        if keepdims:
            for ax in normalized_axes:
                output_shape[ax] = 1
        else:
            # Remove dimensions
            for ax in sorted(normalized_axes, reverse=True):
                del output_shape[ax]
    else:
        # Reduce all dimensions
        if keepdims:
            output_shape = [1] * len(input_shape)
        else:
            output_shape = []
    
    output_shape = tuple(output_shape)
    
    # Define ONNX tensors
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
    
    # Create ONNX computation graph
    # For opset 11, ReduceSum takes the axes as an attribute, not as a second input
    node_attributes = {"keepdims": keepdims}
    if axes is not None:
        node_attributes["axes"] = axes
    
    reducesum_node = helper.make_node(
        "ReduceSum", 
        inputs=["input"], 
        outputs=["output"], 
        name="reducesum_node",
        **node_attributes
    )
    
    graph_def = helper.make_graph([reducesum_node], "reducesum_graph", [input_tensor], [output_tensor])
    model_def = helper.make_model(graph_def, producer_name="reducesum_model", opset_imports=[helper.make_opsetid("", 11)])
    
    # Save ONNX model
    onnx.save(model_def, onnx_file)
    print(f"✅ ONNX model saved to {onnx_file}")
    
    # Run inference using ONNX Runtime
    ort_session = ort.InferenceSession(onnx_file)
    output_data = ort_session.run(None, {"input": input_data})[0]
    
    # Save output data directly
    np.savez(output_file, output=output_data)
    print(f"✅ Output data saved to {output_file}")
    
    # Print additional information
    print(f"📊 Input shape: {input_shape}")
    print(f"📊 Output shape: {output_data.shape}")
    if axes is not None:
        print(f"📊 Reduction axes: {axes}")
    else:
        print(f"📊 Reduction axes: all")
    print(f"📊 Keep dimensions: {keepdims}")

if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_reducesum_onnx_and_data(save_path)