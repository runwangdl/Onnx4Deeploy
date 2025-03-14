import onnx
import onnxruntime as ort
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper

def generate_softmax_grad_onnx_and_data(save_path=None):
    """ Generate ONNX model for SoftmaxGrad operator with test data """
    
    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config.yaml")
    
    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    model_config = config["softmax_grad"]
    input_shape = tuple(model_config["input_shape"])
    
    # Set default save path if not provided
    base_path = save_path if save_path else os.path.join(script_dir, "onnx")
    
    # Define standard filenames
    onnx_file = os.path.join(base_path, "network.onnx")
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")
    
    # Ensure the save directory exists
    os.makedirs(base_path, exist_ok=True)
    
    # Generate random input data
    # y is the result of a softmax operation (probabilities sum to 1 on last axis)
    y = np.random.rand(*input_shape).astype(np.float32)
    y = y / np.sum(y, axis=-1, keepdims=True)
    
    # dy is the upstream gradient
    dy = np.random.randn(*input_shape).astype(np.float32)
    
    np.savez(input_file, y=y, dy=dy)
    
    # Define ONNX tensors
    y_tensor = helper.make_tensor_value_info("y", TensorProto.FLOAT, input_shape)
    dy_tensor = helper.make_tensor_value_info("dy", TensorProto.FLOAT, input_shape)
    dx_tensor = helper.make_tensor_value_info("dx", TensorProto.FLOAT, input_shape)
    
    # Create ONNX computation graph - using the correct domain and input order
    softmax_grad_node = helper.make_node(
        "SoftmaxGrad",  # Op name
        inputs=["dy", "y"],  # Corrected order: dy first, y second
        outputs=["dx"],
        name="softmax_grad_node",
        domain="com.microsoft",  # Microsoft domain
        axis=-1  # Apply softmax along the last axis
    )
    
    graph_def = helper.make_graph(
        [softmax_grad_node], "softmax_grad_graph",
        [dy_tensor, y_tensor],  # Corrected order in graph definition
        [dx_tensor]
    )
    
    # Include both the default opset and the Microsoft opset
    opset_imports = [
        helper.make_opsetid("", 13),  # Standard ONNX domain
        helper.make_opsetid("com.microsoft", 1)  # Microsoft domain
    ]
    
    model_def = helper.make_model(
        graph_def, 
        producer_name="softmax_grad_model", 
        opset_imports=opset_imports
    )
    
    # Save ONNX model
    onnx.save(model_def, onnx_file)
    print(f"✅ ONNX model saved to {onnx_file}")
    
    # Run inference using ONNX Runtime
    ort_session = ort.InferenceSession(onnx_file)
    output_data = ort_session.run(None, {"dy": dy, "y": y})  # Corrected order
    
    # Save output data
    np.savez(output_file, dx=output_data[0])
    
    print(f"✅ Output data saved to {output_file}")

if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_softmax_grad_onnx_and_data(save_path)