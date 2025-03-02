import onnx
import onnxruntime as ort
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper

def generate_adder_onnx_and_data(save_path=None):
    """ Generate ONNX model for Adder operator based on config, with optional save path """

    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config.yaml")

    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    model_config = config["adder"]
    
    input_shape = tuple(model_config["input_shape"])

    # Set default save path if not provided
    base_path = save_path if save_path else os.path.join(script_dir, "onnx")

    # Define standard filenames
    onnx_file = os.path.join(base_path, "network.onnx")
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")

    # Ensure the save directory exists
    os.makedirs(base_path, exist_ok=True)

    # Generate random input data and save it directly
    input_a = np.random.randn(*input_shape).astype(np.float32)
    input_b = np.random.randn(*input_shape).astype(np.float32)
    np.savez(input_file, input_a=input_a, input_b=input_b)

    # Define ONNX tensors
    input_tensor_a = helper.make_tensor_value_info("input_a", TensorProto.FLOAT, input_shape)
    input_tensor_b = helper.make_tensor_value_info("input_b", TensorProto.FLOAT, input_shape)
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, input_shape)

    # Create ONNX computation graph
    add_node = helper.make_node("Add", inputs=["input_a", "input_b"], outputs=["output"], name="add_node")
    graph_def = helper.make_graph([add_node], "adder_graph", [input_tensor_a, input_tensor_b], [output_tensor])
    model_def = helper.make_model(graph_def, producer_name="adder_model", opset_imports=[helper.make_opsetid("", 13)])

    # Save ONNX model
    onnx.save(model_def, onnx_file)
    print(f"✅ ONNX model saved to {onnx_file}")

    # Run inference using ONNX Runtime
    ort_session = ort.InferenceSession(onnx_file)
    output_data = ort_session.run(None, {"input_a": input_a, "input_b": input_b})[0]

    # Save output data directly
    np.savez(output_file, output=output_data)
    print(f"✅ Output data saved to {output_file}")

if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_adder_onnx_and_data(save_path)
