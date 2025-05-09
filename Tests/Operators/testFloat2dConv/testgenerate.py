import onnx
import onnxruntime as ort
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper

def generate_conv2d_onnx_and_data(save_path=None):
    """ Generate ONNX model for Conv2D operator based on config, with optional save path """

    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config.yaml")

    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    model_config = config["conv2d"]

    # Extract parameters from config
    input_shape = tuple(model_config["input_shape"])
    kernel_size = model_config["kernel_size"]
    stride = model_config["stride"]
    padding = model_config["padding"]
    out_channels = model_config["out_channels"]
    use_bias = model_config["use_bias"]
    group = model_config["group"]
    dilation = model_config["dilation"]

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

    batch_size, in_channels, height, width = input_data.shape

    # Compute output shape
    effective_kernel = (kernel_size - 1) * dilation + 1
    output_height = (height + 2 * padding - effective_kernel) // stride + 1
    output_width = (width + 2 * padding - effective_kernel) // stride + 1
    output_shape = (batch_size, out_channels, output_height, output_width)

    # Create weight and bias
    weight_shape = (out_channels, in_channels // group, kernel_size, kernel_size)
    weight = np.random.randn(*weight_shape).astype(np.float32)
    bias = np.random.randn(out_channels).astype(np.float32) if use_bias else None

    weight_hwio = weight.transpose(2, 3, 1, 0)  # [kH, kW, Cin, Cout]
    weight_onnx = weight_hwio.transpose(3, 2, 0, 1)  # 回到[Cout, Cin, kH, kW]

    # Define ONNX tensors
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_data.shape)
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
    weight_tensor = helper.make_tensor("weight", TensorProto.FLOAT, weight.shape, weight_onnx.flatten())

    initializers = [weight_tensor]

    # Include bias if enabled
    if use_bias:
        bias_tensor = helper.make_tensor("bias", TensorProto.FLOAT, bias.shape, bias.flatten())
        initializers.append(bias_tensor)
        conv_inputs = ["input", "weight", "bias"]
    else:
        conv_inputs = ["input", "weight"]

    # Define Conv2D ONNX node
    conv_node = helper.make_node(
        "Conv",
        inputs=conv_inputs,
        outputs=["output"],
        name="conv_node",
        kernel_shape=[kernel_size, kernel_size],
        strides=[stride, stride],
        pads=[padding, padding, padding, padding],
        group=group,
        dilations=[dilation, dilation]
    )

    # Create ONNX graph
    graph_def = helper.make_graph(
        [conv_node],
        "conv_graph",
        [input_tensor],
        [output_tensor],
        initializer=initializers
    )

    # Create ONNX model
    model_def = helper.make_model(graph_def, producer_name="conv_model", opset_imports=[helper.make_opsetid("", 21)])

    # Save ONNX model
    onnx.save(model_def, onnx_file)
    print(f"✅ ONNX model saved to {onnx_file}")

    # Run inference using ONNX Runtime
    ort_session = ort.InferenceSession(onnx_file)
    ort_inputs = {"input": input_data}
    output_data = ort_session.run(None, ort_inputs)[0]

    # Save output data directly
    np.savez(output_file, output=output_data)
    print(f"✅ Output data saved to {output_file}")

if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_conv2d_onnx_and_data(save_path)
