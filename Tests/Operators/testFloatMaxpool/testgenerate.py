import onnx
import onnxruntime as ort
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper

def generate_maxpool_onnx_and_data(save_path=None):
    """ Generate ONNX model for MaxPool operator based on config, with optional save path """
    
    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config.yaml")
    
    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    model_config = config["maxpool"]
    
    input_shape = tuple(model_config["input_shape"])  # N,C,H,W format
    kernel_shape = tuple(model_config["kernel_shape"])  # [kH, kW]
    strides = tuple(model_config["strides"])  # [sH, sW]
    pads = tuple(model_config["pads"]) if "pads" in model_config else [0, 0, 0, 0]  # [top, left, bottom, right]
    ceil_mode = int(model_config["ceil_mode"]) if "ceil_mode" in model_config else 0  # 0 or 1
    
    # Validate input shape for max pooling
    if len(input_shape) != 4:
        raise ValueError("Input shape must be 4D tensor in NCHW format")
    
    if len(kernel_shape) != 2:
        raise ValueError("Kernel shape must be 2D [height, width]")
    
    if len(strides) != 2:
        raise ValueError("Strides must be 2D [height, width]")
    
    if len(pads) != 4:
        raise ValueError("Pads must be [top, left, bottom, right]")
    
    # Calculate output dimensions
    batch_size, channels, height, width = input_shape
    kernel_h, kernel_w = kernel_shape
    stride_h, stride_w = strides
    pad_t, pad_l, pad_b, pad_r = pads
    
    # Calculate output height and width
    if ceil_mode == 0:
        out_height = (height + pad_t + pad_b - kernel_h) // stride_h + 1
        out_width = (width + pad_l + pad_r - kernel_w) // stride_w + 1
    else:
        out_height = (height + pad_t + pad_b - kernel_h + stride_h - 1) // stride_h + 1
        out_width = (width + pad_l + pad_r - kernel_w + stride_w - 1) // stride_w + 1
    output_shape = (batch_size, channels, out_height, out_width)
    
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
    np.savez(input_file, input_data=input_data)
    
    # Define ONNX tensors
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
    
    # Create ONNX computation graph
    maxpool_node = helper.make_node(
        "MaxPool",
        inputs=["input"],
        outputs=["output"],
        kernel_shape=kernel_shape,
        strides=strides,
        pads=pads,  # [top, left, bottom, right]
        ceil_mode=ceil_mode,
        name="maxpool_node"
    )
    
    graph_def = helper.make_graph(
        [maxpool_node],
        "maxpool_graph",
        [input_tensor],
        [output_tensor]
    )
    
    model_def = helper.make_model(
        graph_def,
        producer_name="maxpool_model",
        opset_imports=[helper.make_opsetid("", 13)]
    )
    
    # Save ONNX model
    onnx.save(model_def, onnx_file)
    print(f"✅ ONNX model saved to {onnx_file}")
    
    # Run inference using ONNX Runtime
    ort_session = ort.InferenceSession(onnx_file)
    output_data = ort_session.run(None, {"input": input_data})[0]
    
    # Save output data directly
    np.savez(output_file, output=output_data)
    print(f"✅ Output data saved to {output_file}")

if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_maxpool_onnx_and_data(save_path)