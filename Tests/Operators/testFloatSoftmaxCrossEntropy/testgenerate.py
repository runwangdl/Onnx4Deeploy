import onnx
import onnxruntime as ort
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper

def generate_softmax_cross_entropy_onnx_and_data(save_path=None):
    """ Generate ONNX model for SoftmaxCrossEntropyLoss operator with additional log_prob output """

    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config.yaml")

    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    model_config = config["softmax_cross_entropy"]
    input_shape = tuple(model_config["input_shape"])
    num_classes = input_shape[-1]
    batch_size = input_shape[0]  

    # Set default save path if not provided
    base_path = save_path if save_path else os.path.join(script_dir, "onnx")

    # Define standard filenames
    onnx_file = os.path.join(base_path, "network.onnx")
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")

    # Ensure the save directory exists
    os.makedirs(base_path, exist_ok=True)

    # Generate random input data
    logits = np.random.randn(*input_shape).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=(batch_size,)).astype(np.int64)  
    np.savez(input_file, logits=logits, labels=labels)

    # Define ONNX tensors
    logits_tensor = helper.make_tensor_value_info("logits", TensorProto.FLOAT, input_shape)
    labels_tensor = helper.make_tensor_value_info("labels", TensorProto.INT64, [batch_size])  
    loss_tensor = helper.make_tensor_value_info("loss", TensorProto.FLOAT, [])  # Scalar loss output
    log_prob_tensor = helper.make_tensor_value_info("log_prob", TensorProto.FLOAT, input_shape)  # Log probability output

    # Create ONNX computation graph
    loss_node = helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=["logits", "labels"],
        outputs=["loss", "log_prob"],  # Now producing two outputs
        name="softmax_cross_entropy_node",
        reduction="mean"  # ✅ Set reduction to mean
    )

    graph_def = helper.make_graph(
        [loss_node], "softmax_cross_entropy_graph", 
        [logits_tensor, labels_tensor], 
        [loss_tensor, log_prob_tensor]
    )
    model_def = helper.make_model(graph_def, producer_name="softmax_cross_entropy_model", opset_imports=[helper.make_opsetid("", 13)])

    # Save ONNX model
    onnx.save(model_def, onnx_file)
    print(f"✅ ONNX model saved to {onnx_file}")

    # Run inference using ONNX Runtime
    ort_session = ort.InferenceSession(onnx_file)
    output_data = ort_session.run(None, {"logits": logits, "labels": labels})

    # Save output data directly
    np.savez(output_file, loss=output_data[0].squeeze(), log_prob=output_data[1])  
    print(f"✅ Output data saved to {output_file}")

if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_softmax_cross_entropy_onnx_and_data(save_path)