import onnx
import onnxruntime as ort
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper

def generate_softmax_cross_entropy_grad_onnx_and_data(save_path=None):
    """ Generate ONNX model for SoftmaxCrossEntropyLossGrad operator with log_prob input """

    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config.yaml")

    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    model_config = config["softmax_cross_entropy_grad"]
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

    # Generate input data ensuring log_prob sums to one per sample
    logits = np.random.randn(*input_shape).astype(np.float32)  # Generate logits
    softmax_prob = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)  # Compute softmax probabilities
    log_prob = np.log(softmax_prob)  # Compute log probabilities
    labels = np.random.randint(0, num_classes, size=(batch_size,)).astype(np.int64)  # Ensure labels are int64
    loss_grad = np.ones((batch_size,), dtype=np.float32) / batch_size 
    np.savez(input_file, log_prob=log_prob, labels=labels, loss_grad=loss_grad)

    # Define ONNX tensors
    log_prob_tensor = helper.make_tensor_value_info("log_prob", TensorProto.FLOAT, input_shape)
    labels_tensor = helper.make_tensor_value_info("labels", TensorProto.INT64, [batch_size])  
    loss_grad_tensor = helper.make_tensor_value_info("loss_grad", TensorProto.FLOAT, [batch_size])  
    output_grad_tensor = helper.make_tensor_value_info("output_grad", TensorProto.FLOAT, input_shape)  

    # SoftmaxCrossEntropyLossGrad node (com.microsoft)
    grad_node = helper.make_node(
        "SoftmaxCrossEntropyLossGrad",
        inputs=["loss_grad", "log_prob", "labels"],
        outputs=["output_grad"],
        name="softmax_cross_entropy_grad_node",
        domain="com.microsoft", 
        reduction="mean"  
    )

    graph_def = helper.make_graph(
        [grad_node],  
        "softmax_cross_entropy_grad_graph",
        [log_prob_tensor, labels_tensor, loss_grad_tensor], 
        [output_grad_tensor]
    )
    model_def = helper.make_model(graph_def, producer_name="softmax_cross_entropy_grad_model", opset_imports=[helper.make_opsetid("", 13)])


    onnx.save(model_def, onnx_file)
    print(f"✅ ONNX model saved to {onnx_file}")

   
    ort_session = ort.InferenceSession(onnx_file)
    output_data = ort_session.run(None, {"log_prob": log_prob, "labels": labels, "loss_grad": loss_grad})[0]

    np.savez(output_file, output_grad=output_data.squeeze())  
    print(f"✅ Output data saved to {output_file}")

if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_softmax_cross_entropy_grad_onnx_and_data(save_path)
