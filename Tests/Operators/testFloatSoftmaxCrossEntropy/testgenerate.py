import onnx
import onnxruntime as ort
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper, shape_inference

def generate_softmax_cross_entropy_onnx_and_data(save_path=None):
    """
    Generate ONNX model for SoftmaxCrossEntropyLoss operator with both outputs,
    perform shape inference on complete model, then remove loss output from the model
    and directly use the log_prob output from the complete model inference.
    """
    
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
    complete_onnx_file = os.path.join(base_path, "network_complete.onnx")
    inferred_onnx_file = os.path.join(base_path, "network_inferred.onnx")
    onnx_file = os.path.join(base_path, "network.onnx")
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")
    
    # Ensure the save directory exists
    os.makedirs(base_path, exist_ok=True)
    
    # Generate random input data
    logits = np.random.randn(*input_shape).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=(batch_size,)).astype(np.int64)
    
    np.savez(input_file, logits=logits, labels=labels)
    
    # Step 1: Create complete ONNX model with both outputs
    logits_tensor = helper.make_tensor_value_info("logits", TensorProto.FLOAT, input_shape)
    labels_tensor = helper.make_tensor_value_info("labels", TensorProto.INT64, [batch_size])
    
    loss_tensor = helper.make_tensor_value_info("loss", TensorProto.FLOAT, [])
    log_prob_tensor = helper.make_tensor_value_info("log_prob", TensorProto.FLOAT, input_shape)
    
    loss_node = helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=["logits", "labels"],
        outputs=["loss", "log_prob"],
        name="softmax_cross_entropy_node",
        reduction="mean"
    )
    
    graph_def = helper.make_graph(
        [loss_node], "softmax_cross_entropy_graph",
        [logits_tensor, labels_tensor],
        [loss_tensor, log_prob_tensor]
    )
    
    model_def = helper.make_model(graph_def, producer_name="softmax_cross_entropy_model", opset_imports=[helper.make_opsetid("", 13)])
    
    # Save complete model
    onnx.save(model_def, complete_onnx_file)
    print(f"✅ Complete ONNX model saved to {complete_onnx_file}")
    
    # Step 2: Perform shape inference on complete model
    print("🔄 Performing shape inference on complete model...")
    inferred_model = shape_inference.infer_shapes(model_def)
    onnx.save(inferred_model, inferred_onnx_file)
    print(f"✅ Shape-inferred model saved to {inferred_onnx_file}")
    
    # Step 3: Run inference on complete model
    print("🔄 Running inference on complete model...")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    
    complete_session = ort.InferenceSession(inferred_onnx_file, sess_options)
    complete_outputs = complete_session.run(None, {"logits": logits, "labels": labels})
    
    print(f"   Complete outputs shapes - loss: {complete_outputs[0].shape}, log_prob: {complete_outputs[1].shape}")
    
    # Step 4: Create a modified model with loss output removed
    modified_model = onnx.load(inferred_onnx_file)
    
    # Get the node and modify it to remove loss output
    node = modified_model.graph.node[0]
    node.output.pop(0)  # Remove loss output from node
    
    # Remove loss output from graph outputs
    del modified_model.graph.output[0]
    
    # Save the modified model
    onnx.save(modified_model, onnx_file)
    print(f"✅ Modified ONNX model (loss removed) saved to {onnx_file}")
    
    # Step 5: Directly save log_prob output from complete model inference
    # We skip running inference on the modified model and directly use results from the complete model
    np.savez(output_file, log_prob=complete_outputs[1])
    print(f"✅ Output data (log_prob only) saved to {output_file}")

if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_softmax_cross_entropy_onnx_and_data(save_path)