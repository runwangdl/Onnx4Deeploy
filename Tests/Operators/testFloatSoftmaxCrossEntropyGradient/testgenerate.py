import onnx
import onnxruntime as ort
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper, shape_inference

def generate_softmax_cross_entropy_grad_onnx_and_data(save_path=None):
    """
    Generate ONNX model for SoftmaxCrossEntropyLossGrad operator:
    1. First create complete model with all inputs
    2. Run shape inference and inference on complete model
    3. Modify model to remove loss_grad input
    4. Keep output results from complete model
    """

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
    complete_onnx_file = os.path.join(base_path, "network_complete.onnx")
    inferred_onnx_file = os.path.join(base_path, "network_inferred.onnx")
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
    
    # Save all inputs for complete model
    np.savez(input_file, log_prob=log_prob, labels=labels, loss_grad=loss_grad)

    # Step 1: Create complete ONNX model with all inputs
    # Define ONNX tensors
    loss_grad_tensor = helper.make_tensor_value_info("loss_grad", TensorProto.FLOAT, [batch_size])  
    log_prob_tensor = helper.make_tensor_value_info("log_prob", TensorProto.FLOAT, input_shape)
    labels_tensor = helper.make_tensor_value_info("labels", TensorProto.INT64, [batch_size])  
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
        [loss_grad_tensor, log_prob_tensor, labels_tensor], 
        [output_grad_tensor]
    )
    model_def = helper.make_model(graph_def, producer_name="softmax_cross_entropy_grad_model", opset_imports=[
        helper.make_opsetid("", 13),
        helper.make_opsetid("com.microsoft", 1)
    ])

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
    complete_outputs = complete_session.run(None, {
        "loss_grad": loss_grad,
        "log_prob": log_prob, 
        "labels": labels
    })
    
    print(f"   Complete output shape - output_grad: {complete_outputs[0].shape}")
    
    # Step 4: Create a modified model with loss_grad input removed
    modified_model = onnx.load(inferred_onnx_file)
    
    # Get the node and modify it to remove loss_grad input
    node = modified_model.graph.node[0]
    node.input.pop(0)  # Remove loss_grad input from node
    
    # Remove loss_grad input from graph inputs
    del modified_model.graph.input[0]
    
    # Save the modified model
    onnx.save(modified_model, onnx_file)
    print(f"✅ Modified ONNX model (loss_grad input removed) saved to {onnx_file}")
    
    # Step 5: Directly save output_grad from complete model inference
    np.savez(output_file, output_grad=complete_outputs[0])
    print(f"✅ Output data saved to {output_file}")

if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_softmax_cross_entropy_grad_onnx_and_data(save_path)