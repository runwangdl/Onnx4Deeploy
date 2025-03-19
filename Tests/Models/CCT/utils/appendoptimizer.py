import onnx
from onnx import helper
from onnx import TensorProto

def add_sgd_nodes(model_path, output_path):
    # Load the ONNX model
    model = onnx.load(model_path)
    graph = model.graph
    
    # Get reference to the output nodes
    node_133_output = None
    classifier_fc_weight_grad = None
    
    for output in graph.output:
        if output.name == "node_133_classifier_fc_Gemm_Grad_dC_reduced":
            node_133_output = output
        elif output.name == "classifier_fc_weight_grad":
            classifier_fc_weight_grad = output
    
    if not node_133_output or not classifier_fc_weight_grad:
        raise ValueError("Could not find required output nodes")
    
    # Find the original classifier_fc_Gemm node to get initializers
    classifier_fc_weight = None
    classifier_fc_bias = None
    
    for initializer in graph.initializer:
        if initializer.name == "classifier_fc_weight":
            classifier_fc_weight = initializer
        elif initializer.name == "classifier_fc_bias":
            classifier_fc_bias = initializer
    
    if not classifier_fc_weight or not classifier_fc_bias:
        raise ValueError("Could not find required initializers")
    
    # Create SGD node for the weight gradient
    sgd_weight_node = helper.make_node(
        op_type="SGD",
        inputs=[
            "classifier_fc_weight",           # weights to update
            "classifier_fc_weight_grad",      # gradient
            "learning_rate",                  # learning rate (should be an initializer)
        ],
        outputs=["classifier_fc_weight_updated"],
        name="classifier_fc_weight_sgd",
        domain=""
    )
    
    # Create SGD node for the bias gradient
    sgd_bias_node = helper.make_node(
        op_type="SGD",
        inputs=[
            "classifier_fc_bias",             # bias to update
            "node_133_classifier_fc_Gemm_Grad_dC_reduced",  # gradient
            "learning_rate",                  # learning rate 
        ],
        outputs=["classifier_fc_bias_updated"],
        name="classifier_fc_bias_sgd",
        domain=""
    )
    
    # Create learning rate initializer if it doesn't exist
    initializers_to_add = []
    
    # Check if learning rate initializer exists, otherwise create it
    lr_exists = any(init.name == "learning_rate" for init in graph.initializer)
    if not lr_exists:
        learning_rate = helper.make_tensor(
            name="learning_rate",
            data_type=TensorProto.FLOAT,
            dims=[],
            vals=[0.01]  # Default learning rate
        )
        initializers_to_add.append(learning_rate)
    
    # Add the new SGD nodes to the graph
    graph.node.extend([sgd_weight_node, sgd_bias_node])
    
    # Add the new initializers to the graph
    graph.initializer.extend(initializers_to_add)
    
    # Add the updated weights and biases as outputs
    updated_weight_output = helper.make_value_info(
        name="classifier_fc_weight_updated",
        type_proto=node_133_output.type
    )
    
    updated_bias_output = helper.make_value_info(
        name="classifier_fc_bias_updated",
        type_proto=classifier_fc_weight_grad.type
    )
    
    graph.output.extend([updated_weight_output, updated_bias_output])
    
    # Save the modified model
    onnx.save(model, output_path)
    print(f"Modified model saved to {output_path}")

# Example usage
if __name__ == "__main__":
    add_sgd_nodes("/app/Onnx4Deeploy/Tests/Models/CCT/onnx/CCT_train_16_8_1_1/network.onnx", "/app/Onnx4Deeploy/Tests/Models/CCT/onnx/CCT_train_16_8_1_1/network_sgd.onnx")