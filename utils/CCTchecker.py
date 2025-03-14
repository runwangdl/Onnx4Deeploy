import copy
import numpy as np
import onnx
import onnxruntime as ort
import argparse
from onnx import TensorProto, helper

def trim_onnx_model(model_path, output_node_name, save_path="trimmed_network.onnx"):
    model = onnx.load(model_path)
    nodes = model.graph.node
    target_node_idx = None
    
    for idx, node in enumerate(nodes):
        if node.name == output_node_name:
            target_node_idx = idx
            break
    
    if target_node_idx is None:
        raise ValueError(f"Cannot find {output_node_name} in the model.")
    
    trimmed_model = copy.deepcopy(model)
    
    while len(trimmed_model.graph.node) > target_node_idx + 1:
        trimmed_model.graph.node.pop()
    
    target_outputs = nodes[target_node_idx].output
    
    new_outputs = []
    for output in target_outputs:
        tensor_type = helper.make_tensor_type_proto(
            elem_type=TensorProto.FLOAT,
            shape=None  # Use dynamic shape
        )
        output_value_info = helper.make_value_info(name=output, type_proto=tensor_type)
        new_outputs.append(output_value_info)
    
    while len(trimmed_model.graph.output) > 0:
        trimmed_model.graph.output.pop()
    
    trimmed_model.graph.output.extend(new_outputs)
    
    try:
        onnx.checker.check_model(trimmed_model)
    except Exception as e:
        print(f"Warning: {str(e)}")
        print("Trying to save model...")
    
    onnx.save(trimmed_model, save_path)
    print(f"Saved: {save_path}")
    
    return trimmed_model

def infer_model(model_path, input_data):
    model = onnx.load(model_path)
    model.opset_import[0].version = 20
    input_names = [input.name for input in model.graph.input]
    temp_model_path = "temp_opset20_model.onnx"
    onnx.save(model, temp_model_path)
    
    session = ort.InferenceSession(temp_model_path, providers=['CPUExecutionProvider'])
    ort_inputs = {name: input_data[name] for name in input_names}
    outputs = session.run(None, ort_inputs)
    
    output_names = [output.name for output in session.get_outputs()]
    return dict(zip(output_names, outputs))

def main(output_node_name, model_path="/app/Onnx4Deeploy/Tests/Models/CCT/onnx/CCT_train_16_8_1_1/network_train.onnx", input_path="/app/Deeploy/DeeployTest/Tests/testTrainCCT/CCT_Classifier_Weight_1/inputs.npz", save_path="trimmed_network.onnx"):
    
    model = onnx.load(model_path)
    for node in model.graph.node:
        print(node.name)
        
    trimmed_model = trim_onnx_model(model_path, output_node_name, save_path)
    
    input_data = np.load(input_path)


    
    result_dict = infer_model(save_path, input_data)
    
    
    np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True)

    output = list(result_dict.values())[0]
    print("\nInput:")
    print(input_data['input'])
    print("\nOutput:")
    print(output)
    # for name, output in result_dict.items():
    #     print(f"{name} Output Shape: {output.shape}")
    #     print(f"{name} Output:\n{output}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim ONNX model and run inference.")
    parser.add_argument("output_node", type=str, help="Name of the output node.")
    
    args = parser.parse_args()
    
    main(args.output_node)
