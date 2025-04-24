import copy
import numpy as np
import onnx
import onnxruntime as ort
import argparse
from onnx import TensorProto, helper
import os

def get_initializers(model_path):
    """
    Extract all initializers (weights, biases, etc.) from the ONNX model.
    
    Args:
        model_path: Path to the ONNX model
        
    Returns:
        Dictionary with initializer name as key and numpy array as value
    """
    model = onnx.load(model_path)
    initializers = {}
    
    for initializer in model.graph.initializer:
        name = initializer.name
        # Convert ONNX tensor to numpy array
        np_array = onnx.numpy_helper.to_array(initializer)
        initializers[name] = np_array
        
    return initializers

def get_all_node_outputs(model_path, input_data):
    """
    Extract the output of each node in the ONNX model.
    
    Args:
        model_path: Path to the ONNX model
        input_data: Input data dictionary for inference
        
    Returns:
        Dictionary with node name as key and output as value
    """
    model = onnx.load(model_path)
    all_nodes = model.graph.node
    results = {}
    
    # Create a directory to store temporary models
    temp_dir = "temp_models"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Get input names
    input_names = [input.name for input in model.graph.input]
    
    # Process each node
    for i, node in enumerate(all_nodes):
        node_name = node.name
        print(f"Processing node {i+1}/{len(all_nodes)}: {node_name}")
        
        # Skip nodes without outputs
        if not node.output:
            print(f"  Skipping node {node_name} (no outputs)")
            continue
        
        # Trim the model to this node
        temp_model_path = os.path.join(temp_dir, f"temp_model_{i}.onnx")
        trimmed_model = trim_onnx_model(model_path, node_name, temp_model_path)
        
        try:
            # Try to run inference on the trimmed model
            node_outputs = infer_model(temp_model_path, input_data)
            results[node_name] = node_outputs
        except Exception as e:
            print(f"  Error running inference on node {node_name}: {str(e)}")
    
    # Clean up temporary files
    for file in os.listdir(temp_dir):
        if file.startswith("temp_model_"):
            try:
                os.remove(os.path.join(temp_dir, file))
            except:
                pass
    
    try:
        os.remove("temp_opset20_model.onnx")
        os.rmdir(temp_dir)
    except:
        pass
    
    return results

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
    
    # Remove nodes after the target node
    while len(trimmed_model.graph.node) > target_node_idx + 1:
        trimmed_model.graph.node.pop()
    
    # Set the target node's outputs as the model's outputs
    target_outputs = nodes[target_node_idx].output
    
    new_outputs = []
    for output in target_outputs:
        tensor_type = helper.make_tensor_type_proto(
            elem_type=TensorProto.FLOAT,
            shape=None  # Use dynamic shape
        )
        output_value_info = helper.make_value_info(name=output, type_proto=tensor_type)
        new_outputs.append(output_value_info)
    
    # Clear existing outputs
    while len(trimmed_model.graph.output) > 0:
        trimmed_model.graph.output.pop()
    
    # Add new outputs
    trimmed_model.graph.output.extend(new_outputs)
    
    try:
        onnx.checker.check_model(trimmed_model)
    except Exception as e:
        print(f"  Warning: {str(e)}")
    
    onnx.save(trimmed_model, save_path)
    
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

def save_initializers_to_file(initializers, save_dir="node_outputs"):
    """Save model initializers to files for analysis"""
    initializers_dir = os.path.join(save_dir, "initializers")
    os.makedirs(initializers_dir, exist_ok=True)
    
    # Save a summary file
    with open(os.path.join(initializers_dir, "initializers_summary.txt"), "w") as f:
        f.write("Model Initializers Summary\n")
        f.write("=======================\n\n")
        
        for name, array in initializers.items():
            f.write(f"Initializer: {name}\n")
            f.write(f"  Shape: {array.shape}\n")
            f.write(f"  Type: {array.dtype}\n")
            
            # Add statistical information if the array contains numeric data
            if np.issubdtype(array.dtype, np.number):
                f.write(f"  Min: {np.min(array)}\n")
                f.write(f"  Max: {np.max(array)}\n")
                f.write(f"  Mean: {np.mean(array)}\n")
                f.write(f"  Std: {np.std(array)}\n")
                
                # Count zeros and check if sparse
                zero_count = np.sum(array == 0)
                sparsity = zero_count / array.size
                f.write(f"  Zero elements: {zero_count} ({sparsity:.2%} of total)\n")
            
            f.write("\n")
    
    # Save each initializer
    for name, array in initializers.items():
        # Replace slashes in the filename with underscores
        safe_name = name.replace("/", "_")
        
        # Save as numpy file
        np.save(os.path.join(initializers_dir, f"{safe_name}.npy"), array)
        
        # Save a preview text file
        with open(os.path.join(initializers_dir, f"{safe_name}_preview.txt"), "w") as f:
            f.write(f"Initializer: {name}\n")
            f.write(f"Shape: {array.shape}\n")
            f.write(f"Type: {array.dtype}\n\n")
            
            if np.issubdtype(array.dtype, np.number):
                f.write(f"Min: {np.min(array)}\n")
                f.write(f"Max: {np.max(array)}\n")
                f.write(f"Mean: {np.mean(array)}\n")
                f.write(f"Std: {np.std(array)}\n\n")
            
            # For large arrays, just show a sample
            if array.size > 1000:
                flat_sample = array.flatten()[:1000]
                f.write(f"First 1000 elements (flattened):\n{flat_sample}\n")
            else:
                f.write(f"Full data:\n{array}\n")

def save_outputs_to_file(results, save_dir="node_outputs"):
    """Save node outputs to files for later analysis"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save a summary file
    with open(os.path.join(save_dir, "summary.txt"), "w") as f:
        f.write("Node Output Summary\n")
        f.write("=================\n\n")
        
        for node_name, outputs in results.items():
            f.write(f"Node: {node_name}\n")
            for output_name, output in outputs.items():
                f.write(f"  Output: {output_name}\n")
    
    # Save the actual output arrays
    for node_name, outputs in results.items():
        # Replace slashes in node name with underscores for directory name
        safe_node_name = node_name.replace("/", "_")
        node_dir = os.path.join(save_dir, safe_node_name)
        os.makedirs(node_dir, exist_ok=True)
        
        # Save each output tensor
        for output_name, output in outputs.items():
            # Replace slashes in output name with underscores for file name
            safe_output_name = output_name.replace("/", "_")
            
            # Save as numpy file
            np.save(os.path.join(node_dir, f"{safe_output_name}.npy"), output)
            
            # Save a preview text file with limited data
            with open(os.path.join(node_dir, f"{safe_output_name}_preview.txt"), "w") as f:
                f.write(f"Full data:\n{output}\n")

def print_initializers(initializers, max_elements=20):
    """Print a summary of model initializers to the console"""
    print("\n===== Model Initializers Summary =====")
    
    for name, array in initializers.items():
        print(f"\nInitializer: {name}")
        print(f"  Shape: {array.shape}")
        print(f"  Type: {array.dtype}")
        
        if np.issubdtype(array.dtype, np.number):
            print(f"  Min: {np.min(array)}")
            print(f"  Max: {np.max(array)}")
            print(f"  Mean: {np.mean(array)}")
            print(f"  Std: {np.std(array)}")
            
            # Count zeros to check sparsity
            zero_count = np.sum(array == 0)
            sparsity = zero_count / array.size
            print(f"  Zero elements: {zero_count} ({sparsity:.2%} of total)")
        
        # Print a sample of the data
        print("  Sample data:")
        if array.size > max_elements:
            # For large arrays, reshape to 1D and show first few elements
            flat_array = array.flatten()
            print(f"  {flat_array[:max_elements]} ... (showing first {max_elements} of {array.size} elements)")
        else:
            print(f"  {array}")
        print("")

def print_node_outputs(results, max_elements=20):
    """Print a summary of each node's outputs to the console"""
    print("\n===== Node Outputs Summary =====")
    
    for node_name, outputs in results.items():
        print(f"\nNode: {node_name}")
        
        for output_name, output in outputs.items():
            print(f"  Output: {output_name}")
            
            # Print a sample of the data
            print("    Sample data:")
            print(f"    {output}")
            print("")

def main(model_path, input_path, save_dir="node_outputs", print_to_console=True):
    # Load the model and print all node names
    model = onnx.load(model_path)
    print("Model nodes:")
    for i, node in enumerate(model.graph.node):
        print(f"{i+1}. {node.name}")
    
    # Get and print initializers
    print("\nExtracting initializers...")
    initializers = get_initializers(model_path)
    print(f"Found {len(initializers)} initializers")
    
    # Save initializers to files
    print(f"\nSaving initializers to directory: {save_dir}/initializers")
    save_initializers_to_file(initializers, save_dir)
    
    # Print initializers to console if requested
    if print_to_console:
        print_initializers(initializers)
    
    # Load input data
    print(f"\nLoading input data from: {input_path}")
    input_data = np.load(input_path)
    print(f"Input data keys: {list(input_data.keys())}")
    
    # Get outputs for all nodes
    print("\nExtracting outputs from all nodes...")
    results = get_all_node_outputs(model_path, input_data)
    
    # Save results to files
    print(f"\nSaving outputs to directory: {save_dir}")
    save_outputs_to_file(results, save_dir)
    
    # Print results to console if requested
    if print_to_console:
        print_node_outputs(results)
    
    print(f"\nDone! All node outputs and initializers have been extracted and saved to: {save_dir}")
    print(f"Check {save_dir}/summary.txt for a summary of all node outputs.")
    print(f"Check {save_dir}/initializers/initializers_summary.txt for a summary of all initializers.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract outputs from all nodes and initializers in an ONNX model.")
    parser.add_argument("--model", type=str, default="/app/Onnx4Deeploy/Tests/Models/CCT/onnx/CCT_train_16_8_1_1/network_train.onnx", 
                        help="Path to the ONNX model")
    parser.add_argument("--input", type=str, default="/app/Onnx4Deeploy/Tests/Models/CCT/onnx/CCT_train_16_8_1_1/inputs.npz",
                        help="Path to the input data (NPZ format)")
    parser.add_argument("--output_dir", type=str, default="node_outputs",
                        help="Directory to save the node outputs and initializers")
    parser.add_argument("--no_console_print", action="store_true",
                        help="Disable printing outputs to console")
    
    args = parser.parse_args()
    np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True)

    main(args.model, args.input, args.output_dir, not args.no_console_print)