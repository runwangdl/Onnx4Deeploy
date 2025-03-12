import onnx
import os
import re
import subprocess
import yaml
from onnx import helper, numpy_helper, shape_inference
import numpy as np
import copy

def add_c_to_gemm(input_model_path, output_model_path):
    model = onnx.load(input_model_path)
    graph = model.graph
    
    for node in graph.node:
        if node.op_type == 'Gemm':

            if len(node.input) == 2:
                print(f"Find Gemm without C: {node.name}")
                
                input_a_name = node.input[0]
                input_b_name = node.input[1]
                
    
                b_shape = None
                for init in graph.initializer:
                    if init.name == input_b_name:
                        b_tensor = numpy_helper.to_array(init)
                        b_shape = b_tensor.shape
                        break
                

                if b_shape is None:
                    for vi in graph.value_info:
                        if vi.name == input_b_name:
                            b_shape = [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
                            break
      
                if b_shape is None:
                    output_name = node.output[0]
                    for vi in graph.value_info + [graph.output[i] for i in range(len(graph.output))]:
                        if vi.name == output_name:
                            output_shape = [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
                          
                            transB = 0
                            for attr in node.attribute:
                                if attr.name == 'transB' and attr.i == 1:
                                    transB = 1
                            
                         
                            c_length = output_shape[-1]
                            b_shape = [c_length, 0] if transB == 0 else [0, c_length]
                            break
                
                if b_shape is not None:
        
                    transB = 0
                    for attr in node.attribute:
                        if attr.name == 'transB' and attr.i == 1:
                            transB = 1
                    
                    c_shape = [b_shape[1]] if not transB else [b_shape[0]]
       
                    c_tensor = np.zeros(c_shape, dtype=np.float32)
                    c_name = f"{node.name}_c_bias"
                    
                    c_initializer = numpy_helper.from_array(c_tensor, name=c_name)
                    graph.initializer.append(c_initializer)
                    
                  
                    node.input.append(c_name)
                    print(f"Add C: {c_name}, Shape: {c_shape}")
                else:
                    print(f"Warning: Cannot find {node.name} shape, pass this node.")
    

    onnx.save(model, output_model_path)
    print(f"Saved to: {output_model_path}")

def replace_biasgelu_with_gelu_add(input_model_path, output_model_path):
 
    model = onnx.load(input_model_path)
    
    # Collect all value_info entries by name for easy lookup
    value_info_map = {}
    for vi in model.graph.value_info:
        value_info_map[vi.name] = vi
    
    # Add input and output value_info to the map
    for inp in model.graph.input:
        value_info_map[inp.name] = inp
    
    for out in model.graph.output:
        value_info_map[out.name] = out
    
    # Create new node list and value_info list
    new_nodes = []
    new_value_info = []
    biasgelu_count = 0
    
    # Counter for generating unique names
    unique_id = 0
    def get_unique_name(prefix):
        nonlocal unique_id
        name = f"{prefix}_{unique_id}"
        unique_id += 1
        return name
    
    # Process all nodes
    for node in model.graph.node:
        if node.op_type == 'BiasGelu':
            biasgelu_count += 1
            
            # Get BiasGelu inputs and outputs
            input_name = node.input[0]  # X
            bias_name = node.input[1]   # Bias
            output_name = node.output[0]  # Y
            
            # Generate unique name prefix
            prefix = node.name if node.name else f"gelu_add"
            
            # Step 1: First apply Add operation to add bias
            add_output = get_unique_name(f"{prefix}_add_out")
            add_node = helper.make_node(
                'Add',
                inputs=[input_name, bias_name],
                outputs=[add_output],
                name=f"{prefix}_add"
            )
            new_nodes.append(add_node)
            
            # Create value_info for add_output with proper type and shape
            # Use the same type and shape as the input tensor if available
            if input_name in value_info_map:
                input_value_info = value_info_map[input_name]
                add_output_value_info = helper.make_tensor_value_info(
                    add_output,
                    input_value_info.type.tensor_type.elem_type,
                    [d.dim_value if d.dim_value else d.dim_param for d in input_value_info.type.tensor_type.shape.dim]
                )
                new_value_info.append(add_output_value_info)
                value_info_map[add_output] = add_output_value_info
            
            # Step 2: Then apply Gelu activation function
            gelu_node = helper.make_node(
                'Gelu',
                inputs=[add_output],
                outputs=[output_name],
                name=f"{prefix}_gelu"
            )
            new_nodes.append(gelu_node)
            
            # If we have output value_info, make sure it's preserved
            # Otherwise, create it with the same shape and type as the input to Gelu
            if output_name not in value_info_map and add_output in value_info_map:
                output_value_info = helper.make_tensor_value_info(
                    output_name,
                    value_info_map[add_output].type.tensor_type.elem_type,
                    [d.dim_value if d.dim_value else d.dim_param for d in value_info_map[add_output].type.tensor_type.shape.dim]
                )
                new_value_info.append(output_value_info)
                value_info_map[output_name] = output_value_info
        else:
            # Keep other nodes unchanged
            new_nodes.append(node)
    
    print(f"Replaced {biasgelu_count} BiasGelu nodes with Gelu+Add combinations")
    
    # Create new graph with all collected value_info
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=model.graph.initializer,
        value_info=list(model.graph.value_info) + new_value_info
    )
    
    # Build new model, preserving original model metadata
    new_model = helper.make_model(
        new_graph,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        domain=model.domain,
        model_version=model.model_version,
        doc_string=model.doc_string
    )
    
    # Copy opset imports
    del new_model.opset_import[:]
    new_model.opset_import.extend(model.opset_import)
    
    # Add Microsoft domain if not present (for Gelu)
    has_ms_domain = any(opset.domain == "com.microsoft" for opset in new_model.opset_import)
    if not has_ms_domain:
        ms_opset = helper.make_opsetid("com.microsoft", 1)
        new_model.opset_import.append(ms_opset)
    
    # Copy IR version
    new_model.ir_version = model.ir_version
    
    # Run shape inference to ensure all shapes are properly defined
    try:
        new_model = shape_inference.infer_shapes(new_model)
        print("Shape inference successful")
    except Exception as e:
        print(f"Warning: Shape inference failed: {e}")
    
    # Skip validation and directly save if needed
    try:
        onnx.checker.check_model(new_model)
        print("Model validation successful!")
    except Exception as e:
        print(f"Warning: Model validation failed, but still saving: {e}")
    
    # Save the modified model
    onnx.save(new_model, output_model_path)
    print(f"Saved modified model to {output_model_path}")
    
    return new_model

def fix_layernorm_version(input_model_path, output_model_path, target_opset=13):
    model = onnx.load(input_model_path)

    current_opset = None
    for opset in model.opset_import:
        if opset.domain == "" or opset.domain == "ai.onnx":  
            current_opset = opset.version
            break
    
    print(f"Current Opset: {current_opset}, Target opset: {target_opset}")
    
    if current_opset is None or current_opset <= target_opset:
        print("No need to change")
        onnx.save(model, output_model_path)
        return model
    
    for opset in model.opset_import:
        if opset.domain == "" or opset.domain == "ai.onnx": 
            opset.version = target_opset
    
    modified_nodes = []
    modified_count = 0
    
    for node in model.graph.node:
        if node.op_type == 'LayerNormalization':
            modified_count += 1
            
            new_node = helper.make_node(
                op_type='LayerNormalization',
                inputs=list(node.input),
                outputs=list(node.output[:1]), 
                name=node.name
            )
            
    
            attrs_to_copy = {'epsilon', 'axis'}
            for attr in node.attribute:
                if attr.name in attrs_to_copy:
                    new_node.attribute.append(attr)
            
     
            has_epsilon = any(attr.name == 'epsilon' for attr in new_node.attribute)
            has_axis = any(attr.name == 'axis' for attr in new_node.attribute)
            
        
            if not has_epsilon:
                epsilon_attr = helper.make_attribute('epsilon', 1e-5)
                new_node.attribute.append(epsilon_attr)
            
        
            if not has_axis:
                axis_attr = helper.make_attribute('axis', -1)
                new_node.attribute.append(axis_attr)
            
            modified_nodes.append(new_node)
        else:
            modified_nodes.append(node)
    
    print(f"Fixed {modified_count} LayerNormalization for {target_opset}")
    
    outputs_to_keep = set()
    for node in modified_nodes:
        for output in node.output:
            outputs_to_keep.add(output)
    
    new_outputs = []
    for output in model.graph.output:
        if output.name in outputs_to_keep:
            new_outputs.append(output)
        else:
            print(f"Removed: {output.name}")
    
    new_graph = helper.make_graph(
        nodes=modified_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=new_outputs if new_outputs else model.graph.output, 
        initializer=model.graph.initializer
    )
    

    for vi in model.graph.value_info:
        new_graph.value_info.append(vi)
    

    new_model = helper.make_model(
        new_graph,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        domain=model.domain,
        model_version=model.model_version,
        doc_string=model.doc_string
    )
    

    new_model.ir_version = model.ir_version
    

    try:
        onnx.checker.check_model(new_model)
        print("Verified！")
    except Exception as e:
        print(f"Warning:Not Verified {e}")
    
    onnx.save(new_model, output_model_path)
    print(f"Saved {output_model_path}")
    


def modify_conflict_outputs(input_model_path, output_model_path):
    model = onnx.load(input_model_path)
    graph = model.graph
    
    select_nodes = []
    for node in graph.node:
        if node.op_type == 'LayerNormalization' or node.op_type == 'MaxPool':
            select_nodes.append(node)
    
    print(f"Find {len(select_nodes)} Layernorm or Maxpool")
    
    outputs_to_remove = []
    
    new_nodes = []
    
    for node in graph.node:
        if (node.op_type == 'LayerNormalization' or node.op_type == 'MaxPool') and len(node.output) > 1:
            outputs_to_remove.extend(node.output[1:])
            
            new_node = onnx.NodeProto()
            new_node.CopyFrom(node)
            first_output = node.output[0]
            
            del new_node.output[:]
            new_node.output.append(first_output)
            
            new_nodes.append(new_node)
        else:
            new_nodes.append(node)
    
    del graph.node[:]
    graph.node.extend(new_nodes)
    
    new_outputs = []
    for output in graph.output:
        if output.name not in outputs_to_remove:
            new_outputs.append(output)
    
    del graph.output[:]
    graph.output.extend(new_outputs)
    
    onnx.save(model, output_model_path)
    print(f"Saved to: {output_model_path}")
    
def convert_squeeze_input_to_attr(input_model_path, output_model_path):
    model = onnx.load(input_model_path)
    
    modified_nodes = []
    modified_count = 0
    
    initializers = {init.name: init for init in model.graph.initializer}
    
    for node in model.graph.node:
        if node.op_type == 'Squeeze' and len(node.input) > 1:
            modified_count += 1
            
            data_input = node.input[0]
            
            axes_input_name = node.input[1]
            
            if axes_input_name in initializers:
                axes_initializer = initializers[axes_input_name]
                axes_np = numpy_helper.to_array(axes_initializer)
                axes_list = axes_np.tolist()
                
                new_node = helper.make_node(
                    op_type='Squeeze',
                    inputs=[data_input],  
                    outputs=list(node.output),
                    name=node.name,
                    axes=axes_list 
                )
                
            
                for attr in node.attribute:
                    if attr.name != 'axes': 
                        new_node.attribute.append(attr)
                
                modified_nodes.append(new_node)
            else:
               
                print(f"Warning: Cannot find '{node.name}' axes initializer. Keep the original node.")
                modified_nodes.append(node)
        else:
          
            modified_nodes.append(node)
    
    print(f"Modified {modified_count} Squeeze")
    
    unused_initializers = set()
    for node in modified_nodes:
        if node.op_type == 'Squeeze' and len(node.input) == 1:
        
            for init in model.graph.initializer:
                if init.name not in [inp for node in modified_nodes for inp in node.input]:
                    unused_initializers.add(init.name)
    

    new_graph = helper.make_graph(
        nodes=modified_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=[init for init in model.graph.initializer if init.name not in unused_initializers]
    )
    
   
    for vi in model.graph.value_info:
        new_graph.value_info.append(vi)
    
   
    new_model = helper.make_model(
        new_graph,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        domain=model.domain,
        model_version=model.model_version,
        doc_string=model.doc_string
    )
    

    new_model.ir_version = model.ir_version
    new_model.opset_import.extend(model.opset_import)
    

    onnx.save(new_model, output_model_path)
    print(f"Saved to {output_model_path}")
    
    return new_model

def run_optmization_remove_biasgelu(onnx_train_file, onnx_out_file):
    """
    Replace BiasGelu operations with Add+Gelu while maintaining shape consistency.
    
    Args:
        onnx_train_file: Path to input ONNX model file
        onnx_out_file: Path to output ONNX model file
    """
    # Load the model
    model = onnx.load(onnx_train_file)
    graph = model.graph
    
    # Create new nodes list to replace the old ones
    new_nodes = []
    replaced_count = 0
    
    # Process all nodes
    for node in graph.node:
        if node.op_type == "BiasGelu":
            print(f"🔄 Replacing BiasGeluFusion: {node.name}")
            replaced_count += 1
            
            # Get input and output tensors
            X, Bias = node.input
            output = node.output[0]
            
            # Create intermediate tensor name
            intermediate_output = f"{X}_add_bias"
            
            # Create Add node
            add_node = helper.make_node(
                "Add",
                inputs=[X, Bias],
                outputs=[intermediate_output],
                name=f"{node.name}_Add",
            )
            
            # Create Gelu node
            gelu_node = helper.make_node(
                "Gelu",
                inputs=[intermediate_output],
                outputs=node.output,
                name=f"{node.name}_Gelu",
            )
            
            # Add shape information for the intermediate tensor
            # Try to find X's shape info
            X_shape = None
            X_type = 1  # Default to FLOAT
            
            # Look for X in inputs, outputs, value_info, or initializers
            for info in graph.input:
                if info.name == X:
                    X_shape = [d.dim_value if d.HasField("dim_value") else -1 for d in info.type.tensor_type.shape.dim]
                    X_type = info.type.tensor_type.elem_type
                    break
                    
            if X_shape is None:
                for info in graph.value_info:
                    if info.name == X:
                        X_shape = [d.dim_value if d.HasField("dim_value") else -1 for d in info.type.tensor_type.shape.dim]
                        X_type = info.type.tensor_type.elem_type
                        break
            
            # Add value_info for intermediate tensor
            if X_shape:
                value_info = helper.make_tensor_value_info(
                    intermediate_output,
                    X_type,
                    X_shape
                )
                graph.value_info.append(value_info)
            
            # Add the new nodes to our list
            new_nodes.extend([add_node, gelu_node])
        else:
            # Keep other nodes unchanged
            new_nodes.append(node)
    
    # Replace nodes in the graph
    graph.ClearField("node")
    graph.node.extend(new_nodes)
    
    # Create a copy of the model for selective shape inference
    safe_model = copy.deepcopy(model)
    
    # Remove Microsoft custom operators that might cause shape inference to fail
    ms_nodes = []
    for node in safe_model.graph.node:
        if node.domain == "com.microsoft":
            ms_nodes.append(node)
    
    if ms_nodes:
        print(f"⚠️ Found {len(ms_nodes)} Microsoft custom operators that might affect shape inference")
        
        # Try to run shape inference on the modified model without MS operators
        try:
            # Create a temporary graph without Microsoft operators
            temp_model = copy.deepcopy(safe_model)
            temp_graph = temp_model.graph
            
            # Remove Microsoft custom operators
            temp_nodes = [node for node in temp_graph.node if node.domain != "com.microsoft"]
            temp_graph.ClearField("node")
            temp_graph.node.extend(temp_nodes)
            
            # Run shape inference on this simplified model
            inferred_model = shape_inference.infer_shapes(temp_model)
            
            # Collect inferred shapes for our new nodes
            inferred_value_infos = {}
            for value_info in inferred_model.graph.value_info:
                inferred_value_infos[value_info.name] = value_info
            
            # Update the original model with any newly inferred shapes
            for name, value_info in inferred_value_infos.items():
                # Skip if already exists
                if any(info.name == name for info in model.graph.value_info):
                    continue
                
                model.graph.value_info.append(value_info)
                
            print("✅ Partial shape inference completed for non-Microsoft operators")
        except Exception as e:
            print(f"⚠️ Partial shape inference failed: {e}")
    else:
        # No Microsoft operators, try regular shape inference
        try:
            model = shape_inference.infer_shapes(model)
            print("✅ Shape inference completed successfully")
        except Exception as e:
            print(f"⚠️ Shape inference failed: {e}")
    
    # Save the modified model
    onnx.save(model, onnx_out_file)
    
    if replaced_count > 0:
        print(f"✅ Successfully replaced {replaced_count} BiasGelu nodes with Add + GELU.")
    else:
        print("⚠️ No BiasGelu nodes were replaced.")
    
    return model

def optimize_reshape_fusion(input_model_path: str, output_model_path: str) -> None:
    """
    Optimize ONNX model by fusing consecutive Reshape operations.
    
    Args:
        input_model_path: Path to the input ONNX model
        output_model_path: Path where the optimized ONNX model will be saved
    """
    print(f"Loading model: {input_model_path}")
    model = onnx.load(input_model_path)
    
    # Create mapping from node name to node
    node_map = {}
    for node in model.graph.node:
        node_map[node.name] = node
    
    # Create mapping from input name to producing node
    input_to_node = {}
    for node in model.graph.node:
        for output in node.output:
            input_to_node[output] = node
    
    # Create mapping from output name to consuming nodes
    output_to_nodes = {}
    for node in model.graph.node:
        for input_name in node.input:
            if input_name not in output_to_nodes:
                output_to_nodes[input_name] = []
            output_to_nodes[input_name].append(node)
    
    # Find all Reshape nodes
    reshape_nodes = [node for node in model.graph.node if node.op_type == "Reshape"]
    
    # Track nodes to be removed by index rather than node objects
    # This avoids the "unhashable type: 'NodeProto'" error
    nodes_to_remove_indices = []
    
    # Track value info to keep
    value_info_to_keep = set(vi.name for vi in model.graph.value_info)
    
    # For each Reshape node, check if its input is also from a Reshape node
    for reshape_node in reshape_nodes:
        # Get the input of the current Reshape node
        input_name = reshape_node.input[0]
        
        # Check if the input comes from another Reshape operation
        if input_name in input_to_node and input_to_node[input_name].op_type == "Reshape":
            previous_reshape = input_to_node[input_name]
            
            # Check if the previous Reshape is only used by the current Reshape
            if input_name in output_to_nodes and len(output_to_nodes[input_name]) == 1:
                print(f"Found fusible Reshape pair: {previous_reshape.name} -> {reshape_node.name}")
                
                # Get the shape tensors for both Reshape nodes
                prev_shape_tensor_name = previous_reshape.input[1]
                current_shape_tensor_name = reshape_node.input[1]
                
                # Modify the current Reshape node to connect directly to the input of the previous Reshape
                reshape_node.input[0] = previous_reshape.input[0]
                
                # Mark the previous Reshape node for removal by its index
                for i, node in enumerate(model.graph.node):
                    if (node.name == previous_reshape.name and 
                        node.op_type == previous_reshape.op_type and
                        node.input == previous_reshape.input and 
                        node.output == previous_reshape.output):
                        nodes_to_remove_indices.append(i)
                        break
                
                # Intermediate value info doesn't need to be kept
                if input_name in value_info_to_keep:
                    value_info_to_keep.remove(input_name)
    
    # Handle custom nodes from Microsoft
    # Since Microsoft nodes might have a different structure or behavior
    # We need to be careful when dealing with them
    custom_nodes = [node for node in model.graph.node if node.domain.startswith('com.microsoft')]
    print(f"Found {len(custom_nodes)} Microsoft custom nodes. These will be preserved.")
    
    # Create a new graph excluding the nodes to be removed
    new_nodes = []
    for i, node in enumerate(model.graph.node):
        if i not in nodes_to_remove_indices:
            new_nodes.append(node)
    
    # Create a new value info list, keeping only the needed value info
    new_value_info = [vi for vi in model.graph.value_info if vi.name in value_info_to_keep]
    
    # Create a new graph
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=model.graph.initializer,
        value_info=new_value_info
    )
    
    # Create a new model
    new_model = helper.make_model(
        new_graph, 
        producer_name="ONNX Reshape Fusion Optimizer",
        ir_version=model.ir_version,
        opset_imports=model.opset_import
    )
    
    # Preserve custom opsets from the original model
    new_model.opset_import.extend([opset for opset in model.opset_import if opset.domain.startswith('com.microsoft')])
    
    # Save the optimized model
    onnx.save(new_model, output_model_path)
    
    # Print statistics
    print(f"Original model node count: {len(model.graph.node)}")
    print(f"Optimized model node count: {len(new_model.graph.node)}")
    print(f"Removed Reshape nodes: {len(nodes_to_remove_indices)}")
    print(f"Optimized model saved to: {output_model_path}")


def remove_identity_reducesum(input_model_path, output_model_path):
   
    model = onnx.load(input_model_path)
    
   
    graph = model.graph
    

    node_map = {}
    for node in graph.node:
        node_map[node.name] = node
    

    input_map = {}
    for node in graph.node:
        for output in node.output:
            if output:
                input_map[output] = node
    

    nodes_to_remove = []
    
 
    replacement_map = {}
    
    
    for node in graph.node:
        if node.op_type == "Identity":
           
            input_name = node.input[0]
            output_name = node.output[0]
            
            
            replacement_map[output_name] = input_name
            
          
            nodes_to_remove.append(node)
 
    for node in graph.node:
        if node.op_type == "ReduceSum":
           
            is_removable = False
            
     
            for attr in node.attribute:
                if attr.name == "keepdims" and attr.i == 1:
                    is_removable = True
                    break
            
            if is_removable:
                input_name = node.input[0]
                output_name = node.output[0]
                
               
                replacement_map[output_name] = input_name
                
              
                nodes_to_remove.append(node)
    
    
    for node in graph.node:
        if node not in nodes_to_remove:
            for i, input_name in enumerate(node.input):
         
                if input_name in replacement_map:
                 
                    node.input[i] = replacement_map[input_name]
    
  
    for output in graph.output:
        if output.name in replacement_map:
            output.name = replacement_map[output.name]
    
  
    new_nodes = [node for node in graph.node if node not in nodes_to_remove]
    graph.ClearField("node")
    graph.node.extend(new_nodes)
 
    onnx.save(model, output_model_path)
    
    print(f"Saved to {output_model_path}")
    print(f"Removed {len(nodes_to_remove)}  Identity or ReduceSum")
    
    return model
