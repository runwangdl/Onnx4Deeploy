import onnx
import onnxruntime as ort
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper

def generate_layernorm_grad_onnx_and_data(save_path=None):
    """ Generate ONNX model for LayerNormalizationGrad operator with test data """
    
    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config.yaml")
    
    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    model_config = config["layernorm_grad"]
    input_shape = tuple(model_config["input_shape"])
    feature_dim = input_shape[-1]
    batch_size = input_shape[0]
    seq_len = input_shape[1]
    
    # Set default save path if not provided
    base_path = save_path if save_path else os.path.join(script_dir, "onnx")
    
    # Define standard filenames
    onnx_file = os.path.join(base_path, "network.onnx")
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")
    
    # Ensure the save directory exists
    os.makedirs(base_path, exist_ok=True)
    
    # Generate random input data
    # X is the original input to LayerNorm
    X = np.random.randn(*input_shape).astype(np.float32)
    
    # Generate random weights and bias for LayerNorm
    gamma = np.random.rand(feature_dim).astype(np.float32)
    beta = np.random.rand(feature_dim).astype(np.float32)
    
    # Compute mean and invstd as would be calculated in forward pass
    mean = np.mean(X, axis=-1).astype(np.float32)  # Shape: [batch_size, seq_len]
    variance = np.var(X, axis=-1).astype(np.float32)  # Shape: [batch_size, seq_len]
    epsilon = 1e-5
    invstd = (1.0 / np.sqrt(variance + epsilon)).astype(np.float32)
    
    # dY is the upstream gradient
    dY = np.random.randn(*input_shape).astype(np.float32)
    
    # dY_norm = gamma * dY (pre-compute for the model)
    dY_norm = (dY * gamma).astype(np.float32)
    
    # Save input data with clear names
    np.savez(
        input_file, 
        dY=dY,              # Upstream gradient
        X=X,                # Original input
        mean=mean,          # Saved mean
        invstd=invstd,      # Saved inverse standard deviation
        dY_norm=dY_norm,    # gamma * dY
        gamma=gamma,        # Scale parameter
        beta=beta           # Bias parameter
    )
    
    # Define batch_seq_shape for clarity
    batch_seq_shape = input_shape[:-1]  # [batch_size, seq_len]
    
    # Define ONNX tensors with correct shapes
    dY_tensor = helper.make_tensor_value_info("dY", TensorProto.FLOAT, input_shape)
    X_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)
    mean_tensor = helper.make_tensor_value_info("mean", TensorProto.FLOAT, batch_seq_shape)
    invstd_tensor = helper.make_tensor_value_info("invstd", TensorProto.FLOAT, batch_seq_shape)
    dY_norm_tensor = helper.make_tensor_value_info("dY_norm", TensorProto.FLOAT, input_shape)
    
    # Define output tensors with Microsoft's expected shapes
    grad_in_tensor = helper.make_tensor_value_info("grad_in", TensorProto.FLOAT, input_shape)
    dW_tensor = helper.make_tensor_value_info("dW", TensorProto.FLOAT, batch_seq_shape)
    dB_tensor = helper.make_tensor_value_info("dB", TensorProto.FLOAT, batch_seq_shape)
    
    # Create ONNX computation graph with Microsoft's operator
    layernorm_grad_node = helper.make_node(
        "LayerNormalizationGrad",  # Op name
        inputs=["dY", "X", "mean", "invstd", "dY_norm"],  # Inputs as specified
        outputs=["grad_in", "dW", "dB"],  # Outputs as specified
        name="layernorm_grad_node",
        domain="com.microsoft",  # Microsoft domain
        axis=-1,  # Apply layernorm along the last axis
        epsilon=float(epsilon)
    )
    
    graph_def = helper.make_graph(
        [layernorm_grad_node], "layernorm_grad_graph",
        [dY_tensor, X_tensor, mean_tensor, invstd_tensor, dY_norm_tensor],
        [grad_in_tensor, dW_tensor, dB_tensor]
    )
    
    # Include both the default opset and the Microsoft opset
    opset_imports = [
        helper.make_opsetid("", 13),  # Standard ONNX domain
        helper.make_opsetid("com.microsoft", 1)  # Microsoft domain
    ]
    
    model_def = helper.make_model(
        graph_def, 
        producer_name="layernorm_grad_model",
        opset_imports=opset_imports
    )
    
    # Save ONNX model
    onnx.save(model_def, onnx_file)
    print(f"✅ ONNX model saved to {onnx_file}")
    
    # Run inference using ONNX Runtime
    ort_session = ort.InferenceSession(onnx_file)
    output_data = ort_session.run(
        None, 
        {"dY": dY, "X": X, "mean": mean, "invstd": invstd, "dY_norm": dY_norm}
    )
    
    # Extract outputs with clear names
    grad_in = output_data[0]  # Shape: [batch_size, seq_len, feature_dim]
    dW = output_data[1]       # Shape: [batch_size, seq_len] - Microsoft implementation
    dB = output_data[2]       # Shape: [batch_size, seq_len] - Microsoft implementation
    
    # Save output data with clear labels
    np.savez(
        output_file, 
        grad_in=grad_in,  # Input gradient
        dW=dW,           # Gamma parameter gradient (per batch and seq position)
        dB=dB            # Beta parameter gradient (per batch and seq position)
    )
    
    print(f"✅ Output data saved to {output_file}")
    
    # Print detailed input and output information
    print(f"\nInput shapes:")
    print(f"  dY (upstream gradient): {dY.shape}")
    print(f"  X (original input): {X.shape}")
    print(f"  mean: {mean.shape}")
    print(f"  invstd (inverse std dev): {invstd.shape}")
    print(f"  dY_norm (gamma * dY): {dY_norm.shape}")
    
    print(f"\nOutput shapes:")
    print(f"  grad_in (input gradient): {grad_in.shape}")
    print(f"  dW (gamma gradient): {dW.shape} - Note: Microsoft implementation, one gradient per (batch,seq) position")
    print(f"  dB (beta gradient): {dB.shape} - Note: Microsoft implementation, one gradient per (batch,seq) position")
    
    # Print data range information for validation
    print(f"\nData ranges for validation:")
    print(f"  X: min={X.min():.6f}, max={X.max():.6f}, mean={X.mean():.6f}")
    print(f"  dY: min={dY.min():.6f}, max={dY.max():.6f}, mean={dY.mean():.6f}")
    print(f"  mean: min={mean.min():.6f}, max={mean.max():.6f}")
    print(f"  invstd: min={invstd.min():.6f}, max={invstd.max():.6f}")
    print(f"  grad_in: min={grad_in.min():.6f}, max={grad_in.max():.6f}, mean={grad_in.mean():.6f}")
    print(f"  dW: min={dW.min():.6f}, max={dW.max():.6f}, mean={dW.mean():.6f}")
    print(f"  dB: min={dB.min():.6f}, max={dB.max():.6f}, mean={dB.mean():.6f}")
    
    print(f"\nCompleted - C kernel should accept dW and dB outputs with shape [batch_size, seq_len]")

if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_layernorm_grad_onnx_and_data(save_path)