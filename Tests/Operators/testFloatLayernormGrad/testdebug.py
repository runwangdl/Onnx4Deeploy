import onnx
import onnxruntime as ort
import numpy as np
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_nan_inf(name, tensor):
    """Check if tensor contains NaN or Inf values"""
    has_nan = np.isnan(tensor).any()
    has_inf = np.isinf(tensor).any()
    if has_nan:
        nan_indices = np.where(np.isnan(tensor))
        nan_count = np.isnan(tensor).sum()
        logger.error(f"❌ {name} contains {nan_count} NaN values! First NaN at: {nan_indices[0][0]}")
    if has_inf:
        inf_indices = np.where(np.isinf(tensor))
        inf_count = np.isinf(tensor).sum()
        logger.error(f"❌ {name} contains {inf_count} Inf values! First Inf at: {inf_indices[0][0]}")
    return not (has_nan or has_inf)

def check_ms_custom_ops():
    """Check if Microsoft custom operators are installed"""
    logger.info("Checking Microsoft custom operators...")
    
    # Print ONNX Runtime version
    logger.info(f"ONNX Runtime version: {ort.__version__}")
    
    # Check available execution providers
    providers = ort.get_available_providers()
    logger.info(f"Available execution providers: {providers}")
    
    # Try to get registered operator information
    try:
        operators = ort.get_all_operators_with_domain()
        ms_ops = [op for op in operators if "microsoft" in op[0].lower()]
        
        if ms_ops:
            logger.info(f"Found Microsoft operators: {ms_ops}")
            layernorm_ops = [op for op in ms_ops if "layernorm" in op[1].lower()]
            if layernorm_ops:
                logger.info(f"Found LayerNorm related operators: {layernorm_ops}")
                return True
            else:
                logger.warning("No LayerNorm related operators found")
        else:
            logger.warning("No Microsoft custom operators found")
    except:
        logger.warning("Unable to get registered operator information")
    
    return False

def numpy_layernorm_grad(dY, X, mean, invstd, gamma):
    """Implement LayerNorm backpropagation using NumPy"""
    # Get shape information
    batch_size, seq_len, hidden_size = X.shape
    
    # Calculate normalized input
    X_centered = X - mean.reshape(batch_size, seq_len, 1)
    X_norm = X_centered * invstd.reshape(batch_size, seq_len, 1)
    
    # Calculate gradients for gamma and beta
    dX_norm = dY * gamma
    
    # Calculate gradients for mean and variance
    dvar = -0.5 * np.sum(dX_norm * X_centered, axis=-1) * invstd**3
    dmean = -np.sum(dX_norm * invstd.reshape(batch_size, seq_len, 1), axis=-1)
    
    # Add X_centered contribution to mean gradient
    dmean = dmean - 2.0 * dvar * np.mean(X_centered, axis=-1)
    
    # Calculate input gradient
    dX = dX_norm * invstd.reshape(batch_size, seq_len, 1)
    dX = dX + dvar.reshape(batch_size, seq_len, 1) * 2.0 * X_centered / hidden_size
    dX = dX + dmean.reshape(batch_size, seq_len, 1) / hidden_size
    
    # Calculate gradients for gamma and beta (normally accumulated across batch and sequence positions)
    dgamma_local = np.sum(dY * X_norm, axis=(0, 1))
    dbeta_local = np.sum(dY, axis=(0, 1))
    
    # Stay consistent with Microsoft implementation, return per-position gradients
    dgamma = np.sum(dY * X_norm, axis=-1)
    dbeta = np.sum(dY, axis=-1)
    
    return dX, dgamma, dbeta, dgamma_local, dbeta_local

def create_custom_layernorm_grad(save_path=None):
    """Create manually implemented LayerNorm gradient test"""
    
    # Set default save path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = save_path if save_path else os.path.join(script_dir, "numpy_onnx")
    
    # Ensure save directory exists
    os.makedirs(base_path, exist_ok=True)
    
    # Define standard filenames
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")
    
    # Define input shapes
    batch_size = 2
    seq_len = 3
    hidden_size = 4
    input_shape = (batch_size, seq_len, hidden_size)
    
    logger.info(f"Creating test data with shape: {input_shape}")
    
    # Generate stable random input data
    np.random.seed(42)
    X = np.random.uniform(-1.0, 1.0, input_shape).astype(np.float32)
    gamma = np.random.uniform(0.5, 1.5, hidden_size).astype(np.float32)
    beta = np.random.uniform(-0.5, 0.5, hidden_size).astype(np.float32)
    
    # Calculate values that would be computed during forward pass
    mean = np.mean(X, axis=-1, keepdims=False).astype(np.float32)
    variance = np.var(X, axis=-1, keepdims=False).astype(np.float32)
    epsilon = 1e-5
    invstd = (1.0 / np.sqrt(variance + epsilon)).astype(np.float32)
    
    # Generate upstream gradients
    dY = np.random.uniform(-1.0, 1.0, input_shape).astype(np.float32)
    
    # Save inputs
    np.savez(
        input_file,
        X=X,
        gamma=gamma, 
        beta=beta,
        mean=mean,
        invstd=invstd,
        dY=dY
    )
    
    logger.info(f"Input data saved to {input_file}")
    
    # Compute gradients using NumPy implementation
    logger.info("Computing gradients with NumPy...")
    dX, dgamma, dbeta, dgamma_global, dbeta_global = numpy_layernorm_grad(dY, X, mean, invstd, gamma)
    
    # Save outputs
    np.savez(
        output_file,
        grad_in=dX,  # Corresponds to ONNX grad_in
        dW=dgamma,  # Corresponds to ONNX dW (per-position)
        dB=dbeta,   # Corresponds to ONNX dB (per-position)
        dgamma_global=dgamma_global,  # Accumulated gradients (standard implementation)
        dbeta_global=dbeta_global     # Accumulated gradients (standard implementation)
    )
    
    logger.info(f"Output data saved to {output_file}")
    
    # Check if computation results contain NaN
    if np.isnan(dX).any():
        logger.error(f"❌ grad_in (dX) contains {np.isnan(dX).sum()} NaN values")
    else:
        logger.info("✅ grad_in (dX) computed correctly, no NaN values")
    
    if np.isnan(dgamma).any():
        logger.error(f"❌ dW (dgamma) contains {np.isnan(dgamma).sum()} NaN values")
    else:
        logger.info("✅ dW (dgamma) computed correctly, no NaN values")
    
    if np.isnan(dbeta).any():
        logger.error(f"❌ dB (dbeta) contains {np.isnan(dbeta).sum()} NaN values")
    else:
        logger.info("✅ dB (dbeta) computed correctly, no NaN values")
    
    # Print data information
    logger.info("\nInput shapes:")
    logger.info(f"  X: {X.shape}")
    logger.info(f"  gamma: {gamma.shape}")
    logger.info(f"  beta: {beta.shape}")
    logger.info(f"  mean: {mean.shape}")
    logger.info(f"  invstd: {invstd.shape}")
    logger.info(f"  dY: {dY.shape}")
    
    logger.info("\nOutput shapes:")
    logger.info(f"  dX (grad_in): {dX.shape}")
    logger.info(f"  dgamma (dW): {dgamma.shape}")
    logger.info(f"  dbeta (dB): {dbeta.shape}")
    logger.info(f"  dgamma_global: {dgamma_global.shape} - Standard parameter gradient")
    logger.info(f"  dbeta_global: {dbeta_global.shape} - Standard parameter gradient")
    
    logger.info("\nData ranges (for validation):")
    logger.info(f"  X: min={X.min():.6f}, max={X.max():.6f}, mean={X.mean():.6f}")
    logger.info(f"  dY: min={dY.min():.6f}, max={dY.max():.6f}, mean={dY.mean():.6f}")
    logger.info(f"  dX: min={dX.min():.6f}, max={dX.max():.6f}, mean={dX.mean():.6f}")
    logger.info(f"  dgamma: min={dgamma.min():.6f}, max={dgamma.max():.6f}, mean={dgamma.mean():.6f}")
    logger.info(f"  dbeta: min={dbeta.min():.6f}, max={dbeta.max():.6f}, mean={dbeta.mean():.6f}")
    
    return True

def create_normal_onnx_and_run(save_path=None):
    """Create and run ONNX model without custom operators"""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = save_path if save_path else os.path.join(script_dir, "plain_onnx")
    
    # Ensure save directory exists
    os.makedirs(base_path, exist_ok=True)
    
    # Define standard filenames
    onnx_file = os.path.join(base_path, "network.onnx")
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")
    
    # Define input shapes
    batch_size = 2
    seq_len = 3
    hidden_size = 4
    input_shape = (batch_size, seq_len, hidden_size)
    
    logger.info(f"Creating standard ONNX test with shape: {input_shape}")
    
    # Generate stable random input data
    np.random.seed(42)
    X = np.random.uniform(-1.0, 1.0, input_shape).astype(np.float32)
    
    # Save inputs
    np.savez(input_file, X=X)
    
    from onnx import TensorProto, helper
    
    # Create a simple Identity operator model
    X_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)
    Y_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, input_shape)
    
    node_def = helper.make_node(
        "Identity",
        inputs=["X"],
        outputs=["Y"],
        name="identity_node"
    )
    
    graph_def = helper.make_graph(
        [node_def], "simple_graph",
        [X_tensor],
        [Y_tensor]
    )
    
    model_def = helper.make_model(graph_def, producer_name="test_model")
    
    # Validate model
    try:
        onnx.checker.check_model(model_def)
        logger.info("✅ Standard ONNX model validated")
    except Exception as e:
        logger.error(f"❌ Standard ONNX model validation failed: {e}")
    
    # Save model
    onnx.save(model_def, onnx_file)
    logger.info(f"Standard ONNX model saved to {onnx_file}")
    
    # Run ONNX model
    logger.info("Running standard ONNX model...")
    
    try:
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 0  # Verbose logging
        
        ort_session = ort.InferenceSession(onnx_file, session_options)
        output = ort_session.run(None, {"X": X})
        
        Y = output[0]
        logger.info(f"Standard ONNX model run successfully, output shape: {Y.shape}")
        
        # Save outputs
        np.savez(output_file, Y=Y)
        logger.info(f"Standard ONNX output saved to {output_file}")
        
        # Validate output
        if np.array_equal(X, Y):
            logger.info("✅ Standard ONNX Identity run correctly")
        else:
            logger.error("❌ Standard ONNX Identity run incorrectly, input and output don't match")
            
        if np.isnan(Y).any():
            logger.error(f"❌ Standard ONNX output contains {np.isnan(Y).sum()} NaN values")
        else:
            logger.info("✅ Standard ONNX output has no NaN values")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Standard ONNX model run failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def manual_layernorm_grad_implementation():
    """Provide sample Python implementation code for LayerNormGrad"""
    
    code = """
import torch

def layernorm_backward(dY, X, gamma, eps=1e-5):

    # Get shape information
    batch_size, seq_len, hidden_size = X.shape
    
    # Compute intermediate values from forward pass
    mean = X.mean(dim=-1, keepdim=True)
    var = ((X - mean) ** 2).mean(dim=-1, keepdim=True)
    std = (var + eps).sqrt()
    invstd = 1.0 / std
    X_norm = (X - mean) * invstd
    
    # Compute backpropagation
    dX_norm = dY * gamma
    
    # Compute variance gradient
    dvar = (-0.5 * invstd**3 * (dX_norm * (X - mean)).sum(dim=-1, keepdim=True))
    
    # Compute mean gradient
    N = hidden_size
    dmean = (
        -dX_norm.sum(dim=-1, keepdim=True) * invstd
        - 2.0 * dvar * (X - mean).mean(dim=-1, keepdim=True)
    )
    
    # Compute input gradient
    dX = (
        dX_norm * invstd
        + dvar * 2.0 * (X - mean) / N
        + dmean / N
    )
    
    # Compute parameter gradients
    dgamma = (dY * X_norm).sum(dim=(0, 1))
    dbeta = dY.sum(dim=(0, 1))
    
    return dX, dgamma, dbeta
    """
    
    logger.info("PyTorch implementation sample for LayerNormGrad:\n" + code)
    return code

def generate_layernorm_grad_onnx_and_data(save_path=None):
    """Generate ONNX model for LayerNormalizationGrad operator with test data"""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = save_path if save_path else os.path.join(script_dir, "onnx_fixed")
    
    # Define standard filenames
    onnx_file = os.path.join(base_path, "network.onnx")
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")
    
    # Ensure the save directory exists
    os.makedirs(base_path, exist_ok=True)
    
    # Define input shapes
    batch_size = 2
    seq_len = 3
    hidden_size = 4
    input_shape = (batch_size, seq_len, hidden_size)
    feature_dim = input_shape[-1]
    
    logger.info(f"Creating test data with shape: {input_shape}")
    
    # Generate random input data with limited range
    np.random.seed(42)
    X = np.random.uniform(-1.0, 1.0, input_shape).astype(np.float32)
    logger.info(f"X shape: {X.shape}, range: [{X.min():.6f}, {X.max():.6f}], mean: {X.mean():.6f}")
    check_nan_inf("X", X)
    
    # Normal distribution weights and bias, avoiding extreme values
    gamma = np.random.uniform(0.5, 1.5, feature_dim).astype(np.float32)
    beta = np.random.uniform(-0.5, 0.5, feature_dim).astype(np.float32)
    
    # Compute mean and variance (as in forward pass)
    mean = np.mean(X, axis=-1, keepdims=False).astype(np.float32)
    
    # Compute variance using a numerically stable method
    # Method 1: Direct use of np.var
    variance_direct = np.var(X, axis=-1, keepdims=False).astype(np.float32)
    
    # Method 2: Manual calculation (potentially more stable)
    mean_expanded = np.expand_dims(mean, axis=-1)
    X_centered = X - mean_expanded
    variance_manual = np.mean(X_centered ** 2, axis=-1).astype(np.float32)
    
    # Choose variance (using average of two methods may be more stable)
    variance = ((variance_direct + variance_manual) / 2).astype(np.float32)
    
    # Set appropriate epsilon to avoid numerical issues
    epsilon = 1e-4  # Slightly larger epsilon for stability
    
    # Record minimum of variance + epsilon, check if close to zero
    min_variance_eps = np.min(variance + epsilon)
    logger.info(f"Minimum variance + epsilon: {min_variance_eps:.8f}")
    
    if min_variance_eps < 1e-6:
        logger.warning(f"⚠️ Variance + epsilon is very small ({min_variance_eps:.8f}), may cause instability")
        # For safety, ensure minimum variance is not too small
        variance = np.maximum(variance, 1e-6 - epsilon) 
        min_variance_eps = np.min(variance + epsilon)
        logger.info(f"Adjusted minimum variance + epsilon: {min_variance_eps:.8f}")
    
    # Compute inverse standard deviation
    invstd = (1.0 / np.sqrt(variance + epsilon)).astype(np.float32)
    
    # Generate upstream gradient
    dY = np.random.uniform(-1.0, 1.0, input_shape).astype(np.float32)
    
    # Compute normalized gradient
    dY_norm = (dY * gamma).astype(np.float32)
    
    # Save input data
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
    
    from onnx import TensorProto, helper
    
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
        inputs=["dY", "X", "mean", "invstd", "dY_norm"],  # Inputs
        outputs=["grad_in", "dW", "dB"],  # Outputs
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
    
    # Include both default opset and Microsoft opset
    opset_imports = [
        helper.make_opsetid("", 13),  # Standard ONNX domain
        helper.make_opsetid("com.microsoft", 1)  # Microsoft domain
    ]
    
    model_def = helper.make_model(
        graph_def, 
        producer_name="layernorm_grad_model",
        opset_imports=opset_imports
    )
    
    # Validate model
    try:
        onnx.checker.check_model(model_def)
        logger.info("✅ ONNX model validated")
    except Exception as e:
        logger.error(f"❌ ONNX model validation failed: {e}")
    
    # Save ONNX model
    onnx.save(model_def, onnx_file)
    logger.info(f"✅ ONNX model saved to {onnx_file}")
    
    # Compute manually to compare
    logger.info("Computing with NumPy for comparison...")
    dX, dgamma, dbeta, _, _ = numpy_layernorm_grad(dY, X, mean, invstd, gamma)
    
    # Save outputs from manual calculation
    np.savez(
        output_file, 
        grad_in=dX,    # Input gradient
        dW=dgamma,     # Gamma gradient (per batch and seq position)
        dB=dbeta       # Beta gradient (per batch and seq position)
    )
    
    logger.info(f"✅ Manual output data saved to {output_file}")
    
    logger.info("\nInput shapes:")
    logger.info(f"  dY (upstream gradient): {dY.shape}")
    logger.info(f"  X (original input): {X.shape}")
    logger.info(f"  mean: {mean.shape}")
    logger.info(f"  invstd (inverse std dev): {invstd.shape}")
    logger.info(f"  dY_norm (gamma * dY): {dY_norm.shape}")
    
    logger.info("\nOutput shapes:")
    logger.info(f"  grad_in (input gradient): {dX.shape}")
    logger.info(f"  dW (gamma gradient): {dgamma.shape}")
    logger.info(f"  dB (beta gradient): {dbeta.shape}")
    
    # Print data range information for validation
    logger.info("\nData ranges for validation:")
    logger.info(f"  X: min={X.min():.6f}, max={X.max():.6f}, mean={X.mean():.6f}")
    logger.info(f"  dY: min={dY.min():.6f}, max={dY.max():.6f}, mean={dY.mean():.6f}")
    logger.info(f"  mean: min={mean.min():.6f}, max={mean.max():.6f}")
    logger.info(f"  invstd: min={invstd.min():.6f}, max={invstd.max():.6f}")
    logger.info(f"  grad_in: min={dX.min():.6f}, max={dX.max():.6f}, mean={dX.mean():.6f}")
    logger.info(f"  dW: min={dgamma.min():.6f}, max={dgamma.max():.6f}, mean={dgamma.mean():.6f}")
    logger.info(f"  dB: min={dbeta.min():.6f}, max={dbeta.max():.6f}, mean={dbeta.mean():.6f}")
    
    return True    

if __name__ == "__main__":
    logger.info("Starting LayerNorm compatibility check...")
    
    # Test if Microsoft custom operators are available
    has_ms_ops = check_ms_custom_ops()
    
    if has_ms_ops:
        logger.info("✅ Microsoft custom operators available, can use LayerNormalizationGrad")
    else:
        logger.warning("⚠️ Microsoft custom operators not available, need alternative implementation")
    
    # Test if standard ONNX runs normally
    logger.info("\n=== Testing standard ONNX operation ===")
    standard_onnx_works = create_normal_onnx_and_run()
    
    if standard_onnx_works:
        logger.info("✅ Standard ONNX model runs normally")
    else:
        logger.error("❌ Standard ONNX model has issues, ONNX Runtime may not be installed correctly")
        sys.exit(1)
    
    # Generate test data using NumPy implementation
    logger.info("\n=== Generating LayerNormGrad test data with NumPy implementation ===")
    create_custom_layernorm_grad()
    
    # Generate fixed version of the original ONNX model
    logger.info("\n=== Generating fixed LayerNormGrad ONNX model ===")
    generate_layernorm_grad_onnx_and_data()
    
    # Provide manual implementation example
    logger.info("\n=== Providing LayerNormGrad implementation example ===")
    manual_layernorm_grad_implementation()
    
    logger.info("\nLayerNorm compatibility check completed")