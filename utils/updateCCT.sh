cd /app/scripts/Compact-Transformers

python testgenerate.py

cd /app/Deeploy/DeeployTest/Tests/CCT

python -m onnxruntime.tools.make_dynamic_shape_fixed --input_name "onnx::Conv_0" --input_name "input" --input_shape 1,3,16,16 network.onnx network.onnx

python -m onnxruntime.tools.symbolic_shape_infer --input network.onnx --output network.onnx --verbose 3

python -m onnxruntime.transformers.optimizer \
 --input network_train.onnx \
 --output network_train_optim.onnx \
 --disable_bias_skip_layer_norm \
 --disable_skip_layer_norm \
 --disable_bias_gelu \
 --disable_layer_norm \
 --disable_

cd /app/scripts/Compact-Transformers
python RemoveAddBroadcast.py
python RemoveGemmBroadcast.py