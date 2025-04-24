cd /app/scripts/01Compact-Transformers

python testgenerate.py

cd /app/Deeploy/DeeployTest/Tests/CCT

python -m onnxruntime.tools.make_dynamic_shape_fixed --input_name "onnx::Conv_0" --input_name "input" --input_shape 1,3,16,16 network.onnx network.onnx

python -m onnxruntime.tools.symbolic_shape_infer --input network.onnx --output network.onnx --verbose 3

python -m onnxruntime.transformers.optimizer \
 --input network.onnx \
 --output network.onnx \
 --model_type vit \
 --num_heads 1 \
 --hidden_size 8 \
 --use_multi_head_attention \
 --disable_bias_skip_layer_norm \
 --disable_skip_layer_norm \
 --disable_bias_gelu

cd /app/scripts/01Compact-Transformers
python RemoveAddBroadcast.py
python RemoveGemmBroadcast.py