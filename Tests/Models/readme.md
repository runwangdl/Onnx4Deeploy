1. cct num of conv layer still in cct test
2. layernorm should keep opset < 12 for infer
3. layernorm should keep opset = 15 for train
4. infer should 
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

 5. train should
python -m onnxruntime.transformers.optimizer \
 --input network_train.onnx \
 --output network_train_optim.onnx \
 --model_type vit \
 --num_heads 1 \
 --hidden_size 8 \
 --use_multi_head_attention \
 --disable_bias_skip_layer_norm \
 --disable_skip_layer_norm \
 --disable_bias_gelu \
 --disable_layer_norm 
