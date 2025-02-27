import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
import numpy as np
import onnxruntime.training.onnxblock as onnxblock
from onnxruntime.training.api import CheckpointState, Module, Optimizer
from onnxruntime.training import artifacts
from onnxruntime import InferenceSession
import onnx
import io
import netron

class MNISTNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, model_input):
        out = self.fc1(model_input)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 设置设备和模型尺寸
device = "cpu"
batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10

# 创建PyTorch模型
pt_model = MNISTNet(input_size, hidden_size, output_size).to(device)

# 为导出准备输入
model_inputs = (torch.randn(batch_size, input_size, device=device),)

# 执行前向传播以验证模型
model_outputs = pt_model(*model_inputs)
if isinstance(model_outputs, torch.Tensor):
    model_outputs = [model_outputs]

# 设置输入输出名称和动态轴
input_names = ["input"]
output_names = ["output"]
dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

# 导出模型到ONNX格式
f = io.BytesIO()
torch.onnx.export(
    pt_model,
    model_inputs,
    f,
    input_names=input_names,
    output_names=output_names,
    opset_version=14,
    do_constant_folding=False,
    training=torch.onnx.TrainingMode.TRAINING,
    dynamic_axes=dynamic_axes,
    export_params=True,
    keep_initializers_as_inputs=False,
)
onnx_model = onnx.load_model_from_string(f.getvalue())

# 保存前向传播模型
onnx.save(onnx_model, "mnist_forward.onnx")
print("✅ 前向传播模型已保存: mnist_forward.onnx")

# 获取模型所有参数名称列表
all_param_names = [name for name, _ in pt_model.named_parameters()]
print(f"所有参数: {all_param_names}")

# 只为最后一层的权重启用梯度更新
requires_grad = ['fc2.weight']  # 只包含最后一层的权重

# 冻结其他所有层的参数
frozen_params = [name for name in all_param_names if name not in requires_grad]

print(f"需要梯度的参数: {requires_grad}")
print(f"冻结的参数: {frozen_params}")

# 生成训练相关的artifacts
artifacts.generate_artifacts(
    onnx_model,
    optimizer=artifacts.OptimType.AdamW,
    loss=artifacts.LossType.CrossEntropyLoss,
    requires_grad=requires_grad,  # 只有最后一层的权重会更新
    frozen_params=frozen_params,  # 冻结其他所有参数
    artifact_directory="data",
    additional_output_names=["output"]
)

print("✅ 训练图已生成并保存在 `data/` 目录下")
print("✅ 模型配置为只训练最后一层 (fc2) 的权重")

# 定义一个训练函数
def train_model(num_epochs=5):
    try:
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        
        # 加载MNIST数据集
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('data', train=False, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建训练会话
        train_session = Module(
            "data/training_model.onnx",
            CheckpointState("data/checkpoint.ckpt"),
        )
        eval_session = InferenceSession("data/inference_model.onnx")
        optimizer = Optimizer("data/optimizer_model.onnx")
        
        # 训练模型
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                # 准备输入
                inputs = {
                    "input": data.reshape(data.shape[0], -1).numpy(),
                    "target": target.numpy(),
                }
                
                # 前向和反向传播
                output = train_session.run(inputs)
                loss = output["loss"][0]
                
                # 优化器步骤
                optimizer.step()
                
                # 记录统计信息
                running_loss += loss
                if batch_idx % 100 == 0:
                    print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}, Loss: {loss:.6f}")
            
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1} completed, Average Loss: {avg_loss:.6f}")
        
        # 保存最终的检查点
        train_session.save_checkpoint("data/final_checkpoint.ckpt")
        print("✅ 训练完成，最终检查点已保存")
        
        # 评估模型
        correct = 0
        total = 0
        for data, target in test_loader:
            # 准备输入
            inputs = {"input": data.reshape(data.shape[0], -1).numpy()}
            
            # 运行推理
            outputs = eval_session.run(None, inputs)
            predicted = np.argmax(outputs[0], axis=1)
            
            # 计算准确率
            total += target.size(0)
            correct += (predicted == target.numpy()).sum()
        
        accuracy = 100.0 * correct / total
        print(f"测试集准确率: {accuracy:.2f}%")
        
        return True
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        return False

# 取消注释下面的行以启动训练
train_model(num_epochs=5)