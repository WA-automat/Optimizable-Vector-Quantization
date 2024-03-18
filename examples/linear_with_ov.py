import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

from ovq.nn.modules import LinearWithOV


# 定义线性模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入维度为1，输出维度为1

    def forward(self, x):
        return self.linear(x)


class MLPWithOV(nn.Module):
    def __init__(self, in_features, out_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = LinearWithOV(in_features=in_features, out_features=128)
        self.l2 = LinearWithOV(in_features=128, out_features=64)
        self.l3 = LinearWithOV(in_features=64, out_features=8)
        self.l4 = LinearWithOV(in_features=8, out_features=out_features)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = F.relu(x)
        out = self.l4(x)
        return out


class MLP(nn.Module):
    def __init__(self, in_features, out_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = nn.Linear(in_features=in_features, out_features=128)
        self.l2 = nn.Linear(in_features=128, out_features=64)
        self.l3 = nn.Linear(in_features=64, out_features=8)
        self.l4 = nn.Linear(in_features=8, out_features=out_features)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = F.relu(x)
        out = self.l4(x)
        return out


if __name__ == '__main__':
    epochs = 500

    # 创建模型
    # module = LinearModel()
    # module = LinearWithOV(1, 1)
    module = MLPWithOV(1, 1)
    # module = MLP(1, 1)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(module.parameters(), lr=0.1)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    # 模拟训练数据
    inputs = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
    targets = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
    test_inputs = torch.tensor([[-1], [0], [5], [6]], dtype=torch.float32)
    test_targets = torch.tensor([[-2], [0], [10], [12]], dtype=torch.float32)
    total_time = 0

    # 训练模型
    for epoch in range(epochs):

        module.train()
        # 前向传播
        outputs = module(inputs)

        # 计算损失
        loss = criterion(outputs, targets)

        # 反向传播和参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % (epochs / 10) == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

        # 模拟测试过程
        module.eval()
        start_time = time.time()
        test_outputs = module(test_inputs)
        test_loss = criterion(test_outputs, test_targets)

        end_time = time.time()
        epoch_time = end_time - start_time
        total_time += epoch_time

        if epoch % (epochs / 10) == 0:
            print(f'Test Loss: {test_loss.item()}, Time: {epoch_time}')

    average_time = total_time / epochs
    print(f'Average Time per Epoch: {average_time}')

    # 打印训练后的输出
    module.eval()
    print(module(inputs))
    print(module(test_inputs))

    # 量化
    module.l1.quantize()
    module.l2.quantize()
    module.l3.quantize()
    module.l4.quantize()
    # module.quantize()

    # 推理
    start = time.time()
    print(module(inputs))
    print(module(test_inputs))
    end = time.time()
    print(end - start)

    torch.save(module.state_dict(), "../model/test.pt")

    total_time = 0
    for epoch in range(epochs // 10):
        start_time = time.time()
        test_outputs = module(test_inputs)
        end_time = time.time()
        epoch_time = end_time - start_time
        total_time += epoch_time
    average_time = total_time / (epochs // 10)
    print(f'Average Time per Epoch: {average_time}')
