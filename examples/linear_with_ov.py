import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
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
        self.l1 = LinearWithOV(in_features=in_features, out_features=8)
        self.l2 = LinearWithOV(in_features=8, out_features=4)
        self.l3 = LinearWithOV(in_features=4, out_features=2)
        self.l4 = LinearWithOV(in_features=2, out_features=out_features)

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
    epochs = 5000

    # 创建模型
    # module = LinearModel()
    module = LinearWithOV(1, 1)
    # module = MLPWithOV(1, 1)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(module.parameters(), lr=0.001)

    # 模拟训练数据
    inputs = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
    targets = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

    # 训练模型
    for epoch in range(epochs):
        # 前向传播
        outputs = module(inputs)

        # 计算损失
        loss = criterion(outputs, targets)

        # 反向传播和参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % (epochs / 10) == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

    # 打印训练后的输出
    print(module(inputs))

    # 量化
    # module.l1.quantize()
    # module.l2.quantize()
    # module.l3.quantize()
    # module.l4.quantize()
    module.quantize()

    # 推理
    start = time.time()
    print(module(inputs))
    end = time.time()
    print(end - start)

    print(module.state_dict())
