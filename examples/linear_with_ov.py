import torch
import torch.optim as optim
import torch.nn as nn
from ovq.nn.modules import LinearWithOV


# 定义线性模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入维度为1，输出维度为1

    def forward(self, x):
        return self.linear(x)


if __name__ == '__main__':
    epochs = 10000

    # 创建模型
    # module = LinearModel()
    module = LinearWithOV(1, 1)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(module.parameters(), lr=0.01)

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
