import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ovq.autograd.functional import quantize_per_channel_with_indices, dequantize_tensor_per_channel_with_indices, \
    Quantize

if __name__ == '__main__':
    ov = nn.Parameter(torch.randn(8, 2), requires_grad=True)
    idx = F.gumbel_softmax(ov, tau=1.0, hard=True)
    x = torch.randn((8, 8), requires_grad=True)

    dqx = Quantize.apply(x, idx[:, 1])
    print(x)
    print(idx[:, 1])
    print(dqx)

    # 定义优化器
    optimizer = optim.SGD([ov, x], lr=0.1)
    loss = torch.sum(dqx)  # 使用一个损失函数来定义梯度
    loss.backward()
    optimizer.step()
    # 输出梯度
    print(ov.grad)
    print(x.grad)
