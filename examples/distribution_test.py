import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ovq.autograd.functional import Quantize, quantize_per_channel_with_indices


class LinearWithV(nn.Linear):
    """
    带有可优化向量的线性层
    """

    def __init__(self, in_features, out_features, p=1.0, bias=True, device=None, dtype=torch.float32):
        super(LinearWithV, self).__init__(in_features, out_features, bias, device, dtype)
        indices = torch.multinomial(torch.tensor([0.45, 0.45, 0.1]), in_features, replacement=True)
        self.v = torch.zeros(in_features)
        self.v[indices == 1] = 1
        self.v[indices == 2] = 2
        # 对 self.qw 进行量化
        quantize_weight, sw = quantize_per_channel_with_indices(self.weight, self.v)
        self.register_buffer("sw", sw)
        self.register_buffer("qw", quantize_weight[:, (self.v == 1)].to(torch.int8))
        self.register_buffer("uqw", quantize_weight[:, (self.v == 0)])
        self.dtype = dtype

        # 删除不需要的属性
        del self.weight

    def forward(self, x):
        # 若为推理：首先进行量化输入
        quantize_x, sx = quantize_per_channel_with_indices(x.to(self.dtype), self.v)
        del x

        uqx = quantize_x[:, (self.v == 0)]
        qx = quantize_x[:, (self.v == 1)]

        # 处理计算结果
        s = torch.ger(sx, self.sw)
        y = torch.matmul(uqx, self.uqw.t())
        y += torch.matmul(qx.to(torch.int32), self.qw.t().to(torch.int32)) / (127 * 127) * s

        if self.bias is not None:
            y += self.bias
        return y


class MLPWithV(nn.Module):
    def __init__(self, in_features, out_features, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = LinearWithV(in_features=in_features, out_features=256, p=p)
        self.l2 = LinearWithV(in_features=256, out_features=64, p=p)
        self.l3 = LinearWithV(in_features=64, out_features=16, p=p)
        self.l4 = LinearWithV(in_features=16, out_features=4, p=p)
        self.l5 = LinearWithV(in_features=4, out_features=out_features, p=p)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = F.relu(x)
        x = self.l4(x)
        x = F.relu(x)
        out = self.l5(x)
        return out


if __name__ == '__main__':
    epochs = 1000
    test_inputs = torch.tensor([[-1], [0], [5], [6], [-2], [0], [10], [12]], dtype=torch.float32)
    module = MLPWithV(1, 1, 0.75)
    module.eval()
    total_time = 0
    for epoch in range(epochs):
        start_time = time.time()
        test_outputs = module(test_inputs)
        end_time = time.time()
        epoch_time = end_time - start_time
        total_time += epoch_time
    average_time = total_time / epochs
    print(f'Average Time per Epoch: {average_time}')
