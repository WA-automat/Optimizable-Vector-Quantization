import torch
from torch.autograd import Function


def quantize_per_channel_with_indices(tensor, indices):
    qx = tensor
    rows, cols = qx.size()
    tensor = tensor[:, (indices == 1)].abs()
    if tensor.numel() == 0:
        return qx, torch.zeros(rows)
    scale, _ = tensor.max(dim=1)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    result = ((qx / scale.view(-1, 1)) * 127).round()
    qx = torch.where(indices == 1, result, qx)

    # for i in range(rows):
    #     for j, v in enumerate(indices):
    #         if v == 1:
    #             if scale[i] != 0:
    #                 qx[i, j] = ((qx[i, j] / scale[i]) * 127).round()

    return qx, scale


def dequantize_tensor_per_channel_with_indices(tensor, indices, scale):
    dqx = tensor
    result = ((dqx / 127) * scale.view(-1, 1))
    dqx = torch.where(indices == 1, result, dqx)

    # for i in range(rows):
    #     for j, v in enumerate(indices):
    #         if v == 1:
    #             if scale[i] != 0:
    #                 dqx[i, j] = (dqx[i, j] / 127) * scale[i]

    return dqx


class Quantize(Function):
    """
    伪量化算子，使用可优化向量进行伪量化
    """

    @staticmethod
    def forward(ctx, x, indices):
        ctx.save_for_backward(x, indices)
        x_q, scale = quantize_per_channel_with_indices(x, indices)
        x_dq = dequantize_tensor_per_channel_with_indices(x_q, indices, scale)
        ctx.x_dq = x_dq
        return x_dq

    @staticmethod
    def backward(ctx, grad_output):
        x, indices = ctx.saved_tensors

        # 计算关于 indices 的梯度
        diff = ctx.x_dq - x
        grad_indices = diff.sum(dim=0)

        # 添加正则化项
        # 例如，L2 正则化
        l2_reg = 0.01  # 正则化系数
        # 添加随机噪声到梯度
        random_noise = torch.randn_like(grad_indices)
        grad_indices_reg = torch.zeros_like(indices)
        for i in range(len(indices)):
            grad_indices_reg[i] = l2_reg

        # 合并梯度和正则化项
        grad_indices = grad_indices + grad_indices_reg * random_noise

        return grad_output, grad_indices
