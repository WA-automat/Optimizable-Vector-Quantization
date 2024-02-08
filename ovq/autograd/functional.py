import torch
from torch.autograd import Function


def quantize_per_channel_with_indices(tensor, indices):
    scale, _ = tensor.abs().max(dim=0)
    qx = tensor.clone()
    for i, v in enumerate(indices):
        if v == 1:
            qx[:, i] = ((qx[:, i] / scale[i]) * 127).round()
    return qx, scale


def dequantize_tensor_per_channel_with_indices(tensor, indices, scale):
    dqx = tensor.clone()
    for i, v in enumerate(indices):
        if v == 1:
            dqx[:, i] = (dqx[:, i] / 127) * scale[i]
    return dqx


class Quantize(Function):
    """
    伪量化算子
    """

    @staticmethod
    def forward(ctx, x, indices):
        ctx.save_for_backward(x, indices)
        x_q, scale = quantize_per_channel_with_indices(x, indices)
        x_dq = dequantize_tensor_per_channel_with_indices(x_q, indices, scale)
        return x_dq

    @staticmethod
    def backward(ctx, grad_output):
        x, indices = ctx.saved_tensors

        # 计算关于 indices 的梯度
        grad_indices = torch.zeros_like(indices)
        grad_indices[indices == 1] = grad_output[x != 0].sum(dim=0)

        return grad_output, grad_indices
