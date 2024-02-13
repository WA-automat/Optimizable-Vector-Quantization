import torch
from torch.autograd import Function


def quantize_per_channel_with_indices(tensor, indices):
    qx = tensor.clone()
    rows, cols = qx.size()
    if torch.all(indices == 0):
        return qx, torch.zeros(rows)
    scale, _ = tensor[:, (indices == 1)].abs().max(dim=1)
    for i in range(rows):
        for j, v in enumerate(indices):
            if v == 1:
                if scale[i] != 0:
                    qx[i, j] = ((qx[i, j] / scale[i]) * 127).round()
    return qx, scale


def dequantize_tensor_per_channel_with_indices(tensor, indices, scale):
    dqx = tensor.clone()
    rows, cols = dqx.size()
    for i in range(rows):
        for j, v in enumerate(indices):
            if v == 1:
                if scale[i] != 0:
                    dqx[i, j] = (dqx[i, j] / 127) * scale[i]
    return dqx


def matmul_dequantize(x, sx, w, sw):
    """
    TODO: int8矩阵乘法并返回反量化后的浮点矩阵
    :param x: 输入（int8类型）
    :param sx: 输入的量化范围
    :param w: 权重（int8类型）
    :param sw: 权重的量化范围
    :return: 反量化后的矩阵乘法结果
    """
    pass


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
