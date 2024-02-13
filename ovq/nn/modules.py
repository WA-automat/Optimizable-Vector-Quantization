import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from ovq.autograd.functional import Quantize, quantize_per_channel_with_indices


class LinearWithOV(nn.Linear):
    """
    带有可优化向量的线性层
    """

    def __init__(self, in_features, out_features, bias=True, device=None):
        super(LinearWithOV, self).__init__(in_features, out_features, bias, device)
        self.ov = nn.Parameter(torch.randn(in_features, 2), requires_grad=True)
        self.lock = False

    def forward(self, x: Tensor) -> Tensor:
        if not self.lock:
            # 若为微调：进行伪量化
            ov = F.gumbel_softmax(self.ov, tau=1.0, hard=True)
            x = Quantize.apply(x, ov[:, 1])  # 对输入进行伪量化
            w = Quantize.apply(self.weight, ov[:, 1])  # 对权重进行伪量化
            y = x @ w.t()

            # 若包含偏置，计算包含偏置的结果
            if self.bias is not None:
                y += self.bias
            return y
        else:
            # 若为推理：首先进行量化输入
            quantize_x, sx = quantize_per_channel_with_indices(x, self.v)
            qx = quantize_x[:, (self.v == 1)]
            uqx = quantize_x[:, (self.v == 0)]
            qx = qx.to(torch.int32)

            # 处理计算结果
            s = torch.outer(sx, self.sw)
            y = uqx @ self.uqw.t()
            y += ((qx @ self.qw.t().to(torch.int32)).to(torch.float32) / (127 * 127)) * s
            if self.bias is not None:
                y += self.bias
            return y

    def quantize(self):
        """
        量化函数，在这里实现权重的拆分等操作
        """
        self.lock = True
        self.register_buffer("v", F.gumbel_softmax(self.ov, tau=1.0, hard=True)[:, 1])
        # self.v = F.gumbel_softmax(self.ov, tau=1.0, hard=True)[:, 1]

        # 对 self.qw 进行量化
        quantize_weight, sw = quantize_per_channel_with_indices(self.weight, self.v)
        self.register_buffer("sw", sw)
        self.register_buffer("qw", quantize_weight[:, (self.v == 1)].to(torch.int8))
        self.register_buffer("uqw", quantize_weight[:, (self.v == 0)])
        # self.qw = quantize_weight[:, (self.v == 1)]
        # self.uqw = quantize_weight[:, (self.v == 0)]
        # self.qw = self.qw.to(torch.int8)

        # 删除不需要的属性
        del self.ov
        del self.weight

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """
        加载模型参数
        :param state_dict:
        :param prefix:
        :param local_metadata:
        :param strict:
        :param missing_keys:
        :param unexpected_keys:
        :param error_msgs:
        :return:
        """
        self.lock = True
        self.register_buffer("v", state_dict[prefix + "v"])
        self.register_buffer("sw", state_dict[prefix + "sw"])
        self.register_buffer("qw", state_dict[prefix + "qw"])
        self.register_buffer("uqw", state_dict[prefix + "uqw"])
        self.bias = torch.nn.Parameter(state_dict[prefix + "bias"])
        del self.weight
        del self.ov
