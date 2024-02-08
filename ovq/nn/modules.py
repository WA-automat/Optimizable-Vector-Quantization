import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class LinearWithOV(nn.Linear):
    """
    带有可优化向量的线性层
    """

    def __init__(self, in_features, out_features, bias=True, device=None):
        super(LinearWithOV, self).__init__(in_features, out_features, bias, device)
        self.ov = nn.Parameter(torch.randn(in_features, 2), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        ov = F.gumbel_softmax(self.ov, tau=1.0, hard=True)
        print(ov[:, 1])

        # 拆分矩阵

        # 对输入进行量化

        # 对权重进行量化

        # 计算结果并进行反量化

        # 若包含偏置，计算包含偏置的结果

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """
        TODO: 加载模型函数
        :param state_dict:
        :param prefix:
        :param local_metadata:
        :param strict:
        :param missing_keys:
        :param unexpected_keys:
        :param error_msgs:
        :return:
        """
        return NotImplementedError

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """
        TODO: 保存模型函数
        :param destination:
        :param prefix:
        :param keep_vars:
        :return:
        """
        return NotImplementedError
