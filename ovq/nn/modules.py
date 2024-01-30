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

    def forward(self, input: Tensor) -> Tensor:
        ov = F.gumbel_softmax(self.ov)
        ov, indices = torch.max(ov, dim=1)

        print(torch.mul(self.weight, indices))
