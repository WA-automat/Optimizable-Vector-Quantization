import torch
from torch import nn
import torch.nn.functional as F
from ovq.nn.modules import LinearWithOV


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
    inputs = torch.tensor([[4], [6], [7], [8]], dtype=torch.float32)
    module = MLPWithOV(1, 1)
    state_dict = torch.load("../model/test.pt")
    print(state_dict)
    # print(state_dict["v"])
    # module.quantize()
    module.load_state_dict(state_dict)
    print(module)
    print(module(inputs))
