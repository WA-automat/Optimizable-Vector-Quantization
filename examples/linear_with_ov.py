import torch

from ovq.nn.modules import LinearWithOV

if __name__ == '__main__':

    module = LinearWithOV(32, 2)
    module(torch.randn(32))
