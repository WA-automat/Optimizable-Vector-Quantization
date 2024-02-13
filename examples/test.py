import torch

from ovq.autograd.functional import quantize_per_channel_with_indices

if __name__ == '__main__':
    x = torch.tensor([[1, 2, 3], [1, 4, 1], [1, 2, 3]])
    qx, s = quantize_per_channel_with_indices(x, torch.tensor([1, 0, 1]))
    print(qx)
