import torch

a = torch.randn(18, 256, 864, device='cuda', dtype=torch.half)
w = torch.randn(256, 1, 33, device='cuda', dtype=torch.half)

torch.nn.functional.conv1d(a, w, bias=None, stride=1, padding=16, dilation=1, groups=256)
