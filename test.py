import torch
from models.PSMnet import PSMNet 

torch.manual_seed(2.0)

model = PSMNet(16).cuda()
left = torch.randn(2, 3, 256, 256).cuda()
right = torch.randn(2, 3, 256, 256).cuda()
print(left[:, :, 0, 0])

out1, out2, out3 = model(left, right)
print(out2[0, :3, :3])