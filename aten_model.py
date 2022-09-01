import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_ort import ORTInferenceModule, DebugOptions


from torch.onnx import register_custom_op_symbolic

def triu(g, self, x):
    output = g.op("org.pytorch.aten::ATen", self, x, operator_s="triu")
    output.setType(self.type())
    return output


register_custom_op_symbolic("::triu", triu, 1)

class mynet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.empty(0)) 

    def forward(self, x):
        return torch.triu(x)

net = mynet()

model = ORTInferenceModule(net, DebugOptions(save_onnx=True, onnx_prefix='aten'))

x = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]]).type(torch.float32)

# Run PyTorch model first
out = net(x)
print(out)

# Run with torch-ort-infer
out = model(x)
print('out.size ', out.size()) #(1, 3, 480, 480)
print(out)
