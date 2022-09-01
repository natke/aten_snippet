import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_ort import ORTInferenceModule, DebugOptions
from onnxruntime.training.ortmodule.torch_cpp_extensions.cpu.aten_op_executor import load_aten_op_executor_cpp_extension
from torch.onnx import register_custom_op_symbolic

def triu(g, self, x):
    output = g.op("org.pytorch.aten::ATen", self, x, operator_s="triu")
    output.setType(self.type())
    return output


register_custom_op_symbolic("::triu", triu, 1)

load_aten_op_executor_cpp_extension()

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
#out = net(x)
#print(out)

# Run with torch-ort-infer
print(x)
print("Running triu")
out = model(x)
print(out)
