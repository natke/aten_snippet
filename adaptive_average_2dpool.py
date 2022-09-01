import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_ort import ORTInferenceModule, OpenVINOProviderOptions


provider_options = OpenVINOProviderOptions(backend = "CPU", precision = "FP32")

class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0)) 


    def forward(self, x):
        n, c, h, w = x.size()
        x = F.adaptive_avg_pool2d(x, (1, 1))

        #return F.interpolate(x, (h, w), mode='bilinear', align_corners=False)
                                # RuntimeError: ONNX symbolic expected a constant value in the trace

        #return F.interpolate(x, (480, 640), mode='bilinear', align_corners=False) # this is ok

        #return F.interpolate(x, (h, w), mode='bilinear', align_corners=True)
                                # RuntimeError: ONNX symbolic expected a constant value in the trace

        return F.interpolate(x, (480, 640), mode='bilinear', align_corners=True)
                                # UserWarning: ONNX export failed on upsample_bilinear2d because align_corners == True not supported
                                # RuntimeError: ONNX export failed: Couldn't export operator aten::upsample_bilinear2d

device = "cpu"

net = mynet().to(device)

print(net.dummy_param.device)

model = ORTInferenceModule(net)


x = torch.randn(1, 3, 480, 640).to(device)

out = model(x)
print('out.size ', out.size()) #(1, 3, 480, 640)

#torch.onnx.export(net, x, "test.onnx", verbose=True)
