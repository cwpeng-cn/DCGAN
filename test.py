import torch
from model import Generator

MODEL_PATH = './Net_G.pth'

net = Generator().eval()
net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))

x = torch.randn(1, 3, 100, 1)
traced_script_module = torch.jit.trace(func=net, example_inputs=x)
traced_script_module.save("Generator.pt")
