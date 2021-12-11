import torch
from model import Generator

MODEL_PATH = './Net_G.pth'

net = Generator().eval()
net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))

x = torch.randn(1, 100)
traced_script_module = torch.jit.trace(func=net, example_inputs=x)
traced_script_module.save("DCGAN_Generator.pt")
