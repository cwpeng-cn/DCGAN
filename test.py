import os
import time
import torch
from PIL import Image
from model import Generator
from torchvision import transforms

MODEL_PATH = './Net_G.pth'

net = Generator().eval()
net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))

x = torch.randn(1, 100, 1, 1)
traced_script_module = torch.jit.trace(func=net, example_inputs=x)
traced_script_module.save("Generator.pt")
