import torch
import pylab as plt
import numpy as np
from model import Generator
from data import recover_image

MODEL_PATH = './Net_G.pth'

net = Generator().eval()
net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))

x = torch.randn(25, 100)
data = net(x).detach()
images = recover_image(data)
full_image = np.full((5 * 64, 5 * 64, 3), 0, dtype="uint8")
for i in range(25):
    row = i // 5
    col = i % 5
    full_image[row * 64:(row + 1) * 64, col * 64:(col + 1) * 64, :] = images[i]

plt.imshow(full_image)
plt.show()
plt.imsave("hah.png", full_image)

# x = torch.randn(1, 100)
# traced_script_module = torch.jit.trace(func=net, example_inputs=x)
# traced_script_module.save("Generator.pt")
