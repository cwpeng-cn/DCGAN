import torch
from torch import optim
from torch import nn
from torch.utils import data
from data import AnimeDataset, LossWriter, reminder
from model import Generator, Discriminator

DATA_DIR = "../datasets/selfie2anime/all"
MODEL_G_PATH = "./Net_G.pth"
MODEL_D_PATH = "./Net_D.pth"
LOG_G_PATH = "./Log_G.txt"
LOG_D_PATH = "./Log_D.txt"
IMAGE_SIZE = 64
BATCH_SIZE = 128
WORKER = 1
LR = 0.0002
NZ = 100
num_epochs = 300

dataset = AnimeDataset(dataset_path=DATA_DIR, image_size=IMAGE_SIZE)
data_loader = data.DataLoader(dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=WORKER)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
netD = Discriminator().to(device)
# Initialize BCELoss function
criterion = nn.BCELoss()
# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, 1, NZ, 1, device=device)
# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.
# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.999))
g_writer = LossWriter(save_path=LOG_G_PATH)
d_writer = LossWriter(save_path=LOG_D_PATH)

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print(dataset.__len__())

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for data in data_loader:

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        real_imgs = data[0].to(device)
        b_size = real_imgs.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_imgs).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, 1, NZ, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if iters % 5 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, iters, len(data_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            d_writer.add(loss=errD.item(), i=iters)
            g_writer.add(loss=errG.item(), i=iters)

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        iters += 1

reminder("Finish train:{}".format(iters))
torch.save(netG.state_dict(), MODEL_G_PATH, _use_new_zipfile_serialization=False)
torch.save(netD.state_dict(), MODEL_D_PATH, _use_new_zipfile_serialization=False)
with open(MODEL_G_PATH, 'rb') as f, open(
        '/content/drive/MyDrive/Colab/pytorch-neural-network-practice/{}'.format(MODEL_G_PATH.split('/')[-1]),
        "wb") as fw:
    fw.write(f.read())
with open(MODEL_D_PATH, 'rb') as f, open(
        '/content/drive/MyDrive/Colab/pytorch-neural-network-practice/{}'.format(MODEL_D_PATH.split('/')[-1]),
        "wb") as fw:
    fw.write(f.read())
with open(LOG_G_PATH, 'rb') as f, open(
        '/content/drive/MyDrive/Colab/pytorch-neural-network-practice/{}'.format(LOG_G_PATH.split('/')[-1]),
        "wb") as fw:
    fw.write(f.read())
with open(LOG_D_PATH, 'rb') as f, open(
        '/content/drive/MyDrive/Colab/pytorch-neural-network-practice/{}'.format(LOG_D_PATH.split('/')[-1]),
        "wb") as fw:
    fw.write(f.read())
reminder("Finish save:{}".format(iters))
