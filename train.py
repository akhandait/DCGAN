import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import Discriminator, Generator
from utils import weightsInit
from data.dataset import CelebA
import torchvision.utils as vutils

import time

cudaAvailable = False
if torch.cuda.is_available():
    cudaAvailable = True

# Networks.
G = Generator(nResBlocks = 3)
D = Discriminator(nResBlocks = 3)

if cudaAvailable:
    G.cuda()
    D.cuda()

G.apply(weightsInit)
D.apply(weightsInit)

loss = nn.BCELoss()

# Learning rates suggested in the DCGAN paper are used.
optimizerG = torch.optim.Adam(G.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerD = torch.optim.Adam(D.parameters(), lr = 0.0002, betas = (0.5, 0.999))
# print(list(G.parameters())[4].shape)

# Inputs and targets memory allocation.
Tensor = torch.cuda.FloatTensor if cudaAvailable else torch.Tensor

# Hyperparameters.
batchSize = 64
nofEpochs = 5

# Path to data.
imgDirectory = '../Datasets/CelebA/celebA_64x64/'

dataloader = DataLoader(CelebA(imgDirectory), batch_size = batchSize,
    shuffle = True, num_workers = 8)

nofImages = len(dataloader)

realLabel = 1
fakeLabel = 0

fixedNoise = torch.randn(64, 100, 1, 1).type(Tensor)
niterD = 1
niterG = 1

# startTime = time.time()
for epoch in range(nofEpochs):
    for i, batch in enumerate(dataloader):
        print(i)
        batchS = batch.shape[0]
        # print(batch[0])
        # vutils.save_image(batch.detach(),
                    # 'output/real_samples.png', normalize=True)
        # if epoch == 0 and i == 1500:
        #     niterG = 4
        # elif epoch == 0 and i == 2500:
        #     niterG = 3
        # elif epoch == 1 and i == 1000:
        #     niterG = 2

        for j in range(niterD):
        # while ld > lg / 4:
            D.zero_grad()
            output = D(batch.type(Tensor))
            label = torch.full((batchS,), realLabel).type(Tensor)
            ldreal = loss(output, label)
            ldreal.backward()
            # print(D.model[-2].weight.grad.norm())

            noise = torch.randn(batchS, 100, 1, 1).type(Tensor)
            output = G(noise)
            # print(output[0])
            # print("real data -> " + str(batch[0]))
            # print("Generator output ->" + str(output[0]))
            outputd = D(output.detach())
            label = label.fill_(fakeLabel)
            ldfake = loss(outputd, label)
            ldfake.backward()

            ld = ldreal + ldfake
            print("Discriminator loss -> " + str(ld.item()))

            optimizerD.step()

        for k in range(niterG):
        # while lg > ld * 4:
            G.zero_grad()

            noise = torch.randn(batchS, 100, 1, 1).type(Tensor) # Should we use the old sampled one?
            output = G(noise)
            outputd = D(output)
            label = label.fill_(realLabel)
            lg = loss(outputd, label)
            lg.backward()
            # print(G.model[0].weight.grad.norm())

            print("Generator loss -> " + str(lg.item()))

            optimizerG.step()


        # print(label.shape)
        # l = loss(output, label)
        # print(l)
        # l.backward()
        # optimizerD.step()
        # if i == 100:
        #     t = time.time() - startTime
        #     print("Time " + str(t))

        if i % 50 == 0:
            # print(i)
            # print("Discriminator loss -> " + str(ld.item()))
            # print("Generator loss -> " + str(lg.item()))

            testimg = G(fixedNoise)
            vutils.save_image(testimg.detach(),
                    'output/fake_samples_%03d_%03d.png' % (epoch, i), normalize=True)

    torch.save(G.state_dict(), 'saved_models/G_2epoch_%d.pth' % (epoch))
    torch.save(D.state_dict(), 'saved_models/D_2epoch_%d.pth' % (epoch))


