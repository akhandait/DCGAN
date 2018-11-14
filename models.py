import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, inFeatures, LeakyReLU = False):
        super().__init__()

        convBlock = [nn.ZeroPad2d(1),
                     nn.Conv2d(inFeatures, inFeatures, 3, bias=False),
                     # nn.InstanceNorm2d(inFeatures)]
                     nn.BatchNorm2d(inFeatures)]

        if LeakyReLU:
            convBlock += [nn.LeakyReLU(inplace=True)]
        else:
            convBlock += [nn.ReLU(inplace=True)]

        convBlock += [nn.ZeroPad2d(1),
                     nn.Conv2d(inFeatures, inFeatures, 3, bias=False),
                     # nn.InstanceNorm2d(inFeatures)]
                     nn.BatchNorm2d(inFeatures)]

        self.convBlock = nn.Sequential(*convBlock)

    def forward(self, x):
        return x + self.convBlock(x)

class Generator(nn.Module):
    def __init__(self, nz=100, nc=3, ng=64, nResBlocks=0):
        super().__init__()

        # model = [nn.ConvTranspose2d(zFeatures, 512, 3, 1, 0, bias=False),
        #          nn.BatchNorm2d(512), # Try instance norm
        #          nn.ReLU(inplace=True),
        #          nn.ConvTranspose2d(512, 256, 3, 2, 1, output_padding = 1, bias=False),
        #          nn.BatchNorm2d(256),
        #          nn.ReLU(inplace=True)]

        # for _ in range(nResBlocks):
        #     model += [ResidualBlock(256)]

        # model += [nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding = 1, bias=False),
        #           nn.BatchNorm2d(128),
        #           nn.ReLU(inplace=True),
        #           nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding = 1, bias=False),
        #           nn.BatchNorm2d(64),
        #           nn.ReLU(inplace=True),
        #           nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding = 1, bias=False),
        #           nn.BatchNorm2d(32),
        #           nn.ReLU(inplace=True),
        #           nn.ConvTranspose2d(32, 16, 5, 1, 0, output_padding = 0, bias=False),
        #           nn.BatchNorm2d(16),
        #           nn.ReLU(inplace=True),
        #           nn.ConvTranspose2d(16, 16, 5, 1, 0, output_padding = 0, bias=False),
        #           nn.BatchNorm2d(16),
        #           nn.ReLU(inplace=True),
        #           nn.ConvTranspose2d(16, 16, 5, 1, 0, output_padding = 0, bias=False),
        #           nn.BatchNorm2d(16),
        #           nn.ReLU(inplace=True),
        #           nn.ConvTranspose2d(16, 3, 5, 1, 0, output_padding = 0, bias=False),
        #           nn.Tanh()] # Outputs 64x64x3 ############ Make general

        # self.model = nn.Sequential(*model)

        model = [nn.ConvTranspose2d(nz, ng * 8, 4, 1, 0, bias=False),
                 nn.BatchNorm2d(ng * 8),
                 nn.ReLU(inplace=True),
                 nn.ConvTranspose2d(ng * 8, ng * 4, 4, 2, 1, bias=False),
                 nn.BatchNorm2d(ng * 4),
                 nn.ReLU(inplace=True),
                 nn.ConvTranspose2d(ng * 4, ng * 2, 4, 2, 1, bias=False),
                 nn.BatchNorm2d(ng * 2),
                 nn.ReLU(inplace=True)]

        for _ in range(nResBlocks):
            model += [ResidualBlock(ng * 2)]

        model += [nn.ConvTranspose2d(ng * 2, ng, 4, 2, 1, bias=False),
                  nn.BatchNorm2d(ng),
                  nn.ReLU(inplace=True),
                  nn.ConvTranspose2d(ng, nc, 4, 2, 1, bias=False),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, nc=3, nd=64, nResBlocks=0):
        super().__init__()

        # model = [nn.Conv2d(imgFeatures, 16, 5, 1, 0, bias=False),
        #          nn.BatchNorm2d(16),
        #          nn.LeakyReLU(inplace=True),
        #          nn.Conv2d(16, 16, 5, 1, 0, bias=False),
        #          nn.BatchNorm2d(16),
        #          nn.LeakyReLU(inplace=True),
        #          nn.Conv2d(16, 16, 5, 1, 0, bias=False),
        #          nn.BatchNorm2d(16),
        #          nn.LeakyReLU(inplace=True),
        #          nn.Conv2d(16, 32, 5, 1, 0, bias=False),
        #          nn.BatchNorm2d(32),
        #          nn.LeakyReLU(inplace=True),
        #          nn.Conv2d(32, 64, 3, 2, 1, bias=False),
        #          nn.BatchNorm2d(64),
        #          nn.LeakyReLU(inplace=True),
        #          nn.Conv2d(64, 128, 3, 2, 1, bias=False),
        #          nn.BatchNorm2d(128),
        #          nn.LeakyReLU(inplace=True),
        #          nn.Conv2d(128, 256, 3, 2, 1, bias=False),
        #          nn.BatchNorm2d(256),
        #          nn.LeakyReLU(inplace=True)]

        # for _ in range(nResBlocks):
        #     model += [ResidualBlock(256, True)]

        # model += [nn.Conv2d(256, 512, 3, 2, 1, bias=False),
        #           nn.BatchNorm2d(512),
        #           nn.LeakyReLU(inplace=True),
        #           nn.Conv2d(512, 1, 3, 1, 0, bias=False),
        #           nn.Sigmoid()]

        # self.model = nn.Sequential(*model)

        model = [nn.Conv2d(nc, nd, 4, 2, 1, bias=False),
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.Conv2d(nd, nd * 2, 4, 2, 1, bias=False),
                 nn.BatchNorm2d(nd * 2),
                 nn.LeakyReLU(0.2, inplace=True)]

        for _ in range(nResBlocks):
            model += [ResidualBlock(nd * 2)]

        model += [nn.Conv2d(nd * 2, nd * 4, 4, 2, 1, bias=False),
                  nn.BatchNorm2d(nd * 4),
                  nn.LeakyReLU(0.2, inplace=True),
                  nn.Conv2d(nd * 4, nd * 8, 4, 2, 1, bias=False),
                  nn.BatchNorm2d(nd * 8),
                  nn.LeakyReLU(0.2, inplace=True),
                  nn.Conv2d(nd * 8, 1, 4, 1, 0, bias=False),
                  nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x).squeeze()

