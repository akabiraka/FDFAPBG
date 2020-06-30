import sys
sys.path.append("../FDFAPBG")

import torch
import torch.nn as nn

# num of input channels
nc = 1
# size of the latent space
nz = 100
# size of feature maps
nf = 32

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # input: nc x 256 x 256, 
            nn.Conv2d(nc, nf, 9, 2, 4),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),
            # state: nf x 128 x 128
            nn.Conv2d(nf, nf*2, 5, 2, 2),
            nn.BatchNorm2d(nf*2),
            nn.ReLU(inplace=True),
            # state: nf*2 x 64 x 64
            nn.Conv2d(nf*2, nf*4, 5, 4, 2),
            nn.BatchNorm2d(nf*4),
            nn.ReLU(inplace=True),
            # state: nf*4 x 16 x 16
            nn.Conv2d(nf*4, nf*8, 5, 4, 2),
            nn.BatchNorm2d(nf*8),
            nn.ReLU(inplace=True),
            # state: nf*8 x 4 x 4
            nn.Conv2d(nf*8, nf*16, 4, 4, 0),
            nn.ReLU(inplace=True),
            # # state: nf*16 x 1 x 1
        )

        self.fc1 = nn.Linear(nf*16*1*1, 256)
        self.fc2 = nn.Linear(256, nz)
        self.fc3 = nn.Linear(nz, 256)
        self.fc4 = nn.Linear(256, nf*16)

        self.decoder = nn.Sequential(
            # state: nf*16 x 1 x 1
            nn.ConvTranspose2d(nf*16, nf*8, 4, 4, 0),
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(inplace=True),
            # state: nf*8 x 4 x 4
            nn.ConvTranspose2d(nf*8, nf*4, 8, 4, 2),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(inplace=True),
            # state: nf*4 x 16 x 16
            nn.ConvTranspose2d(nf*4, nf*2, 8, 4, 2),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(inplace=True),
            # state: nf*2 x 64 x 64
            nn.ConvTranspose2d(nf*2, nf, 6, 2, 2),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),
            # state: nf x 128 x 128
            nn.ConvTranspose2d(nf, nc, 10, 2, 4),
            nn.Sigmoid()
            # state: nc x 256 x 256
        )

    def encode(self, x):
        x = self.encoder(x)
        # print("encoded shape:", x.shape, x.view(x.shape[0], -1).shape)
        x = self.fc1(x.view(x.shape[0], -1))
        # print("encoded shape:", x.shape)
        z = self.fc2(x)
        # print("latent space:", z.shape)
        return z

    def decode(self, z):
        x = self.fc3(z)
        x = self.fc4(x)
        x_prime = self.decoder(x.view(-1, nf*16, 1, 1))
        # print("decoded shape:", x_prime.shape)
        return x_prime

    def forward(self, x):
        # print("inp shape:", x.shape)
        z = self.encode(x)
        o = self.decode(z)
        return o, z

AE = Autoencoder()
# # print(AE)
x = torch.randn((20, 1, 256, 256), dtype=torch.float32, requires_grad=True)
x_prime, z = AE(x)
# criterion = nn.MSELoss(reduction='mean')
# loss = criterion(x_prime, x)
# print(loss)
# loss.backward()
# # print(x.grad)
