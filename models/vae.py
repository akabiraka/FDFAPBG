import sys
sys.path.append("../FDFAPBG")

import torch
import torch.nn as nn
# from torch.autograd import Variable
import torch.nn.functional as F

# num of input channels
nc = 1
# size of the latent space
nz = 100
# size of feature maps
nf = 32

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.have_cuda = True if torch.cuda.is_available() else False
        self.encoder = nn.Sequential(
            # input: nc x 256 x 256, 
            nn.Conv2d(nc, nf, 9, 2, 4),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # state: nf x 128 x 128
            nn.Conv2d(nf, nf*2, 5, 2, 2),
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # state: nf*2 x 64 x 64
            nn.Conv2d(nf*2, nf*4, 5, 4, 2),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # state: nf*4 x 16 x 16
            nn.Conv2d(nf*4, nf*8, 5, 4, 2),
            nn.BatchNorm2d(nf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # state: nf*8 x 4 x 4
            nn.Conv2d(nf*8, nf*16, 4, 4, 0),
            nn.LeakyReLU(0.2, inplace=True),
            # # state: nf*16 x 1 x 1
        )
        self.fc1 = nn.Linear(nf*16*1*1, 256)
        self.mean = nn.Linear(256, nz)
        self.var = nn.Linear(256, nz)
        self.fc2 = nn.Linear(nz, 256)
        self.fc3 = nn.Linear(256, nf*16*1*1)
        self.relu = nn.ReLU()

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
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.shape[0], -1)
        # print("encoded shape:", encoded.shape)
        h1 = self.fc1(encoded)
        mu = self.mean(h1)
        logvar = self.var(h1)
        # print("mean:", mu.size(), "variance:", logvar.size())
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.have_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        # eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h2 = self.relu(self.fc2(z))
        deconv_input = self.fc3(h2)
        # print("deconv_input:", deconv_input.size())
        deconv_input = deconv_input.view(-1, nf*16, 1, 1)
        # print("deconv_input:", deconv_input.size())
        return self.decoder(deconv_input)

    def forward(self, x):
        # print("inp shape:", x.shape)
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        # print("z shape:", z.size())
        decoded = self.decode(z)
        # print("decoded:", decoded.size())
        return decoded, mu, logvar

class VAELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(VAELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)
        self.kld = nn.KLDivLoss(reduction=reduction)

    def forward_1(self, input, target):
        """
            loss = reconstruction + KLD
        """
        bce_loss = self.bce(input, target)
        kld_loss = self.kld(input, target)
        # print(bce_loss, kld_loss)
        loss = bce_loss + kld_loss
        return loss

    def forward(self, input, target, mu, logvar):
        """
            loss = reconstruction + KLD
        """
        bce_loss = self.bce(input, target)
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # print(bce_loss, kld_loss)
        loss = bce_loss + kld_loss
        return loss


# model = VAE()
# model.to(device='cuda')
# x = torch.randn((5, 1, 256, 256), dtype=torch.float32, requires_grad=True, device='cuda')
# x_prime, mu, logvar = model(x)
# criterion = VAELoss(reduction='mean')
# # loss = criterion(x_prime, x)
# loss = criterion(x_prime, x, mu, logvar)
# print(loss)
# loss.backward()
# # print(x.grad)
