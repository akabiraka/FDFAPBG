# Task: reconstruction of distance-matrix to distance-matrix using autoencoder
# Model: Autoencoder
import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.autoencoder import Autoencoder
from datasets.protein_dataset import ProteinDataset
import configs.general_config as CONFIGS

lrs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
for ith_start, lr in enumerate(lrs):
    print("ith_start=", ith_start)
    print("declaring variables and hyperparameters... ... ")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Autoencoder()
    model.to(device)
    criterion = nn.MSELoss(reduction='mean')
    init_lr = lr #1e-5
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    batch_size = 30
    n_epochs = 50
    print_every = 2
    test_every = 5
    plot_every = 5 
    print("device=", device)
    print("batch_size=", batch_size)
    print("n_epochs=", n_epochs)
    print("init_lr=", init_lr) 
    print("loss=MSE mean")
    print(model)

    print("loading training dataset ... ...")
    train_dataset = ProteinDataset(file=CONFIGS.TRAIN_FILE)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("train dataset len:", train_dataset.__len__())
    x, _ = train_dataset.__getitem__(0)
    print(x.shape)
    print("train loader size:", len(train_loader))
    print("successfully loaded training dataset ... ...")

    print("loading validation dataset ... ...")
    val_dataset = ProteinDataset(file=CONFIGS.VAL_FILE)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    print("val dataset len:", val_dataset.__len__())
    print("val loader size: ", len(val_loader))
    print("successfully loaded validation dataset ... ...")

    print("loading test dataset ... ...")
    test_dataset = ProteinDataset(file=CONFIGS.TEST_FILE)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print("test dataset len:", test_dataset.__len__())
    print("test loader size: ", len(test_loader))
    print("successfully loaded test dataset ... ...")

    def train():
        model.train()
        loss = 0.0
        losses = []
        for i, (x, y) in enumerate(train_loader):
            x = x.unsqueeze(dim=1).to(device)
            optimizer.zero_grad()
            # print(x.shape)
            x_prime, z = model(x)
            # print("x_prime: {}, x: {}".format(x_prime.shape, x.shape))
            loss = criterion(x_prime, x)
            # print(loss)
            loss.backward()
            optimizer.step()
            losses.append(loss)
            # break
        return torch.stack(losses).mean().item()

    # print(train())

    def test(data_loader):
        model.eval()
        loss = 0.0
        losses = []
        with torch.no_grad():
            for i, (x, _) in enumerate(data_loader):
                x = x.unsqueeze(dim=1).to(device)
                x_prime, z = model(x)
                loss = criterion(x_prime, x)
                # print(loss)
                losses.append(loss)
        return torch.stack(losses).mean().item()

    # print(test(test_loader))
    # print(test(val_loader))

    train_losses = []
    val_losses = []
    best_test_loss = np.inf
    for epoch in range(1, n_epochs + 1):
        print("\nStarting epoch {}/{}".format(epoch, n_epochs + 1))

        train_loss = train()
        train_losses.append(train_loss)

        if epoch % print_every == 0:
            print("epoch:{}/{}, train_loss: {:.7f}".format(epoch, n_epochs + 1, train_loss))
            for param_group in optimizer.param_groups:
                print("learning rate: ", param_group['lr'])

        if epoch % test_every == 0:
            print("Starting testing epoch {}/{}".format(epoch, n_epochs + 1))
            val_loss = test(val_loader)
            print("epoch:{}/{}, val_loss: {:.7f}".format(epoch, n_epochs + 1, val_loss))
            val_losses.append(val_loss)
            if val_loss < best_test_loss:
                best_test_loss = val_loss
                print('Updating best val loss: {:.7f}'.format(best_test_loss))
                torch.save(model.state_dict(), 'outputs/models/ae_cmap_recon_{}.pth'.format(ith_start))

        if epoch % plot_every == 0:
            pass
            # plt.plot(train_losses)
            # plt.plot(val_losses)
            # plt.show()
            # plt.savefig("outputs/raw_img_with_raw_pt_2_{}.jpg".format(epoch))

    print("train losses:", train_losses)
    print("val losses:", val_losses)