import sys
sys.path.append('../FDFAPBG')

import torch
import torch.nn as nn
from TorchProteinLibrary import RMSD

class RMSDLoss(nn.Module):
    """
    Rigid body alignment loss using RMSD (root mean squared deviation) loss.
    This use TorchProteinLibrary's RMSD implementation.
    """

    def __init__(self, device):
        super(RMSDLoss, self).__init__()
        self.device = device
        if 'cuda' in device:
            self.rmsd = RMSD.Coords2RMSD().cuda()
        else:
            self.rmsd = RMSD.Coords2RMSD()


    def forward(self, y_prime, y):
        """
        y_prime: (batch_size, n_coords, d) d-dimensional coordinates
        y: (batch_size, n_coords, d) d-dimensional coordinates
        """
        batch_size, n_coords, d = y_prime.shape
        n_atoms = torch.ones(batch_size, dtype=torch.int) * n_coords
        y_prime = y_prime.view(batch_size, n_coords * d)
        y = y.view(batch_size, n_coords * d)

        return self.rmsd(y_prime, y, n_atoms).mean()

# # working example
# y = torch.randn(5, 5, 3) # ground truth value
# y_prime = torch.randn(5, 5, 3, requires_grad=True) # predicted value
# y_prime = y_prime + torch.randn(5, 5, 3)

# criterion = RMSDLoss(device='cpu')
# loss = criterion(y_prime, y)
# print(loss)
# loss.backward()
