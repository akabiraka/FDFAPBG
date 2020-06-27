import torch
import torch.nn as nn

class RMSD_loss(nn.Module):
    """
    Code idea is taken from:
    [1] https://towardsdatascience.com/tensorflow-rmsd-using-tensorflow-for-things-it-was-not-designed-to-do-ada4c9aa0ea2
    [2] https://github.com/mdtraj/tftraj/blob/master/tftraj/rmsd.py
    """
    def __init__(self):
        super(RMSD_loss, self).__init__()

    def optimal_rotational_quaternion(self, r):
        """Just need the largest eigenvalue of this to minimize RMSD over rotations
        
        References
        ----------
        [1] http://dx.doi.org/10.1002/jcc.20110
        """
        # @formatter:off
        return [
            [r[0][0] + r[1][1] + r[2][2], r[1][2] - r[2][1], r[2][0] - r[0][2], r[0][1] - r[1][0]],
            [r[1][2] - r[2][1], r[0][0] - r[1][1] - r[2][2], r[0][1] + r[1][0], r[0][2] + r[2][0]],
            [r[2][0] - r[0][2], r[0][1] + r[1][0], -r[0][0] + r[1][1] - r[2][2], r[1][2] + r[2][1]],
            [r[0][1] - r[1][0], r[0][2] + r[2][0], r[1][2] + r[2][1], -r[0][0] - r[1][1] + r[2][2]],
        ]

    def translate(self, x):
        """
        x: (batch_size, n_atoms, 3)
        """
        x_translated = x - x.mean(dim=1, keepdim=True)
        return x_translated

    def squared_deviation(self, y_prime, y):
        """
        y_prime, y: (n_atoms, 3)
        """
        R = torch.matmul(y_prime.t(), y) # ordinary cross-correlation of xyz coordinates. (3, 3)
        R_parts = [ torch.unbind(t) for t in torch.unbind(R)]
        F_parts = self.optimal_rotational_quaternion(R_parts)
        F = torch.tensor(F_parts)
        # backward pass in only supported in symeig
        # returned eigenvalues are in acending order
        vals, vecs = torch.symeig(F, eigenvectors=True)
        lamx = vals[-1] # getting the max eigen value
        sd = torch.sum(y_prime**2 + y**2) - 2 * lamx
        return sd

    def rmsd(self, y_prime, y):
        """
        y_prime, y: (batch_size, n_atoms, 3)
        """
        batch_size, n_atoms, d = y_prime.shape
        y_prime_translated = self.translate(y_prime)
        y_translated = self.translate(y)
        total_sd = 0.0
        for i in range(batch_size):
            total_sd += self.squared_deviation(y_prime[i], y[i]) 
        return torch.sqrt(total_sd/n_atoms)

    def forward(self, y_prime, y):
        """
        y_prime: (batch_size, n_coords, 3) 
        y: (batch_size, n_coords, 3)
        """
        return self.rmsd(y_prime, y)

# y = torch.randn((30, 256, 3))
# y_prime = torch.randn((30, 256, 3), requires_grad=True)
# rmsd_loss = RMSD_loss()
# loss = rmsd_loss(y_prime, y)
# print(loss)