import sys
sys.path.append("../FDFAPBG")

import numpy as np
import torch
from models.rmsd_loss import RMSD_loss
from Bio.SVDSuperimposer import SVDSuperimposer

for row in [16, 32, 64, 128, 256]:
    size = (row, 3)
    print(size)
    for i in range(100):
        x = np.random.uniform(low=-30.0, high=30.0, size=size)
        y = np.random.uniform(low=-30.0, high=30.0, size=size)

        sup = SVDSuperimposer()
        sup.set(x, y)
        sup.run()
        bio_rmsd = sup.get_rms()

        rmsdLoss = RMSD_loss()
        my_rmsd = rmsdLoss.rmsd(torch.tensor(x).unsqueeze(dim=0), torch.tensor(y).unsqueeze(dim=0))
        print("bio-rmsd: {:.3f}, our-rmsd: {:.3f}, diff: {:.3f}".format(bio_rmsd, my_rmsd.item(), bio_rmsd-my_rmsd.item()))
