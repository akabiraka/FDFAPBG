import sys
sys.path.append("../FDFAPBG")

import numpy as np
import torch
from models.rmsd_loss import RMSD_loss
from models.rmsd_loss_1 import RMSDLoss
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.PDB.Superimposer import Superimposer

sup = SVDSuperimposer()
myRmsdLoss = RMSD_loss()
proteinTorchLibRMSDLoss = RMSDLoss(device='cpu')

for row in [16, 32, 64, 128, 256]:
    size = (row, 3)
    print(size)
    for i in range(10):
        x = np.random.uniform(low=-30.0, high=30.0, size=size)
        y = np.random.uniform(low=-30.0, high=30.0, size=size)
        
        # x = np.random.randn(size[0], size[1])
        # y = np.random.randn(size[0], size[1])
        
        sup.set(x, y)
        sup.run()
        bio_rmsd = sup.get_rms()
        
        my_rmsd = myRmsdLoss(torch.tensor(x).unsqueeze(dim=0), torch.tensor(y, requires_grad=True).unsqueeze(dim=0))
        
        ptb_rmsd = proteinTorchLibRMSDLoss(torch.tensor(x).unsqueeze(dim=0), torch.tensor(y, requires_grad=True).unsqueeze(dim=0))
        
        # print(bio_rmsd, my_rmsd, ptb_rmsd)
        print("bio-rmsd: {:.3f}, our-rmsd: {:.3f}, ptb_rmsd: {:.3f}".format(bio_rmsd, my_rmsd, ptb_rmsd))