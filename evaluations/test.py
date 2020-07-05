import sys
sys.path.append("../FDFAPBG")
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from sklearn.metrics import mean_squared_error
import pickle

import matplotlib.pyplot as plt
import numpy as np

from models.architectures import Recover
from datasets.protein_dataset import ProteinDataset
from models.rmsd_loss import RMSD_loss
import utils.data_utils as DataUtils


def test(test_set):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    DCNN = Recover().to(device)
    DCNN.load_state_dict(torch.load('outputs/models/pretrained_DCNN_model_compatible'))
    DCNN.eval()
    rmsd = RMSD_loss().cuda()
    rmsd_scores = []
    total_losses = []
    with torch.no_grad():
        for i, data in enumerate(test_set, 0):
            # print(i)
            label = (data[1].unsqueeze(dim=1)).to(device)
            data_X = (data[0].unsqueeze(dim=1)).type(torch.FloatTensor).to(device)
            output = DCNN(data_X)
            batch_size, _, n_coords, d = label.shape
            output_prime = output.view(batch_size, n_coords, d)
            target_prime = label.view(batch_size, n_coords, d)
            loss_rb = rmsd(output_prime, target_prime)
            rmsd_scores.append(loss_rb.item())
            gram_matrix = torch.matmul(output, torch.transpose(output,2,3))
            mse_ls = nn.MSELoss()
            loss_ae = mse_ls(gram_matrix, data_X)/torch.mean(data_X**2)
            total_losses.append((loss_ae+loss_rb).mean().item())
            # if i==5: break
    print("total_losses =",total_losses)
    print("rmsd_scores =",rmsd_scores)
    print("avg_loss =",np.mean(total_losses))
    print("avg_rmsd =",np.mean(rmsd_scores))
    print("min_rmsd =",min(rmsd_scores))
    print("max_rmsd =",max(rmsd_scores))

def prepare_test_dataset():
    pd = ProteinDataset(data_dir="data/1hz6A_c_map_vs_coord_pairs/", file="data/1hz6A_record_ids.txt")
    print("datasets =",pd.__len__())
    protein_data_loader = torch.utils.data.DataLoader(pd, batch_size=1, shuffle=False)
    return protein_data_loader

if __name__ == "__main__":
    test_set = prepare_test_dataset()
    test(test_set)

# run this file as the following way:
# python evaluations/test.py > outputs/logs/1fwpA_eval_result.py