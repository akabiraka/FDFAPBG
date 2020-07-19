import sys
sys.path.append("../FDFAPBG")

import numpy as np
import cvxpy as cp
import time
import torch
from sklearn.decomposition import TruncatedSVD
from datasets.protein_dataset import ProteinDataset
from models.rmsd_loss import RMSD_loss

RMSD = RMSD_loss()

def normalize(x):
    xmax, xmin = x.max(), x.min()
    x = (x - xmin)/(xmax - xmin)
    return x

def solve_sdp(dist_map):
    mat_len = dist_map.shape[0]
    X = cp.Variable((mat_len,mat_len))
    constraints = [X >> 0]
    prob = cp.Problem(cp.Minimize(np.sum(([(X[i][i]+X[j][j]-2*X[i][j]-dist_map[i][j]**2)**2 for i in range(mat_len) for j in range(mat_len)]))),constraints)
    #prob = cp.Problem(cp.Minimize(cp.trace(X)), constraints)
    prob.solve()

    # print (X.value)
    svd = TruncatedSVD(n_components=3, n_iter=7, random_state=42)
    svd.fit(X.value)
    res = svd.transform(X.value)
    return res

def compute_rmsd_loss(dist_map, coords):
    rmsd_loss = torch.tensor(0.0)
    predicted_3d_coords = normalize(solve_sdp(dist_map))
    predicted_3d_coords = torch.tensor(predicted_3d_coords, dtype=torch.float32).unsqueeze(dim=0)
    # print(CA_coords.shape, CA_dist_map.shape, predicted_3d_coords.shape)
    rmsd_loss = RMSD(y_prime=predicted_3d_coords, y=coords)
    return rmsd_loss.numpy()

def test(test_set, log_file=None, n_models=None):
    log_file_handle = open(log_file, "a")
    log_file_handle.write("CA_rmsd_loss, CB_rmsd_loss, N_rmsd_loss, O_rmsd_loss, total_rmsd_loss, total_time\n")
    for i, data in enumerate(test_set, 0):
        print("testing {}th model for ADMM ... ...".format(i+1))
        
        dist_map = data[0].squeeze(dim=0)
        native_3d_coords = data[1].squeeze(dim=0)
        
        start_time = time.time()
        
        CA_dist_map = dist_map[0:256:4, 0:256:4]
        CA_coords = native_3d_coords[0:256:4].unsqueeze(dim=0)
        CA_rmsd_loss = compute_rmsd_loss(CA_dist_map, CA_coords)
        
        CB_dist_map = dist_map[1:256:4, 1:256:4]
        CB_coords = native_3d_coords[1:256:4].unsqueeze(dim=0)
        CB_rmsd_loss = compute_rmsd_loss(CB_dist_map, CB_coords)
        
        N_dist_map = dist_map[2:256:4, 2:256:4]
        N_coords = native_3d_coords[2:256:4].unsqueeze(dim=0)
        N_rmsd_loss = compute_rmsd_loss(N_dist_map, N_coords)
        
        O_dist_map = dist_map[3:256:4, 3:256:4]
        O_coords = native_3d_coords[3:256:4].unsqueeze(dim=0)
        O_rmsd_loss = compute_rmsd_loss(O_dist_map, O_coords)
        
        total_loss = CA_rmsd_loss + CB_rmsd_loss + N_rmsd_loss + O_rmsd_loss
        run_time = (time.time() - start_time)/60
        
        print(CA_rmsd_loss, CB_rmsd_loss, N_rmsd_loss, O_rmsd_loss, total_loss, run_time)
        log_file_handle.write("{} {} {} {} {} {}\n".format(str(CA_rmsd_loss), str(CB_rmsd_loss), str(N_rmsd_loss), str(O_rmsd_loss), str(total_loss), str(run_time)))
        # log_file_handle.write(str(CA_rmsd_loss)+", "+str(CB_rmsd_loss)+", "+\
        #     str(N_rmsd_loss)+", "+str(O_rmsd_loss)+", "+str(total_loss)+", "+str(run_time) + "\n")
        
        if n_models is not None and (i+1)==n_models: break

def prepare_test_dataset(data_dir, file):
    pd = ProteinDataset(data_dir=data_dir, file=file)
    print("datasets =",pd.__len__())
    protein_data_loader = torch.utils.data.DataLoader(pd, batch_size=1, shuffle=False)
    return protein_data_loader

if __name__ == "__main__":
    test_set = prepare_test_dataset(data_dir="ADMM/data/2h5ndA_c_map_vs_coord_pairs/", file="ADMM/data/2h5ndA_record_ids.txt")
    test(test_set, log_file="ADMM/outputs/results/2h5ndA_test_admm.csv", n_models=120)