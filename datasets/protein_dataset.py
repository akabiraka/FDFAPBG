import sys
sys.path.append("../FDFAPBG")
import numpy as np
import math

import torch
from torch.utils.data import Dataset
import utils.data_utils as DataUtils
import configs.general_config as CONFIGS

class ProteinDataset(Dataset):
    def __init__(self):
        super(ProteinDataset, self).__init__()

    def __len__(self):
        return 0

    def __getitem__(self, i):
        pass

    def generate_inp_out_sets(self):
        pdb_id, chain_id = "5sy8", "O"
        full_c_map = DataUtils.read_contact_map_tensor(pdb_id + chain_id)
        full_d3_coords = DataUtils.read_3d_coords_tensor(pdb_id + chain_id)
        # print(full_c_map.shape, d3_coords.shape)
        rows, cols = full_c_map.shape
        half_width = math.floor(CONFIGS.WINDOW_SIZE / 2)
        a_input_output_set = []
        all_input_output_set = []
        for i in range(half_width, rows - half_width, CONFIGS.WINDOW_STRIDE):
            s1_from_idx = i - half_width
            s1_to_idx = i + half_width
            d3_coords_out = full_d3_coords[s1_from_idx:s1_to_idx]
            for j in range(half_width, rows - half_width, CONFIGS.WINDOW_STRIDE):
                s2_from_idx = j - half_width
                s2_to_idx = j + half_width
                
                c_map_inp = full_c_map[s1_from_idx:s1_to_idx, s2_from_idx:s2_to_idx]
                c_map_inp = c_map_inp.type(torch.float32)
                # print(s1_from_idx, s1_to_idx, s2_from_idx, s2_to_idx)
                # print(c_map_inp.shape, d3_coords_out.shape)
                a_input_output_set = [c_map_inp, d3_coords_out]
                all_input_output_set.append(a_input_output_set)
        return all_input_output_set

pd = ProteinDataset()
# a contact map is divided into fixed size contact-map and 3d coordinate matrix
print(len(pd.generate_inp_out_sets())) 
# accessing the 0th [inp, out] set, where inp=fixed_size_contact_map, out=fixed_side_3d_coords_matrix
print(len(pd.generate_inp_out_sets()[0]))
# accessing the 0th [inp, out] pair's contact-map/distance-matrix shape
print(pd.generate_inp_out_sets()[0][0].shape)
# accessing the 0th [inp, out] pair's 3d-coordinate-matrix shape
print(pd.generate_inp_out_sets()[0][1].shape)