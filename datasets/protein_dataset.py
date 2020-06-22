import sys
sys.path.append("../FDFAPBG")
import numpy as np
import math
import pickle

import torch
from torch.utils.data import Dataset
import utils.data_utils as DataUtils
import configs.general_config as CONFIGS

class ProteinDataset(Dataset):
    def __init__(self):
        super(ProteinDataset, self).__init__()
        self.records = DataUtils.load_using_pickle(CONFIGS.FULL_DATA_FILE)

    def __len__(self):
        """
        Returns the length of the full records.
        """
        return len(self.records)

    def __getitem__(self, i):
        """
        Returns a single record [inp, out]
        inp: contact-map/distance-map matrix
        out: 3d coordinate matrix.
        """
        return self.records[i]

pd = ProteinDataset()
print(pd.__len__())
print(len(pd.__getitem__(0)))
# # a contact map is divided into fixed size contact-map and 3d coordinate matrix
# print(len(pd.get_inp_out_sets("5sy8" + "O"))) 
# # accessing the 0th [inp, out] set, where inp=fixed_size_contact_map, out=fixed_side_3d_coords_matrix
# print(len(pd.get_inp_out_sets("5sy8" + "O")[0]))
# # accessing the 0th [inp, out] pair's contact-map/distance-matrix shape
# print(pd.get_inp_out_sets("5sy8" + "O")[0][0].shape)
# # accessing the 0th [inp, out] pair's 3d-coordinate-matrix shape
# print(pd.get_inp_out_sets("5sy8" + "O")[0][1].shape)