import sys
sys.path.append("../FDFAPBG")
import numpy as np
import pickle

from torch.utils.data import Dataset
import configs.general_config as CONFIGS

class ProteinDataset(Dataset):
    def __init__(self):
        super(ProteinDataset, self).__init__()
        with open(CONFIGS.FULL_DATA_FILE, 'rb') as filehandle:
            self.records = pickle.load(filehandle)

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
# accessing a fixed size contact-map/distance-matrix and 3d-coordinate matrix
print(pd.__getitem__(0)[0].dtype, pd.__getitem__(0)[1].dtype)