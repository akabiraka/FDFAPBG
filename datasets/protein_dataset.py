import sys
sys.path.append("../FDFAPBG")
import numpy as np
import pickle

from torch.utils.data import Dataset
import configs.general_config as CONFIGS
import utils.data_utils as DataUtils

class ProteinDataset(Dataset):
    def __init__(self, file=CONFIGS.TRAIN_FILE):
        """
        file should have an id per line, this id will used to access the data.
        """
        super(ProteinDataset, self).__init__()
        self.record_ids = DataUtils.get_ids(file)

    def __len__(self):
        """
        Returns the length of the full records.
        """
        return len(self.record_ids)

    def __getitem__(self, i):
        """
        Returns a single record [inp, out]
        inp: contact-map/distance-map matrix
        out: 3d coordinate matrix.
        """
        filename = CONFIGS.CONTACT_MAP_VS_COORDINATES_DIR + self.record_ids[i] + CONFIGS.DOT_PKL
        return DataUtils.load_using_pickle(filename)

# pd = ProteinDataset(file=CONFIGS.VAL_FILE)
# print(pd.__len__())
# print(len(pd.__getitem__(0)))
# # accessing a fixed size contact-map/distance-matrix and 3d-coordinate matrix
# print(pd.__getitem__(0)[0].shape, pd.__getitem__(0)[1].shape)