import sys
sys.path.append("../FDFAPBG")
import numpy as np
import pickle
from skimage.util import random_noise

import torch
from torch.utils.data import Dataset

import configs.general_config as CONFIGS
import utils.data_utils as DataUtils
import vizualizations.data_viz as DataViz

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

    def add_gaussian_noise(self, dist_mat, mean=0, var=0.05):
        """
        dist_mat: 2d distance-matrix
        var: defines the amount of noise to be added. Higher value means noisier data.
            variences = [1, .1, .01, .001, .0001]
            .1 for much noisier than .0001
        """
        noisy_dist_mat = torch.tensor(random_noise(dist_mat, mode='gaussian', mean=mean, var=var, clip=True))
        return noisy_dist_mat


pd = ProteinDataset(file=CONFIGS.VAL_FILE)
# print(pd.__len__())
# print(len(pd.__getitem__(0)))
# # accessing a fixed size contact-map/distance-matrix and 3d-coordinate matrix
# print(pd.__getitem__(0)[0].shape, pd.__getitem__(0)[1].shape)

# adding gaussian noise with distance matrix
dist_mat = pd.__getitem__(0)[0]
# variences = [1, .1, .01, .001, .0001]
# titles = ["var:"+str(var) for var in variences]
# titles.insert(0, "ground truth")
# noisy_dist_matrices = [pd.add_gaussian_noise(dist_mat, var=var) for var in variences]
# noisy_dist_matrices.insert(0, dist_mat) # adding lground-truth contact-map at the end
# DataViz.plot_images(noisy_dist_matrices, img_name="matrix", titles=titles, cols=3) 

# adding salt&pepper noise with distance matrix

# s_and_p = torch.tensor(random_noise(dist_mat, mode='s&p', salt_vs_pepper=0.5, clip=True))