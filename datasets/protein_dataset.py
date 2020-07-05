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
    def __init__(self, file=CONFIGS.TRAIN_FILE, noise_type=None, noise_mode=None):
        """
        file: a file path that have an id per line, this id will used to access the data.
        noise_type: optional. Available options are:
            salt_pepper, speckle, gaussian
        noise_mode: optional. Available options are:
            little, medium, much
        """
        super(ProteinDataset, self).__init__()
        self.record_ids = DataUtils.get_ids(file)
        self.noise_type = noise_type
        self.noise_mode = noise_mode

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
        dist_mat, d3_coords = DataUtils.load_using_pickle(filename)
        noisey_dist_mat = self.add_noise(dist_mat=dist_mat, noise_type=self.noise_type, mode=self.noise_mode)
        return noisey_dist_mat, d3_coords

    def get_ground_truth(self, i):
        """
        Returns a single record [inp, out]
        inp: contact-map/distance-map matrix
        out: 3d coordinate matrix.
        """
        filename = CONFIGS.CONTACT_MAP_VS_COORDINATES_DIR + self.record_ids[i] + CONFIGS.DOT_PKL
        return DataUtils.load_using_pickle(filename)

    def add_noise(self, dist_mat, noise_type, mode):
        """
        noise_type: salt_pepper, speckle, gaussian
        noise_mode: little, medium, much
        """
        if noise_type=="salt_pepper" and mode=="little":
            return self.add_saltpepper_noise(dist_mat, amount=0.05)
        elif noise_type=="salt_pepper" and mode=="medium":
            return self.add_saltpepper_noise(dist_mat, amount=0.1)
        elif noise_type=="salt_pepper" and mode=="much":
            return self.add_saltpepper_noise(dist_mat, amount=0.3)
        elif noise_type=="speckle" and mode=="little":
            return self.add_speckle_noise(dist_mat, var=0.01)
        elif noise_type=="speckle" and mode=="medium":
            return self.add_speckle_noise(dist_mat, var=0.1)
        elif noise_type=="speckle" and mode=="much":
            return self.add_speckle_noise(dist_mat, var=1)
        elif noise_type=="gaussian" and mode=="little":
            return self.add_gaussian_noise(dist_mat, var=0.001)
        elif noise_type=="gaussian" and mode=="medium":
            return self.add_gaussian_noise(dist_mat, var=0.01)
        elif noise_type=="gaussian" and mode=="much":
            return self.add_gaussian_noise(dist_mat, var=0.1)
        else:
            return dist_mat

    def add_gaussian_noise(self, dist_mat, mean=0, var=0.1):
        """
        dist_mat: n-d distance-matrix
        var: the var value defines the amount of noise to be added. 
            Higher value means noisier data.
            variences = [1, .1, .01, .001, .0001]
            example: https://github.com/akabiraka/FDFAPBG/blob/master/outputs/images/noisy_distance_matrix_using_gaussian.png
        
        """
        noisy_dist_mat = torch.tensor(random_noise(dist_mat, mode='gaussian', mean=mean, var=var, clip=True), dtype=torch.float32)
        return noisy_dist_mat

    def add_saltpepper_noise(self, dist_mat, amount=0.3):
        """
        dist_mat: n dimensional matrix
        amount: the amount value defines how much salt&pepper noise will be added.
            Greater amount value means noisier data.
            n_amount = [.5, .3, .1, .05, .005]
            example: https://github.com/akabiraka/FDFAPBG/blob/master/outputs/images/noise_distance_matrix_using_saltpepper.png
        """
        noisy_dist_mat = torch.tensor(random_noise(dist_mat, mode='s&p', salt_vs_pepper=0.5, amount=amount, clip=True), dtype=torch.float32)
        return noisy_dist_mat

    def add_speckle_noise(self, dist_mat, mean=0, var=0.1):
        """
        dist_mat: n-d distance-matrix
        var: the var value defines the amount of noise to be added. 
            Higher value means noisier data.
            variences = [1, .1, .01, .001, .0001]
            example: https://github.com/akabiraka/FDFAPBG/blob/master/outputs/images/noise_distance_matrix_using_speckle.png
        """
        noisy_dist_mat = torch.tensor(random_noise(dist_mat, mode='speckle', mean=mean, var=var, clip=True), dtype=torch.float32)
        return noisy_dist_mat

    def add_poisson_noise(self, dist_mat):
        """
        dist_mat: n-d distance-matrix
        example: https://github.com/akabiraka/FDFAPBG/blob/master/outputs/images/noisy_distance_matrix_using_poisson.png
        """
        return torch.tensor(random_noise(dist_mat, mode="poisson"))


# pd = ProteinDataset("data/record_ids.txt")#(file=CONFIGS.VAL_FILE)
# print(pd.__len__())
# print(len(pd.__getitem__(0)))
# # accessing a fixed size contact-map/distance-matrix and 3d-coordinate matrix
# print(pd.__getitem__(0)[0].shape, pd.__getitem__(0)[1].shape)

# # adding little salt_pepper noise
# # pd = ProteinDataset(file=CONFIGS.VAL_FILE, noise_type="salt_pepper", noise_mode='little')
# ground_truth = pd.get_ground_truth(0)[0]
# noisy_dist_mat = pd.__getitem__(0)[0]
# DataViz.plot_images([ground_truth, noisy_dist_mat], img_name="matrix", cols=2) 

# adding gaussian noise with distance matrix
# pd = ProteinDataset(file=CONFIGS.VAL_FILE)
# dist_mat = pd.get_ground_truth(0)[0]
# variences = [1, .1, .01, .001, .0001]
# titles = ["var:"+str(var) for var in variences]
# titles.insert(0, "ground truth")
# noisy_dist_matrices = [pd.add_gaussian_noise(dist_mat, var=var) for var in variences]
# noisy_dist_matrices.insert(0, dist_mat) # adding lground-truth contact-map at the end
# DataViz.plot_images(noisy_dist_matrices, img_name="matrix", titles=titles, cols=3) 

# adding salt&pepper noise with distance matrix
# pd = ProteinDataset(file=CONFIGS.VAL_FILE)
# dist_mat = pd.get_ground_truth(0)[0]
# n_amount = [.5, .3, .1, .05, .005]
# titles = ["amount:"+str(amount) for amount in n_amount]
# titles.insert(0, "ground truth")
# noisy_dist_matrices = [pd.add_saltpepper_noise(dist_mat, amount=amount) for amount in n_amount]
# noisy_dist_matrices.insert(0, dist_mat) # adding lground-truth contact-map at the end
# DataViz.plot_images(noisy_dist_matrices, img_name="matrix", titles=titles, cols=3) 

# adding speckle noise with distance matrix
# pd = ProteinDataset(file=CONFIGS.VAL_FILE)
# dist_mat = pd.get_ground_truth(0)[0]
# variences = [1, .1, .01, .001, .0001]
# titles = ["var:"+str(var) for var in variences]
# titles.insert(0, "ground truth")
# noisy_dist_matrices = [pd.add_speckle_noise(dist_mat, var=var) for var in variences]
# noisy_dist_matrices.insert(0, dist_mat) # adding lground-truth contact-map at the end
# DataViz.plot_images(noisy_dist_matrices, img_name="matrix", titles=titles, cols=3) 

# adding poisson noise with distance matrix
# pd = ProteinDataset(file=CONFIGS.VAL_FILE)
# dist_mat = pd.get_ground_truth(0)[0]
# titles = ["ground truth", "poisson"]
# noisy_dist_matrices = [dist_mat, pd.add_poisson_noise(dist_mat)]
# DataViz.plot_images(noisy_dist_matrices, img_name="matrix", titles=titles, cols=2) 

