import sys
sys.path.append('../FDFAPBG')
import numpy as np
import torch

from configs.general_config import *

def save_tensor(tensor, file):
    torch.save(tensor, file)

def save_contact_map(np_array, pdb_code):
    file = CONTACT_MAP_DIR + pdb_code
    save_tensor(torch.tensor(np_array), file + DOT_PT)
