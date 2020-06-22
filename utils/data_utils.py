import sys
sys.path.append('../FDFAPBG')
import numpy as np
import torch
import pickle

from configs.general_config import *

def save_tensor(tensor, file):
    torch.save(tensor, file)

def save_contact_map(np_array, pdb_code):
    file = CONTACT_MAP_DIR + pdb_code
    save_tensor(torch.tensor(np_array), file + DOT_PT)

def save_molecule_coordinates(np_array, id):
    file = MOLECULE_COORDINATES_DIR + id
    save_tensor(torch.tensor(np_array), file + DOT_PT)

def read_tensor(path, filename):
    return torch.load(path + filename)

def read_contact_map_tensor(pdb_code):
    filename = pdb_code + DOT_PT
    return read_tensor(CONTACT_MAP_DIR, filename)

def read_3d_coords_tensor(pdb_code):
    filename = pdb_code + DOT_PT
    return read_tensor(MOLECULE_COORDINATES_DIR, filename)

def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 

def save_itemlist(itemlist, file):
    with open(file, 'w') as f:
        for item in itemlist:
            f.write("%s\n" % item)

def save_using_pickle(data, filename):
    with open(filename, 'wb') as filehandle:
        pickle.dump(data, filehandle)

def load_using_pickle(filename):
    with open(filename, 'rb') as filehandle:
        return pickle.load(filehandle)