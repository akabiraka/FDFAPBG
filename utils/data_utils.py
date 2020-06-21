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

def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 

def filter_aa_residues(self, chain):
        """
        A chain can be heteroatoms(water, ions, etc; anything that 
        isn't an amino acid or nucleic acid)
        so this function get rid of atoms excepts amino-acids
        """
        aa_residues = []
        non_aa_residues = []
        non_aa = []
        seq = ""
        for i in chain:
            if i.get_resname() in standard_aa_names:
                aa_residues.append(i)
                seq += CONFIGS.AMINO_ACID_3TO1[i.get_resname()]
            else:
                non_aa.append(i.get_resname())
                non_aa_residues.append(i.get_resname())
        return aa_residues, seq, non_aa_residues
