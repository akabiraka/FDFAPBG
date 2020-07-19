import sys
sys.path.append("../FDFAPBG")

import numpy as np
import torch
from Bio.PDB import *

import configs.general_config as CONFIGS
import utils.data_utils as DataUtils

class Coordinates(object):
    def __init__(self, input_dir=CONFIGS.FRAGMENTS_DIR, input_file_format=CONFIGS.DOT_PDB,\
         output_dir=CONFIGS.CONTACT_MAP_DIR, parser=PDBParser(QUIET=True), save=True, atoms=CONFIGS.BACKBONE_ATOMS):
        
        super(Coordinates, self).__init__()
        self.input_dir = input_dir
        self.input_file_format = input_file_format
        self.output_dir = output_dir
        self.parser = parser
        self.save = save
        self.atoms = atoms

    def get_3d_coords(self, chain, atoms=["CB"]):
        """
        Prepare 3d coordinates in [kn x 3] matrix, where k is the number of atoms given.
        """
        d3_coords_matrix = []
        for i, residue in enumerate(chain):
            for j, atom in enumerate(atoms):
                if atom=="CB" and residue.get_resname()=='GLY':
                    atom = "CA"
                d3_coords_matrix.append(residue[atom].coord)
        return np.array(d3_coords_matrix)
    
    def generate(self, filename, pdb_id, chain_id):
        pdb_filename = self.input_dir + filename + self.input_file_format
        structure = self.parser.get_structure(pdb_id, pdb_filename)
        residues = structure[0][chain_id].get_residues()
        list_residues = list(residues)
        
        d3_coords = self.get_3d_coords(list_residues, self.atoms)
        
        if self.save:
            DataUtils.save_using_pickle(d3_coords, self.output_dir + filename + CONFIGS.DOT_PKL)
            
        print("Prepared coordinates for {}:{} ... ...".format(filename, d3_coords.shape))
        return d3_coords
        
        
