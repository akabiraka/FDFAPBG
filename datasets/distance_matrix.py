import sys
sys.path.append("../FDFAPBG")
import numpy as np
from Bio.PDB import *
import torch

import configs.general_config as CONFIGS
import utils.data_utils as DataUtils

class DistanceMatrix(object):
    def __init__(self, output_matrix_type="4N4N", input_dir=CONFIGS.FRAGMENTS_DIR, input_file_format=CONFIGS.DOT_PDB,\
         output_dir=CONFIGS.CONTACT_MAP_DIR, parser=PDBParser(QUIET=True), atom_1="CB", atom_2="CB", save=False):
        """Compute distance matrix.

        Args:
            output_matrix_type (str, optional): [description]. Defaults to "4N4N".
            input_dir ([type], optional): [description]. Defaults to CONFIGS.FRAGMENTS_DIR.
            file_format ([type], optional): [description]. Defaults to CONFIGS.DOT_PDB.
            parser ([type], optional): [description]. Defaults to PDBParser(QUIET=True).
            atom_1 (str, optional): [description]. Defaults to "CB".
            atom_2 (str, optional): [description]. Defaults to "CB".
            save (bool, optional): [description]. Defaults to False.
        """
        super(DistanceMatrix, self).__init__()
        self.output_matrix_type = output_matrix_type
        self.input_dir = input_dir
        self.input_file_format = input_file_format
        self.output_dir = output_dir
        self.parser = parser
        self.atom_1 = atom_1
        self.atom_2 = atom_2
        self.save = save
        self.bb_atoms = CONFIGS.BACKBONE_ATOMS

    def compute_atom_atom_distance(self, residue_1, residue_2, atom_1="CB", atom_2="CB"):
        """
        Compute distance between atom-atom coordinates of two residues'.
        An atom could be CA, CB, N, O.
        Default atoms are beta-beta carbon.
        """
        try:
            if atom_1=="CB" and residue_1.get_resname()=='GLY':
                atom_1 = "CA"
            
            if atom_2=="CB" and residue_2.get_resname()=='GLY':
                atom_2 = "CA"

            diff_vector = residue_1[atom_1].coord - residue_2[atom_2].coord
        except Exception as e:
            print("Can not resolve distance: ", residue_1.get_resname(), residue_2.get_resname(), atom_1, atom_2)
            traceback.print_exc()
            raise
            # in case, there is an error but I want the distance matrix, comment out above lines and comment in next line
            # return 0.0 

        return np.sqrt(np.sum(diff_vector * diff_vector))
                
    def compute_4n4n_distance_matrix(self, chain_1, chain_2):
        """
        All pairwise backbone atom distance. Is is also called full-atom distance matrix.
        4 backbone atoms CA, CB, N and O. If ther are n residues in a chain,
        the distance matrix is of size (4n x 4n)
        """
        l = len(self.bb_atoms)
        dist_matrix = np.zeros((l*len(chain_1), l*len(chain_2)), np.float)
        for row, residue_1 in enumerate(chain_1):
            for col, residue_2 in enumerate(chain_2):
                for k, atom_1 in enumerate(self.bb_atoms):
                    for l, atom_2 in enumerate(self.bb_atoms):
                        dist_matrix[4*row+k, 4*col+l] = self.compute_atom_atom_distance(residue_1, residue_2, atom_1, atom_2)
        return dist_matrix  
    
    def compute_nn_distance_matrix(self, chain_1, chain_2, atom_1="CB", atom_2="CB"):
        """
        Compute nxn distance matrix of two chains where n is residue length. Default atoms are beta-beta carbon.
        """
        dist_matrix = np.zeros((len(chain_1), len(chain_2)), np.float)
        for row, residue_1 in enumerate(chain_1):
            for col, residue_2 in enumerate(chain_2):
                dist_matrix[row, col] = self.compute_atom_atom_distance(residue_1, residue_2, atom_1, atom_2)
        return dist_matrix 

    def generate(self, filename, pdb_id, chain_id):
        
        pdb_filename = self.input_dir + filename + self.input_file_format
        structure = self.parser.get_structure(pdb_id, pdb_filename)
        residues = structure[0][chain_id].get_residues()
        list_residues = list(residues)
        
        dist_matrix = None
        if self.output_matrix_type=="4N4N":
            dist_matrix = self.compute_4n4n_distance_matrix(list_residues, list_residues)
        elif self.output_matrix_type=="NN":
            dist_matrix = self.compute_nn_distance_matrix(list_residues, list_residues, self.atom_1, self.atom_2)
        
        if self.save:
            DataUtils.save_using_pickle(dist_matrix, self.output_dir + filename + CONFIGS.DOT_PKL)
            
        print("Computed distance-matrix for {}:{}".format(filename, dist_matrix.shape))
        return dist_matrix
    