import sys
sys.path.append("../FDFAPBG")
import numpy as np

from Bio.PDB import *

import configs.general_config as CONFIGS
import vizualizations.data_viz as DataViz

class ContactMap(object):
    def __init__(self):
        super(ContactMap, self).__init__()
        self.parser = MMCIFParser(QUIET=True)
        self.len_bb_atoms = len(CONFIGS.BACKBONE_ATOMS)

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

    def compute_distance_matrix(self, chain_1, chain_2):
        """
        compute distance matrix of two chains
        """
        dist_matrix = np.zeros((len(chain_1), len(chain_2)), np.float)
        for row, residue_1 in enumerate(chain_1):
            for col, residue_2 in enumerate(chain_2):
                dist_matrix[row, col] = self.get_beta_beta_carbon_distance(residue_1, residue_2)
        return dist_matrix   

    def get_atom_atom_distance(self, residue_1, residue_2, atom_1, atom_2):
        """
        Compute distance between atom-atom coordinates of two residues'.
        An atom could be CA, CB, N, O.
        """
        if atom_1=="CB" and residue_1.get_resname()=='GLY':
            atom_1 = "CA"
        
        if atom_2=="CB" and residue_2.get_resname()=='GLY':
            atom_2 = "CA"

        diff_vector = residue_1[atom_1].coord - residue_2[atom_2].coord
        return np.sqrt(np.sum(diff_vector * diff_vector))

    def get_O_O_distance(self, residue_1, residue_2):
        """
        Compute distance between oxigen-oxigen coordinates of two residues'.
        """
        return self.get_atom_atom_distance(residue_1, residue_2, "O", "O")

    def get_N_N_distance(self, residue_1, residue_2):
        """
        Compute distance between nitrogen-nitrogen coordinates of two residues'.
        """
        return self.get_atom_atom_distance(residue_1, residue_2, "N", "N")

    def get_alpha_alpha_carbon_distance(self, residue_1, residue_2):
        """
        Compute distance between alpha-carbon coordinates of two residues'.
        """
        return self.get_atom_atom_distance(residue_1, residue_2, "CA", "CA")

    def get_beta_beta_carbon_distance(self, residue_1, residue_2):
        """
        Compute distance between beta-carbon coordinates of two residues',
        except for GLYcine.
        """
        return self.get_atom_atom_distance(residue_1, residue_2, "CB", "CB")

    def compute_full_atom_distance_matrix(self, chain_1, chain_2):
        """
        4 backbone atoms CA, CB, N and O. If ther are n residues in a chain,
        the distance matrix is of size (4n x 4n)
        """
        dist_matrix = np.zeros((self.len_bb_atoms*len(chain_1), self.len_bb_atoms*len(chain_2)), np.float)
        for row, residue_1 in enumerate(chain_1):
            for col, residue_2 in enumerate(chain_2):
                for k, atom_1 in enumerate(CONFIGS.BACKBONE_ATOMS):
                    for l, atom_2 in enumerate(CONFIGS.BACKBONE_ATOMS):
                        dist_matrix[4*row+k, 4*col+l] = self.get_atom_atom_distance(residue_1, residue_2, atom_1, atom_2)
        return dist_matrix  

    def get_full(self, pdb_id, chain_id):
        print("computing contact-map for {}:{} ... ...".format(pdb_id, chain_id))
        pdb_filename = CONFIGS.PDB_DIR + pdb_id + CONFIGS.DOT_CIF
        is_defected = False
        # reading whole structure
        structure = self.parser.get_structure(pdb_id, pdb_filename)
        models = list(structure.get_models())
        chains = list(models[0].get_chains())
        # for each chain
        for chain in chains:
            if chain.id == chain_id:
                all_residues = list(chain.get_residues())
                aa_residues, seq, _ = self.filter_aa_residues(all_residues)
                n_aa_residues = len(aa_residues)
                # print(_)
                dist_matrix = self.compute_full_atom_distance_matrix(aa_residues, aa_residues)
                # dist_matrix = np.zeros((n_aa_residues, n_aa_residues), np.float)
                # try:
                #     # computing distance matrix
                #     dist_matrix = self.compute_full_atom_distance_matrix(aa_residues, aa_residues)
                # except Exception as e:
                #     is_defected = True
                break
        DataViz.plot_images([dist_matrix], pdb_id+chain_id, cols=1)
        # print(dist_matrix)
        return is_defected, dist_matrix


c_map = ContactMap()
c_map.get_full("5sy8", "O")