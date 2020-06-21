import sys
sys.path.append("../FDFAPBG")
import numpy as np

from Bio.PDB import *

import configs.general_config as CONFIGS
import vizualizations.data_viz as DataViz
import utils.data_utils as DataUtils

class ContactMap(object):
    def __init__(self, mat_type="norm_dist", map_type='NN', atom_1="CB", atom_2="CB", save_map=True, th=8):
        """
        mat_type: output matrix type
            values: c_map, dist, norm_dist
            default: norm_dist 
            description: 
                c_map (contact map using given threshhold),
                dist (distance matrix),
                norm_dist (normalized distance matrix between 0 and 1)
        map_type: NN, 4NN, 4N4N
            NN: CA-CA or CB-CB or N-N or O-O or CA-CB and so on
            4NN: CA-CA, CB-CB, N-N, O-O
            4N4N: CA-CA, CA-CB, CA-N, CA-O and other combinations
        atom_i: CA, CB, N, O
        th: threshhold to compute short, medium or long range contact
        """
        super(ContactMap, self).__init__()
        self.parser = MMCIFParser(QUIET=True)
        self.len_bb_atoms = len(CONFIGS.BACKBONE_ATOMS)
        self.mat_type = mat_type
        self.map_type = map_type
        self.atom_1 = atom_1
        self.atom_2 = atom_2
        self.save_map = save_map
        self.th = th
        
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

    def compute_atom_atom_distance(self, residue_1, residue_2, atom_1="CB", atom_2="CB"):
        """
        Compute distance between atom-atom coordinates of two residues'.
        An atom could be CA, CB, N, O.
        Default atoms are beta-beta carbon.
        """
        if atom_1=="CB" and residue_1.get_resname()=='GLY':
            atom_1 = "CA"
        
        if atom_2=="CB" and residue_2.get_resname()=='GLY':
            atom_2 = "CA"

        diff_vector = residue_1[atom_1].coord - residue_2[atom_2].coord
        
        return np.sqrt(np.sum(diff_vector * diff_vector))

    def compute_nn_distance_matrix(self, chain_1, chain_2, atom_1="CB", atom_2="CB"):
        """
        Compute nxn distance matrix of two chains where n is residue length. Default atoms are beta-beta carbon.
        """
        dist_matrix = np.zeros((len(chain_1), len(chain_2)), np.float)
        for row, residue_1 in enumerate(chain_1):
            for col, residue_2 in enumerate(chain_2):
                dist_matrix[row, col] = self.compute_atom_atom_distance(residue_1, residue_2, atom_1, atom_2)
        return dist_matrix 

    def compute_4nn_distance_matrix(self, chain_1, chain_2):
        """
        Compute 4xnxn distance matrix for CA, CB, N and O. No distance is computed as cross atom
        distace, i.e CA-N or N-O and so on.
        """
        dist_matrix_1 = np.zeros((len(chain_1), len(chain_2)), np.float)
        dist_matrix_2 = np.zeros((len(chain_1), len(chain_2)), np.float)
        dist_matrix_3 = np.zeros((len(chain_1), len(chain_2)), np.float)
        dist_matrix_4 = np.zeros((len(chain_1), len(chain_2)), np.float)

        for row, residue_1 in enumerate(chain_1):
            for col, residue_2 in enumerate(chain_2):
                dist_matrix_1[row, col] = self.compute_atom_atom_distance(residue_1, residue_2, "CA", "CA")
                dist_matrix_2[row, col] = self.compute_atom_atom_distance(residue_1, residue_2, "CB", "CB")
                dist_matrix_3[row, col] = self.compute_atom_atom_distance(residue_1, residue_2, "N", "N")
                dist_matrix_4[row, col] = self.compute_atom_atom_distance(residue_1, residue_2, "O", "O")

        result = np.stack((dist_matrix_1, dist_matrix_2, dist_matrix_3, dist_matrix_4), axis=0)
        # print(result.shape)
        return result

    def compute_4n4n_distance_matrix(self, chain_1, chain_2):
        """
        All pairwise backbone atom distance. Is is also called full-atom distance matrix.
        4 backbone atoms CA, CB, N and O. If ther are n residues in a chain,
        the distance matrix is of size (4n x 4n)
        """
        dist_matrix = np.zeros((self.len_bb_atoms*len(chain_1), self.len_bb_atoms*len(chain_2)), np.float)
        for row, residue_1 in enumerate(chain_1):
            for col, residue_2 in enumerate(chain_2):
                for k, atom_1 in enumerate(CONFIGS.BACKBONE_ATOMS):
                    for l, atom_2 in enumerate(CONFIGS.BACKBONE_ATOMS):
                        dist_matrix[4*row+k, 4*col+l] = self.compute_atom_atom_distance(residue_1, residue_2, atom_1, atom_2)
        return dist_matrix  

    def get(self, pdb_id, chain_id):
        print("computing contact-map for {}:{} ... ...".format(pdb_id, chain_id))
        pdb_filename = CONFIGS.PDB_DIR + pdb_id + CONFIGS.DOT_CIF
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
                dist_matrix = 0.0
                if self.map_type == "4N4N":
                    dist_matrix = self.compute_4n4n_distance_matrix(aa_residues, aa_residues)
                elif self.map_type == "4NN":
                    dist_matrix = self.compute_4nn_distance_matrix(aa_residues, aa_residues)
                else: 
                    dist_matrix = self.compute_nn_distance_matrix(aa_residues, aa_residues, self.atom_1, self.atom_2)

                # dist_matrix = np.zeros((n_aa_residues, n_aa_residues), np.float)
                # try:
                #     # computing distance matrix
                #     dist_matrix = self.compute_full_atom_distance_matrix(aa_residues, aa_residues)
                # except Exception as e:
                #     is_defected = True
                break

        if self.mat_type == "c_map":
            dist_matrix = np.where(dist_matrix < self.th, 1, 0)
        elif self.mat_type == "norm_dist" and self.map_type!="4NN":
            dist_matrix = DataUtils.scale(dist_matrix, 0, 1)
        elif self.mat_type == "norm_dist" and self.map_type=="4NN":
            for i in range(0, 4):
                dist_matrix[i] = DataUtils.scale(dist_matrix[i], 0, 1)
        else:
            # dist_matrix remains same
            pass

        if self.save_map:
            DataUtils.save_contact_map(dist_matrix, pdb_id+chain_id)
        return dist_matrix


# c_map = ContactMap(mat_type="c_map", map_type='NN', atom_1="CA", atom_2="CA")
# c_map = ContactMap(mat_type="c_map", map_type='4NN')
c_map = ContactMap(mat_type="norm_dist", map_type='4N4N')
dist_matrix = c_map.get("5sy8", "O")
print(dist_matrix.shape)
if c_map.map_type == "4NN":
    DataViz.plot_images([dist_matrix[0], dist_matrix[1], dist_matrix[2], dist_matrix[3]], "5sy8"+"O", cols=2)
else: 
    DataViz.plot_images([dist_matrix], "5sy8"+"O", cols=1)