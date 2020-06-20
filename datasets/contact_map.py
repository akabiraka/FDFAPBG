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
                dist_matrix[row, col] = self.get_alpha_alpha_carbon_distance(residue_1, residue_2)
        return dist_matrix    

    def get_alpha_alpha_carbon_distance(self, residue_1, residue_2):
        """
        Compute distance between alpha-carbon coordinates of two residues'.
        """
        diff_vector = residue_1["CA"].coord - residue_2["CA"].coord
        return np.sqrt(np.sum(diff_vector * diff_vector))

    def get_beta_beta_carbon_distance(self, residue_1, residue_2):
        """
        Compute distance between beta-carbon coordinates of two residues',
        except for GLYcine.
        """
        GLY = 'GLY'
        res_1_name = residue_1.get_resname()
        res_2_name = residue_2.get_resname()
        diff_vector = 0.0
        try:
            if res_1_name == GLY and res_2_name != GLY:
                diff_vector = residue_1["CA"].coord - residue_2["CB"].coord
            elif res_1_name != GLY and res_2_name == GLY:
                diff_vector = residue_1["CB"].coord - residue_2["CA"].coord
            elif res_1_name == GLY and res_2_name == GLY:
                diff_vector = residue_1["CA"].coord - residue_2["CA"].coord
            else:
                diff_vector = residue_1["CB"].coord - residue_2["CB"].coord
        except Exception as e:
            print("Can not resolve distance: ", res_1_name, res_2_name)
            # traceback.print_exc()
            raise
        return np.sqrt(np.sum(diff_vector * diff_vector))

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
                dist_matrix = np.zeros((n_aa_residues, n_aa_residues), np.float)
                try:
                    # computing distance matrix
                    dist_matrix = self.compute_distance_matrix(aa_residues, aa_residues)
                except Exception as e:
                    is_defected = True
                    break
        DataViz.plot_images([dist_matrix], pdb_id+chain_id, cols=1)
        # print(dist_matrix)
        return is_defected, dist_matrix


c_map = ContactMap()
c_map.get_full("5sy8", "O")