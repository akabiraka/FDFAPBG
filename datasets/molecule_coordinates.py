import sys
sys.path.append("../FDFAPBG")
import numpy as np

from Bio.PDB import *

import configs.general_config as CONFIGS

class MoleculeCoordinates(object):
    def __init__(self):
        super(MoleculeCoordinates, self).__init__()
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

    def get_4nn_3d_coords(self, chain):
        """
        Deprecated. Output shape is [4xnx3] where n is chain length.
        4 is for four backbone atoms. 3 is for 3D-coordinates.
        """
        d3_coords_matrix = [[],[], [],[]]
        for i, residue in enumerate(chain):
            d3_coords_matrix[0].append(residue['CA'].coord)
            if residue.get_resname()=='GLY':
                d3_coords_matrix[1].append(residue['CA'].coord)
            else:
                d3_coords_matrix[1].append(residue['CB'].coord)
            d3_coords_matrix[2].append(residue['N'].coord)
            d3_coords_matrix[3].append(residue['O'].coord)
        return np.array(d3_coords_matrix)

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
    
    def get(self, pdb_id, chain_id, atoms=["CB"]):
        print("preparing coordinates of molecules for {}:{} ... ...".format(pdb_id, chain_id))
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
                
                d3_coords = self.get_3d_coords(aa_residues, atoms)

        return d3_coords

coords = MoleculeCoordinates()
d3_coords = coords.get("5sy8", "O", ["CA", "CB"])
print(d3_coords.shape)
d3_coords = coords.get("5sy8", "O", CONFIGS.BACKBONE_ATOMS)
print(d3_coords.shape)