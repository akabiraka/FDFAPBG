import sys
sys.path.append("../FDFAPBG")
import numpy as np

import configs.general_config as CONFIGS
from datasets.a_pdb_data import APDBData

class MoleculeCoordinates(APDBData):
    def __init__(self):
        super(MoleculeCoordinates, self).__init__()

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
        aa_residues = self.get_a_chain(pdb_id, chain_id)
        d3_coords = self.get_3d_coords(aa_residues, atoms)
        return d3_coords

coords = MoleculeCoordinates()
d3_coords = coords.get("5sy8", "O", ["CA", "CB"])
print(d3_coords.shape)
d3_coords = coords.get("5sy8", "O", CONFIGS.BACKBONE_ATOMS)
print(d3_coords.shape)