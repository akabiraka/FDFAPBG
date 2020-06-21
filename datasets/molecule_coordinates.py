import sys
sys.path.append("../FDFAPBG")
import numpy as np

import configs.general_config as CONFIGS
from datasets.a_pdb_data import APDBData
import utils.data_utils as DataUtils

class MoleculeCoordinates(APDBData):
    def __init__(self, normalized=True, save=True):
        super(MoleculeCoordinates, self).__init__()
        self.normalized = normalized
        self.save = save

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

    def get(self, pdb_id, chain_id, atoms=CONFIGS.BACKBONE_ATOMS):
        """
        Returns 3d coordinates extracted from pdb data for given atoms.
        """
        print("Preparing coordinates of molecules for {}:{} ... ...".format(pdb_id, chain_id))
        aa_residues = self.get_a_chain(pdb_id, chain_id)
        d3_coords = self.get_3d_coords(aa_residues, atoms)
        
        # post-operations: normalization
        if self.normalized:
            d3_coords = DataUtils.scale(d3_coords, 0, 1)
        
        # post-operations: saving coordinates
        if self.save:
            DataUtils.save_molecule_coordinates(d3_coords, pdb_id+chain_id)

        return d3_coords

# coords = MoleculeCoordinates(normalized=True)
# d3_coords = coords.get("5sy8", "O", ["CA", "CB"])
# print(d3_coords.shape)
# # print(d3_coords)
# d3_coords = coords.get("5sy8", "O", CONFIGS.BACKBONE_ATOMS)
# print(d3_coords.shape)