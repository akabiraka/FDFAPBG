import sys
sys.path.append('../FDFAPBG')

from Bio.PDB import *

import configs.general_config as CONFIGS
import utils.data_utils as DataUtils
from datasets.contact_map import ContactMap
from datasets.molecule_coordinates import MoleculeCoordinates

class DataGenerator(object):
    def __init__(self):
        super(DataGenerator, self).__init__()
        print("alhumdulillah")

        self.pdbl = PDBList()
        
    def get_pdb_id(self, line):
        """
        Given a line where first item is a pdb_id with chain_id like '1A62A',
        this method returns the pdb_id and chain_id. If ther are multiple chain_ids 
        like '1A62ACD", it only returns first chain id like 'A'.
        """
        line = line.split()
        pdb_id = line[0][:4].lower()
        chain_id = line[0][4]
        return pdb_id, chain_id

    def download(self, pdb_code):
        """
        Download protein data in .cif format in CONFIGS.PDB_DIR.
        """
        self.pdbl.retrieve_pdb_file(pdb_code, pdir=CONFIGS.PDB_DIR, file_format=CONFIGS.CIF)


generator = DataGenerator()
c_map = ContactMap(map_type='4N4N')
coords = MoleculeCoordinates()
file_content = open(CONFIGS.ALL_PDB_IDS, "r")
for i, line in enumerate(file_content):
    print("{}th protein:".format(i+1))
    pdb_id, chain_id = generator.get_pdb_id(line)
    pdb_with_chain = pdb_id + chain_id
    # print(pdb_id, chain_id)
    generator.download(pdb_id)
    dist_matrix = c_map.get(pdb_id, chain_id)
    d3_coords = coords.get(pdb_id, chain_id)
    print(dist_matrix.shape, d3_coords.shape)
    print()