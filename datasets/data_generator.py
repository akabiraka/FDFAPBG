import sys
sys.path.append('../FDFAPBG')

from Bio.PDB import *

import configs.general_config as CONFIGS
from datasets.contact_map import ContactMap
import utils.data_utils as DataUtils

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

    def get_3D_coordinate_matrix(self, pdb_id, chain_id):
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
                aa_residues, seq, _ = DataUtils.filter_aa_residues(all_residues)
                n_aa_residues = len(aa_residues)


generator = DataGenerator()
c_map = ContactMap(mat_type="norm_dist", map_type='4N4N')
file_content = open(CONFIGS.ALL_PDB_IDS, "r")
for i, line in enumerate(file_content):
    print("{}th protein:".format(i+1))
    pdb_id, chain_id = generator.get_pdb_id(line)
    pdb_with_chain = pdb_id + chain_id
    # print(pdb_id, chain_id)
    generator.download(pdb_id)
    c_map.get(pdb_id, chain_id)
    print()