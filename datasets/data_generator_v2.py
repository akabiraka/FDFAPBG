import sys
sys.path.append('../FDFAPBG')
import time
import traceback

from Bio.PDB import *

import configs.general_config as CONFIGS
from utils.clean_slate import CleanSlate
from datasets.my_pdb_data import MyPDBData
from datasets.protein_sampler import ProteinSampler

from datasets.contact_map import ContactMap
from datasets.molecule_coordinates import MoleculeCoordinates
from datasets.input_output_generator import InputOutputGenerator

def get_pdb_id(line):
        """
        Given a line where first item is a pdb_id with chain_id like '1A62A',
        this method returns the pdb_id and chain_id. If ther are multiple chain_ids 
        like '1A62ACD", it only returns first chain id like 'A'.
        """
        line = line.split()
        pdb_id = line[0][:4].lower()
        chain_id = line[0][4]
        return pdb_id, chain_id
    
myPDBData = MyPDBData()
cln = CleanSlate()
proteinSampler = ProteinSampler()
file_content = open(CONFIGS.ALL_PDB_IDS, "r")
for i, line in enumerate(file_content):
    print("Processing {}th protein ... ...:".format(i+1))
    pdb_id, chain_id = get_pdb_id(line)
    
    myPDBData.download(pdb_id)
    cleaned_structure = myPDBData.clean(pdb_id, chain_id)
    chain = myPDBData.get_chain_from_structure(cleaned_structure, chain_id)
    
    fragment_ids = proteinSampler.sample_from_chain(pdb_id, chain_id, chain)
    
    
    cln.clean_all_files(mydir=CONFIGS.PDB_DIR, ext=CONFIGS.DOT_CIF)
    cln.clean_all_files(mydir=CONFIGS.CLEAN_PDB_DIR, ext=CONFIGS.DOT_PDB)
    print()