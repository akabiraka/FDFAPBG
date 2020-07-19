import sys
sys.path.append('../FDFAPBG')
import traceback
from termcolor import colored
from Bio.PDB import *

import configs.general_config as CONFIGS
from utils.clean_slate import CleanSlate
from datasets.my_pdb_data import MyPDBData
from datasets.protein_sampler import ProteinSampler
from datasets.distance_matrix import DistanceMatrix
from datasets.coordinates import Coordinates
import utils.data_utils as DataUtils

def get_pdb_id(line):
        """
        Given a line where first item is a pdb_id with chain_id like '1A62A',
        this method returns the pdb_id and chain_id. If ther are multiple chain_ids 
        like '1A62ACD", it only returns first chain id like 'A'.
        """
        line = line.split()
        pdb_id = line[0][:4].lower()
        chain_id = line[0][4:]
        return pdb_id, chain_id
    

myPDBData = MyPDBData()
cln = CleanSlate()
proteinSampler = ProteinSampler()
distanceMatrix = DistanceMatrix(output_matrix_type="4N4N", input_dir=CONFIGS.FRAGMENTS_DIR, input_file_format=CONFIGS.DOT_PDB, output_dir=CONFIGS.CONTACT_MAP_DIR, parser=PDBParser(QUIET=True), save=False)
coordinates = Coordinates(input_dir=CONFIGS.FRAGMENTS_DIR, input_file_format=CONFIGS.DOT_PDB, output_dir=CONFIGS.MOLECULE_COORDINATES_DIR, parser=PDBParser(QUIET=True), save=False, atoms=CONFIGS.BACKBONE_ATOMS)

bd_fragment_id_handle = open("data/val_batch_bad_fragment_ids.txt", "a")
good_fragment_id_handle = open("data/val_batch_good_fragment_ids.txt", "a")
# file_content = open(CONFIGS.TRAIN_FILE, "r")
# file_content = open(CONFIGS.TEST_FILE, "r")
file_content = open(CONFIGS.VAL_FILE, "r")

n_proteins_to_skip = 0
total_proteins_to_evaluate = n_proteins_to_skip + 5000

for i, line in enumerate(file_content):
    if i < n_proteins_to_skip: continue
    print("Processing {}th protein ... ...:".format(i+1))
    
    pdb_id, chain_id = get_pdb_id(line)
    if len(chain_id) > 1: continue
    
    myPDBData.download(pdb_id)
    cleaned_structure = myPDBData.clean(pdb_id, chain_id)
    chain = myPDBData.get_chain_from_structure(cleaned_structure, chain_id)
    fragment_ids = proteinSampler.sample_from_chain(pdb_id, chain_id, chain)
    
    for fragment_id in fragment_ids:
        try:
            dist_matrix = distanceMatrix.generate(filename=fragment_id, pdb_id=pdb_id, chain_id=chain_id)
            d3_coords = coordinates.generate(filename=fragment_id, pdb_id=pdb_id, chain_id=chain_id)
            DataUtils.save_using_pickle([dist_matrix, d3_coords], CONFIGS.CONTACT_MAP_VS_COORDINATES_DIR+fragment_id+CONFIGS.DOT_PKL)
            good_fragment_id_handle.write(fragment_id+"\n")
            print(colored("Comment: good {}".format(fragment_id), "green"))
        except Exception as e:
            traceback.print_exc()
            bd_fragment_id_handle.write(fragment_id+"\n")
            print(colored("Comment: corrupted {}".format(fragment_id), "red"))
            continue
    
    if i+1 == total_proteins_to_evaluate: break
        
    cln.clean_all_files(mydir=CONFIGS.CLEAN_PDB_DIR, ext=CONFIGS.DOT_PDB)    
    cln.clean_all_files(mydir=CONFIGS.CONTACT_MAP_DIR, ext=CONFIGS.DOT_PKL)
    cln.clean_all_files(mydir=CONFIGS.MOLECULE_COORDINATES_DIR, ext=CONFIGS.DOT_PKL)
    cln.clean_all_files(mydir=CONFIGS.PDB_DIR, ext=CONFIGS.DOT_CIF)
    print()