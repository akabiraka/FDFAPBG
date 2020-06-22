import sys
sys.path.append('../FDFAPBG')
import math
import numpy as np
import torch

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

    def get_1st_frag(self, pdb_code):
        """
        Compute left-top block as 1st-frag of contact-map/distance-matrix 
        and 3d coordinates.
        pdb_code = "5sy8" + "O"
        """
        print("Generating input-output sets for {} ... ...".format(pdb_code))
        a_input_output_set = []
        all_input_output_set = []
        full_c_map = DataUtils.read_contact_map_tensor(pdb_code)
        full_d3_coords = DataUtils.read_3d_coords_tensor(pdb_code)
        c_map_inp = full_c_map[0:CONFIGS.WINDOW_SIZE, 0:CONFIGS.WINDOW_SIZE]
        c_map_inp = c_map_inp.type(torch.float32)
        d3_coords_out = full_d3_coords[0:CONFIGS.WINDOW_SIZE]
        a_input_output_set = [c_map_inp, d3_coords_out]
        all_input_output_set.append(a_input_output_set)
        return all_input_output_set

    def get_inp_out_sets(self, pdb_code):
        """
        Given a pdb_code, looks like pdb_code = "5sy8" + "O", 
        this method generates fixed size contact-map and 3d coordinate
        matrix based on WINDOW_SIZE and WINDOW_STRIDE.
        pdb_code = "5sy8" + "O"
        """
        print("Generating input-output sets for {} ... ...".format(pdb_code))
        full_c_map = DataUtils.read_contact_map_tensor(pdb_code)
        full_d3_coords = DataUtils.read_3d_coords_tensor(pdb_code)
        # print(full_c_map.shape, d3_coords.shape)
        rows, cols = full_c_map.shape
        half_width = math.floor(CONFIGS.WINDOW_SIZE / 2)
        a_input_output_set = []
        all_input_output_set = []
        for i in range(half_width, rows - half_width, CONFIGS.WINDOW_STRIDE):
            s1_from_idx = i - half_width
            s1_to_idx = i + half_width
            d3_coords_out = full_d3_coords[s1_from_idx:s1_to_idx]
            for j in range(half_width, rows - half_width, CONFIGS.WINDOW_STRIDE):
                s2_from_idx = j - half_width
                s2_to_idx = j + half_width
                
                c_map_inp = full_c_map[s1_from_idx:s1_to_idx, s2_from_idx:s2_to_idx]
                c_map_inp = c_map_inp.type(torch.float32)
                # print(s1_from_idx, s1_to_idx, s2_from_idx, s2_to_idx)
                # print(c_map_inp.shape, d3_coords_out.shape)
                a_input_output_set = [c_map_inp, d3_coords_out]
                all_input_output_set.append(a_input_output_set)
        return all_input_output_set
        
generator = DataGenerator()
c_map = ContactMap(map_type='4N4N')
coords = MoleculeCoordinates()
file_content = open(CONFIGS.ALL_PDB_IDS, "r")
good_proteins = []
bad_proteins = []
records = []
for i, line in enumerate(file_content):
    print("{}th protein:".format(i+1))
    pdb_id, chain_id = generator.get_pdb_id(line)
    # print(pdb_id, chain_id)
    generator.download(pdb_id)
    try:
        dist_matrix = c_map.get(pdb_id, chain_id)
        d3_coords = coords.get(pdb_id, chain_id)
        records.extend(generator.get_1st_frag(pdb_id + chain_id))
        # records.extend(generator.get_inp_out_sets(pdb_id + chain_id))
        good_proteins.append(pdb_id + chain_id)
        print("Comment: good")
    except Exception as e:
        print("Comment: corrupted\n")
        bad_proteins.append(pdb_id + chain_id)
        continue
    print(dist_matrix.shape, d3_coords.shape)
    print()

DataUtils.save_itemlist(bad_proteins, CONFIGS.BAD_PDB_IDS)
DataUtils.save_itemlist(good_proteins, CONFIGS.GOOD_PDB_IDS)
DataUtils.save_using_pickle(records, CONFIGS.FULL_DATA_FILE)
# print(good_proteins)
# print(bad_proteins)