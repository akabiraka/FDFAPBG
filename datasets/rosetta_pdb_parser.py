import sys
sys.path.append('../FDFAPBG')
from Bio.PDB import *

from datasets.a_pdb_data import APDBData
from datasets.contact_map import ContactMap
from datasets.molecule_coordinates import MoleculeCoordinates
from datasets.input_output_generator import InputOutputGenerator
import utils.data_utils as DataUtils
from utils.clean_slate import CleanSlate
import configs.general_config as CONFIGS
import traceback
import random

class RosettaPDBParser(object):
    def __init__(self):
        super(RosettaPDBParser, self).__init__()
        self.c_map = ContactMap(mat_type="norm_dist", map_type='4N4N', parser=PDBParser(QUIET=True))
        self.coords = MoleculeCoordinates(parser=PDBParser(QUIET=True))
        self.inp_out_gen = InputOutputGenerator()
        self.cln = CleanSlate()

    def clean_dirs(self):
        self.cln.clean_all_files(mydir=CONFIGS.CONTACT_MAP_DIR, ext=CONFIGS.DOT_PT)
        self.cln.clean_all_files(mydir=CONFIGS.MOLECULE_COORDINATES_DIR, ext=CONFIGS.DOT_PT)
        self.cln.clean_all_files(mydir=CONFIGS.PDB_DIR, ext=CONFIGS.DOT_CIF)

    def parse(self, filename="data/1ail.pdb", out_file_prefix="sample", chain_id="A", n_models=None):
        """
        Each model will be dumped into separate file for RAM usage minimization.
        Input parameters:
            filename: a .pdb file contains rosetta created models which will be parsed
            out_file_prefix: i.e pdb_id
            chain_id: which chain to use to compute distance-matrix/contact-map and 3d-coordinates
            n_models: n number of models to dump. If None then all models will be dumped.
        Returns:
            in out_dir output_file_prefix_i.pdb: "i" means i-th model.

        """
        records_ids_filehandle = open(CONFIGS.RECORD_IDS, "a")
        rosetta_file_content = open(filename, "r")
        ith_model = 1
        output_file_handle, pdb_id = self.update_out_file_handle(ith_model, CONFIGS.PDB_DIR, out_file_prefix)
        records = []
        bad_records = []
        n_skipped = 1
        for i, line in enumerate(rosetta_file_content):
            if "END MODEL" in line :
                output_file_handle.write(line) # saving last line 'END MODEL' in the model file"
                output_file_handle.close()
                
                # check if to select this model to work with with 50% probability
                if random.uniform(0, 1) < 0.5: 
                    print("Comment: skipped {}th models\n".format(n_skipped))
                    n_skipped += 1
                    self.clean_dirs()
                    output_file_handle, pdb_id = self.update_out_file_handle(ith_model, CONFIGS.PDB_DIR, out_file_prefix)
                    continue

                print("working for {}th model".format(ith_model))
                try:
                    dist_matrix = self.c_map.get(pdb_id, chain_id)
                    d3_coords = self.coords.get(pdb_id, chain_id, CONFIGS.BACKBONE_ATOMS)
                    # records.extend(self.inp_out_gen.get_inp_out_sets(pdb_id+chain_id, save_separately=True))
                    record_ids = self.inp_out_gen.get_inp_out_sets(pdb_id+chain_id, save_separately=True)
                    # records_ids_filehandle.write(record_ids)
                    for record_id in record_ids:
                        records_ids_filehandle.write("%s\n" % record_id)
                except Exception as e:
                    traceback.print_exc()
                    bad_records.append(pdb_id+chain_id)

                self.clean_dirs()

                print("\n")
                if n_models is not None and ith_model==n_models:
                    break
                ith_model += 1
                output_file_handle, pdb_id = self.update_out_file_handle(ith_model, CONFIGS.PDB_DIR, out_file_prefix)

            else:
                output_file_handle.write(line) 

        # DataUtils.save_itemlist(records, CONFIGS.RECORD_IDS)
        DataUtils.save_itemlist(bad_records, CONFIGS.BAD_PDB_IDS)
    
    def update_out_file_handle(self, model_id, out_dir="data/pdbs/", out_file_prefix="sample"):
        pdb_id = out_file_prefix + "_" + str(model_id)
        filename = out_dir + pdb_id + CONFIGS.DOT_CIF
        file_handle = open(filename, "w")
        return file_handle, pdb_id

rosetta_pdb_parser = RosettaPDBParser()
rosetta_pdb_parser.parse(filename="data/1FWP_Output_RosettaDecoys_SubDirs0-99999-002.pdb", out_file_prefix="1fwp", chain_id="A", n_models=5000)
