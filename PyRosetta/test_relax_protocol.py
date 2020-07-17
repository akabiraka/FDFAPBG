import sys
sys.path.append('../FDFAPBG')
from Bio.PDB import *
import pandas
from datasets.a_pdb_data import APDBData
from datasets.contact_map import ContactMap
from datasets.molecule_coordinates import MoleculeCoordinates
from datasets.input_output_generator import InputOutputGenerator
import utils.data_utils as DataUtils
from utils.clean_slate import CleanSlate
import configs.general_config as CONFIGS
from PyRosetta.relax_protocol_runner import RelaxProtocolRunner
from datasets.protein_dataset import ProteinDataset
from models.rmsd_loss import RMSD_loss
from evaluations.test_DCNN import TestDCNN
import torch
import numpy as np
import traceback
import random
import pickle

class TestRelaxProtocol(object):
    def __init__(self):
        super(TestRelaxProtocol, self).__init__()
        self.c_map_normalized = ContactMap(mat_type="norm_dist", map_type='4N4N', parser=PDBParser(QUIET=True))
        self.coords_normalized = MoleculeCoordinates(normalized=True, parser=PDBParser(QUIET=True))
        
        self.c_map_not_normalized = ContactMap(mat_type="dist", map_type='4N4N', parser=PDBParser(QUIET=True))
        self.coords_not_normalized = MoleculeCoordinates(normalized=False, parser=PDBParser(QUIET=True))
        
        self.inp_out_gen = InputOutputGenerator()
        self.cln = CleanSlate()
        self.relaxProtocolRunner = RelaxProtocolRunner()
        self.testDCNN = TestDCNN()

    def clean_dirs(self):
        self.cln.clean_all_files(mydir=CONFIGS.CONTACT_MAP_DIR, ext=CONFIGS.DOT_PT)
        self.cln.clean_all_files(mydir=CONFIGS.MOLECULE_COORDINATES_DIR, ext=CONFIGS.DOT_PT)
        self.cln.clean_all_files(mydir=CONFIGS.PDB_DIR, ext=CONFIGS.DOT_CIF)
        
    def create_pdb_structure(self, seq=None, coords=None):
        seq = "MATVKFKYKGEEKQVDISKIKKVWRVGKMISFTYDEGGGKTGRGAVSEKDAPKELLQMLAKQKK"
        coords = np.random.randn(256, 3)
        pdb_file = open("PyRosetta/outputs/1c8c_1A_dcnn_predicted.pdb", "a")
         

    def parse(self, filename="data/1ail.pdb", out_file_prefix="sample", chain_id="A", n_models=None, record_ids_file="record_ids.csv", predicted_statistics_file="stat.csv"):
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
        with open(record_ids_file, "a") as records_ids_filehandle:
            records_ids_filehandle.write("relative_id, record_id\n")
        
        with open(predicted_statistics_file, "a") as predicted_statistics_file_handle:
            # RP_ = relax_protocol
            predicted_statistics_file_handle.write("relative_id, pdb_code, RP_energy_score_1, RP_energy_score_2, RP_energy_score_3, RP_energy_score_4, RP_energy_score_5, RP_energy_score_6, RP_rmsd_score_1, RP_rmsd_score_2, RP_rmsd_score_3, RP_rmsd_score_4, RP_rmsd_score_5, RP_rmsd_score_6, run_time_in_minutes, DCNN_rmsd_score, who_is_better\n")
        
        rosetta_file_content = open(filename, "r")
        records = []
        bad_records = []
        relative_id = 0 # relative id according to whole pdb file
        ith_model = 1
        bb_atoms = pandas.DataFrame(np.tile(["CA", "CB", "N", "O"], 64))
        output_file_handle, pdb_id = self.update_out_file_handle(ith_model, CONFIGS.PDB_DIR, out_file_prefix)
        
        relax_protocol_better_flag, dcnn_beter_flag, relax_dcnn_close_flag = False, False, False
        
        for i, line in enumerate(rosetta_file_content):
            if "END MODEL" in line :
                output_file_handle.write(line) # saving last line 'END MODEL' in the model file"
                output_file_handle.close()
                relative_id += 1
                
                # skipping 1st 24 models
                if relative_id < 0: 
                    self.clean_dirs()
                    output_file_handle, pdb_id = self.update_out_file_handle(ith_model, CONFIGS.PDB_DIR, out_file_prefix)
                    continue
                
                # check if to select this model to work with with 50% probability
                if random.uniform(0, 1) < 0.5: 
                    print("Comment: skipped {}th models\n".format(relative_id))
                    self.clean_dirs()
                    output_file_handle, pdb_id = self.update_out_file_handle(ith_model, CONFIGS.PDB_DIR, out_file_prefix)
                    continue

                print("working for {}th model relative to {} models".format(ith_model, relative_id))
                try:
                    self.relaxProtocolRunner.modify_pose(pdb_id=pdb_id, chain_id=chain_id)
                    # running relax protocol
                    dist_matrix_not_normalized = self.c_map_not_normalized.get(pdb_id, chain_id)
                    d3_coords_not_normalized = self.coords_not_normalized.get(pdb_id, chain_id, CONFIGS.BACKBONE_ATOMS)
                    out = pandas.concat([bb_atoms, pandas.DataFrame(d3_coords_not_normalized)], axis=1)
                    out.to_csv("PyRosetta/outputs/{}_coords_ground_truth_not_normalized.xyz".format(pdb_id+chain_id), header=False, index=False, sep=" ")
                    np.savetxt("PyRosetta/outputs/{}_coords_ground_truth_not_normalized.txt".format(pdb_id+chain_id), d3_coords_not_normalized, fmt='%.2f', newline=" ")
                    
                    # energy_scores, rmsd_scores_relax, run_time, filtered_seq = self.relaxProtocolRunner.run(pdb_id=pdb_id, chain_id=chain_id, dist_mat=dist_matrix_not_normalized)
                    # rmsd_score_relax = min(rmsd_scores_relax)
                    # finished relax protocol
                    
                    # running DCNN
                    # dist_matrix_normalized = self.c_map_normalized.get(pdb_id, chain_id)
                    # d3_coords_normalized = self.coords_normalized.get(pdb_id, chain_id, CONFIGS.BACKBONE_ATOMS)
                    # out = pandas.concat([bb_atoms, pandas.DataFrame(d3_coords_normalized)], axis=1)
                    # out.to_csv("PyRosetta/outputs/{}_coords_ground_truth_normalized.xyz".format(pdb_id+chain_id), header=False, index=False, sep=" ")
                    # np.savetxt("PyRosetta/outputs/{}_coords_ground_truth_normalized.txt".format(pdb_id+chain_id), d3_coords_normalized, fmt='%.2f', newline=" ")
                    record_ids = self.inp_out_gen.get_inp_out_sets(pdb_id+chain_id, save_separately=True)
                    pd = ProteinDataset(data_dir="data/c_map_vs_coord_pairs/", record_ids=record_ids)
                    protein_data_loader = torch.utils.data.DataLoader(pd, batch_size=1, shuffle=False)
                    _, _, rmsd_scores_DCNN, rmsd_score_mean_DCNN, _, _, predicted_coords = self.testDCNN.test(protein_data_loader)
                    predicted_coords = predicted_coords.squeeze(0).squeeze(0).cpu().numpy()                    
                    out = pandas.concat([bb_atoms, pandas.DataFrame(predicted_coords)], axis=1)
                    out.to_csv("PyRosetta/outputs/{}_coords_dcnn_predicted.xyz".format(pdb_id+chain_id), header=False, index=False, sep=" ")
                    np.savetxt("PyRosetta/outputs/{}_coords_dcnn_predicted.txt".format(pdb_id+chain_id), predicted_coords, fmt='%.2f', newline=" ")
                    print("rmsd score: DCNN: {}".format(rmsd_score_mean_DCNN))
                    # finished DCNN
                    
                    
                    with open(record_ids_file, "a") as records_ids_filehandle:
                        for record_id in record_ids:
                            records_ids_filehandle.write("{}, {}\n".format(relative_id, record_id))
                            
                    # with open(predicted_statistics_file, "a") as predicted_statistics_file_handle:
                    #     predicted_statistics_file_handle.write("{}, {}, ".format(relative_id, pdb_id+chain_id))
                    #     for energy_score in energy_scores:
                    #         predicted_statistics_file_handle.write("{}, ".format(energy_score))
                    #     for rmsd_score in rmsd_scores_relax:
                    #         predicted_statistics_file_handle.write("{}, ".format(rmsd_score))
                    #     predicted_statistics_file_handle.write("{}, {}, ".format(run_time, rmsd_score_mean_DCNN))
                    
                    #     if rmsd_score_relax < rmsd_score_mean_DCNN: 
                    #         predicted_statistics_file_handle.write("RelaxProtocol")
                    #         relax_protocol_better_flag = True
                    #     if rmsd_score_relax > rmsd_score_mean_DCNN: 
                    #         predicted_statistics_file_handle.write("DCNN")
                    #         dcnn_beter_flag = True 
                    #     if abs(rmsd_score_relax - rmsd_score_mean_DCNN) < 0.5: 
                    #         predicted_statistics_file_handle.write("RelaxProtocol-DCNN")
                    #         relax_dcnn_close_flag = True
                    #     predicted_statistics_file_handle.write("\n")
                    
                    # if relax_protocol_better_flag==True and dcnn_beter_flag==True and relax_dcnn_close_flag==True: break
                    
                    # print("rmsd score: DCNN: {}, RelaxProtocol: {}".format(rmsd_score_mean_DCNN, rmsd_score_relax))
                    # print(relax_protocol_better_flag, dcnn_beter_flag, relax_dcnn_close_flag)
                    
                    self.clean_dirs()
                    
                except Exception as e:
                    traceback.print_exc()
                    bad_records.append(pdb_id+chain_id)

                self.clean_dirs()

                print("\n")
                if n_models is not None and ith_model==n_models: break
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

test_relax_protocol = TestRelaxProtocol()
test_relax_protocol.parse(filename="PyRosetta/data/1C8CA_Output_RosettaDecoys_SubDirs0-999999-002.pdb", \
    out_file_prefix="1c8c", chain_id="A", record_ids_file="PyRosetta/outputs/1c8cA_record_ids.csv", n_models=5, \
        predicted_statistics_file="PyRosetta/outputs/1c8cA_predicted_statistics.csv")
