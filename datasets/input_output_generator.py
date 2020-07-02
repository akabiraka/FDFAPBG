import sys
sys.path.append('../FDFAPBG')
import math
import torch
import configs.general_config as CONFIGS
import utils.data_utils as DataUtils

class InputOutputGenerator(object):
    def __init(self):
        super(InputOutputGenerator, self).__init__()

    def get_inp_out_sets(self, pdb_code, save_separately=True):
        """
        Given a pdb_code, looks like pdb_code = "5sy8" + "O", 
        this method generates fixed size contact-map and 3d coordinate
        matrix based on WINDOW_SIZE and WINDOW_STRIDE.
            pdb_code: i.e "5sy8" + "O"
            save_separately: whether save data in a single file or separately
            returns the file name as record id
        """
        print("Generating input-output sets for {} ... ...".format(pdb_code))
        full_c_map = DataUtils.read_contact_map_tensor(pdb_code)
        full_d3_coords = DataUtils.read_3d_coords_tensor(pdb_code)
        # print(full_c_map.shape, d3_coords.shape)
        rows, cols = full_c_map.shape
        half_width = math.floor(CONFIGS.WINDOW_SIZE / 2)
        a_input_output_set = []
        all_input_output_set = []
        all_record_ids = []
        k = 0
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
                
                # post-operations: saving contact-map/distance-matrix, molecule-coordinates pair
                if save_separately:
                    # I wanna build up something like: "data/c_map_vs_coord_pairs/5sy8O_k.pkl"
                    filename =  CONFIGS.CONTACT_MAP_VS_COORDINATES_DIR + pdb_code + '_' + str(k) + CONFIGS.DOT_PKL 
                    all_record_ids.append(pdb_code + '_' + str(k))
                    k+=1
                    DataUtils.save_using_pickle(a_input_output_set, filename)
                else:
                    all_input_output_set.append(a_input_output_set)

        if not save_separately:
            # saving (contact-map,coordinates) pairs as protein-wise
            filename =  CONFIGS.CONTACT_MAP_VS_COORDINATES_DIR + pdb_code + CONFIGS.DOT_PKL 
            DataUtils.save_using_pickle(a_input_output_set, filename)
            all_record_ids.append(pdb_code)

        return all_record_ids