import sys
sys.path.append('../FDFAPBG')

import numpy as np
import random

import configs.general_config as CONFIGS
import utils.data_utils as DataUtils


class DataSpliter(object):
    """docstring for DataSpliter."""

    def __init__(self):
        super(DataSpliter, self).__init__()

    def split_from_list(self, ids, train_size=.60, val_size=.20, save=True):
        """
        ids: list of ids. i.e. ['a', 'b', 'c']
        Returns: train_set, val_set, test_set
        # test_size = .20
        """
        random.shuffle(ids)
        to_train_index = round(len(ids) * train_size)
        train_set = ids[0:to_train_index]
        remaining = ids[to_train_index:]
        to_val_index = round(len(ids) * val_size)
        val_set = remaining[0:to_val_index]
        test_set = remaining[to_val_index:]
        print("train size:", len(train_set), "val size:", len(val_set), "test size:", len(test_set))
        DataUtils.save_itemlist(train_set, CONFIGS.TRAIN_FILE)
        DataUtils.save_itemlist(val_set, CONFIGS.VAL_FILE)
        DataUtils.save_itemlist(test_set, CONFIGS.TEST_FILE)
        return train_set, val_set, test_set

    def split_from_file(self, file, train_size=.60, val_size=.20, save=True):
        """
        file should have only ids. An id per line.
        i.e. CONFIGS=CONSTANTS.ALL_PDB_IDS
        It uses class method named split_from_list().
        Returns: train_set, val_set, test_set
        """
        file_content = open(file).read()
        return self.split_from_list(file_content.split())
    
    def split_from_cullpdb_file(self, file, train_size=.60, val_size=.20, save=True):
        """cullpdb file as 6 columns. Only extract the pdb_code from 1st column.

        Args:
            file (string): file path
            train_size (float, optional): [description]. Defaults to .60.
            val_size (float, optional): [description]. Defaults to .20.
            save (bool, optional): [description]. Defaults to True.

        Returns:
            3 lists: train_set, val_set, test_set
        """
        file_handle = open(file, "r")
        pdb_codes = []
        for i, line in enumerate(file_handle):
            pdb_codes.append(line.split()[0])
        
        # random.shuffle(pdb_codes)
        return self.split_from_list(pdb_codes)

    
ds = DataSpliter()
# ds.split_from_file(CONFIGS.RECORD_IDS)
ds.split_from_cullpdb_file("data/cullpdb_pc20_res1.6_R0.25_d200618_chains3680")
        
