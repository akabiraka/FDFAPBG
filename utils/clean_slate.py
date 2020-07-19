import sys
sys.path.append('../FDFAPBG')
import os

import configs.general_config as CONFIGS

class CleanSlate(object):
    def __init__(self):
        super(CleanSlate, self).__init__()

    def clean_all(self):
        cln.clean_all_files(mydir=CONFIGS.CONTACT_MAP_VS_COORDINATES_DIR, ext=CONFIGS.DOT_PKL)
        cln.clean_all_files(mydir=CONFIGS.CLEAN_PDB_DIR, ext=CONFIGS.DOT_PDB)
        cln.clean_all_files(mydir=CONFIGS.CONTACT_MAP_DIR, ext=CONFIGS.DOT_PT)
        cln.clean_all_files(mydir=CONFIGS.FRAGMENTS_DIR, ext=CONFIGS.DOT_PDB)
        cln.clean_all_files(mydir=CONFIGS.MOLECULE_COORDINATES_DIR, ext=CONFIGS.DOT_PT)
        cln.clean_all_files(mydir=CONFIGS.PDB_DIR, ext=CONFIGS.DOT_CIF)
        cln.clean_all_files(mydir=CONFIGS.PDB_DIR, ext=CONFIGS.DOT_PDB)

    def clean_all_files(self, mydir, ext):
        """
        dir: "/data/pdbs"
        ext: ".cif"
        """
        for f in os.listdir(mydir): 
            if f.endswith(ext): 
                os.remove(os.path.join(mydir, f))


cln = CleanSlate()
cln.clean_all()