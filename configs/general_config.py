# data directories
DATA_DIR = "data/"
PDB_DIR = DATA_DIR + "pdbs/"

# input files
ALL_PDB_IDS = DATA_DIR + "all_pdb_ids_tiny.txt"
BAD_PDB_IDS = DATA_DIR + "bad_pdb_ids.txt"
GOOD_PDB_IDS = DATA_DIR + "good_pdb_ids.txt"

# file format
CIF = 'mmCif'

# file extensions
DOT_CIF = ".cif"

# output directories
OUTPUT_DIR = "outputs/"
OUTPUT_IMAGES_DIR = OUTPUT_DIR + "images/"
OUTPUT_MODELS_DIR = OUTPUT_DIR + "models/"
OUTPUT_LOGS_DIR = OUTPUT_DIR + "logs/"

# 20 amino-acids
AMINO_ACID_3TO1 = {'ALA': 'A',
                   'CYS': 'C',
                   'ASP': 'D',
                   'GLU': 'E',
                   'PHE': 'F',
                   'GLY': 'G',
                   'HIS': 'H',
                   'ILE': 'I',
                   'LYS': 'K',
                   'LEU': 'L',
                   'MET': 'M',
                   'ASN': 'N',
                   'PRO': 'P',
                   'GLN': 'Q',
                   'ARG': 'R',
                   'SER': 'S',
                   'THR': 'T',
                   'VAL': 'V',
                   'TRP': 'W',
                   'TYR': 'Y'}