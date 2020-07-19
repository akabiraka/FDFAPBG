# window definations while cropping fixed side data
WINDOW_SIZE = 256 # since 0 to 255 is 256
WINDOW_STRIDE = 256

FRAGMENT_SIZE = 64
FRAGMENT_STRIDE = 64

# data directories
DATA_DIR = "data/"
PDB_DIR = DATA_DIR + "pdbs/"
CLEAN_PDB_DIR = DATA_DIR + "clean_pdbs/"
CONTACT_MAP_DIR = DATA_DIR + "contact_maps/"
MOLECULE_COORDINATES_DIR = DATA_DIR + "molecule_coordinates/"
CONTACT_MAP_VS_COORDINATES_DIR = DATA_DIR + "c_map_vs_coord_pairs/"
FRAGMENTS_DIR = DATA_DIR + "fragments/"

# input files
ALL_PDB_IDS = DATA_DIR + "all_pdb_ids_tiny.txt"
BAD_PDB_IDS = DATA_DIR + "bad_pdb_ids.txt"
GOOD_PDB_IDS = DATA_DIR + "good_pdb_ids.txt"
TRAIN_FILE = DATA_DIR + 'train.txt'
VAL_FILE = DATA_DIR + 'val.txt'
TEST_FILE = DATA_DIR + 'test.txt'
RECORD_IDS = DATA_DIR + 'record_ids.txt'

# file format
CIF = 'mmCif'

# file extensions
DOT_CIF = ".cif"
DOT_PDB = ".pdb"
DOT_PT = ".pt"
DOT_PKL = ".pkl"
DOT_TXT = ".txt"

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
BACKBONE_ATOMS = ['CA', 'CB', 'N', 'O']