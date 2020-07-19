import sys
sys.path.append('../FDFAPBG')
from termcolor import colored

from Bio.PDB import *
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.StructureBuilder import StructureBuilder

import configs.general_config as CONFIGS
       
class ProteinSampler():
    def __init__(self):
        super(ProteinSampler, self).__init__()
        self.pdbio = PDBIO()
        
    def sample_from_chain(self, pdb_id, chain_id, chain):
        print("Sampling protein tertiary structure {}:{} ... ...".format(pdb_id, chain_id))
        residues = list(chain.get_residues())
        fragment_ids = []
        if len(residues) >= CONFIGS.FRAGMENT_SIZE: 
            try:
                fragments = [residues[i:i + CONFIGS.FRAGMENT_SIZE] for i in range(0, len(residues)-CONFIGS.FRAGMENT_SIZE+1, CONFIGS.FRAGMENT_STRIDE)]
                print(colored("{} fragments found. Protein has {} amino-acids whereas mnimum fragment size is {}.".format(len(fragments), len(chain), CONFIGS.FRAGMENT_SIZE), "green"))
                for i, fragment in enumerate(fragments):
                    fragment_id = pdb_id + chain_id + "_" + str(i)
                    structure = self.build_structure(fragment, chain_id)
                    self.pdbio.set_structure(structure)
                    self.pdbio.save(CONFIGS.FRAGMENTS_DIR + fragment_id + CONFIGS.DOT_PDB)
                    fragment_ids.append(fragment_id)
            except Exception as e:
                print(colored("Error happended while saving the fragment for {}:{}.".format(pdb_id, chain_id), "red"))
                raise
                pass
        else:
            print(colored("No fragment found. Protein has {} amino-acids whereas mnimum fragment size is {}.".format(len(chain), CONFIGS.FRAGMENT_SIZE), "yellow"))
        
        return fragment_ids
        
    
    def build_structure(self, fragments, chain_id="A"):
        sb = StructureBuilder()
        sb.init_structure("pdb")
        sb.init_seg(" ")
        sb.init_model(0)
        sb.init_chain(chain_id)
        for residue in fragments:
            sb.structure[0][chain_id].add(residue.copy())
        return sb.structure