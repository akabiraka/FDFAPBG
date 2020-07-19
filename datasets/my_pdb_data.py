import sys
sys.path.append("../FDFAPBG")

from Bio.PDB import *
from Bio.PDB.PDBIO import PDBIO, Select

import configs.general_config as CONFIGS

class ChainAndAminoAcidSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id
        
    def  accept_chain(self, chain):
        # print(chain.id)
        if chain.id == self.chain_id:
            return 1
        else:
            return 0
            
    def accept_residue(self, residue):
        if residue.get_resname() in standard_aa_names:
            return 1
        else:
            return 0


class MyPDBData(object):
    """
    Abstract parent class. When new datatype needs to be extracted from PDB data,
    implement this class. Example subclasses are: ContactMap, MoleculeCoordinates.
    """
    def __init__(self, parser=MMCIFParser(QUIET=True)):
        super(MyPDBData, self).__init__()
        self.parser = parser
        self.pdbl = PDBList()
        self.pdbio = PDBIO()
        
    def download(self, pdb_code):
        """
        Download protein data in .cif format in CONFIGS.PDB_DIR.
        """
        self.pdbl.retrieve_pdb_file(pdb_code, pdir=CONFIGS.PDB_DIR, file_format=CONFIGS.CIF)
        
    def clean(self, pdb_id, chain_id):
        """[summary]

        Args:
            pdb_id ([type]): [description]
            chain_id ([type]): [description]

        Returns:
            Structure: Return the cleaned structure
        """
        print("Cleaning non-amino-acid residues and extracting given chain for {}:{} ... ..".format(pdb_id, chain_id))
        pdb_filename = CONFIGS.PDB_DIR + pdb_id + CONFIGS.DOT_CIF
        structure = self.parser.get_structure(pdb_id, pdb_filename)
        self.pdbio.set_structure(structure)
        self.pdbio.save(CONFIGS.CLEAN_PDB_DIR + pdb_id + chain_id + CONFIGS.DOT_PDB, ChainAndAminoAcidSelect(chain_id))
        pdb_filename = CONFIGS.CLEAN_PDB_DIR + pdb_id+chain_id + CONFIGS.DOT_PDB
        return PDBParser(QUIET=True).get_structure(pdb_id, pdb_filename)
        
    def get_chain_from_structure(self, structure, chain_id):
        """Given a structure and chain_id, it returns 
        corresponding chain of the structure

        Args:
            structure ([type]): [description]
            chain_id ([type]): [description]

        Returns:
            chain: protein chain
        """
        models = list(structure.get_models())
        chains = list(models[0].get_chains())
        for chain in chains:
            if chain.id == chain_id:
                return chain
            
    def get_chain_from_clean_pdb(self, pdb_id, chain_id):
        pdb_filename = CONFIGS.CLEAN_PDB_DIR + pdb_id+chain_id + CONFIGS.DOT_PDB
        structure = PDBParser(QUIET=True).get_structure(pdb_id, pdb_filename)
        models = list(structure.get_models())
        chains = list(models[0].get_chains())
        # for each chain
        for chain in chains:
            if chain.id == chain_id:
                return chain