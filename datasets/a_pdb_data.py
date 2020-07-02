import sys
sys.path.append("../FDFAPBG")

from Bio.PDB import *

import configs.general_config as CONFIGS

class APDBData(object):
    """
    Abstract parent class. When new datatype needs to be extracted from PDB data,
    implement this class. Example subclasses are: ContactMap, MoleculeCoordinates.
    """
    def __init__(self, parser=MMCIFParser(QUIET=True)):
        super(APDBData, self).__init__()
        self.parser = parser

    def filter_aa_residues(self, chain):
        """
        A chain can be heteroatoms(water, ions, etc; anything that 
        isn't an amino acid or nucleic acid)
        so this function get rid of atoms excepts amino-acids
        """
        aa_residues = []
        non_aa_residues = []
        non_aa = []
        seq = ""
        for i in chain:
            if i.get_resname() in standard_aa_names:
                aa_residues.append(i)
                seq += CONFIGS.AMINO_ACID_3TO1[i.get_resname()]
            else:
                non_aa.append(i.get_resname())
                non_aa_residues.append(i.get_resname())
        return aa_residues, seq, non_aa_residues

    def get_a_chain(self, pdb_id, chain_id):
        """
        Returns a chain filterd with only amino-acid residues.
        """
        pdb_filename = CONFIGS.PDB_DIR + pdb_id + CONFIGS.DOT_CIF
        # reading whole structure
        structure = self.parser.get_structure(pdb_id, pdb_filename)
        models = list(structure.get_models())
        chains = list(models[0].get_chains())
        # for each chain
        for chain in chains:
            if chain.id == chain_id:
                all_residues = list(chain.get_residues())
                aa_residues, seq, non_aa_residues = self.filter_aa_residues(all_residues)
                return aa_residues
    
    def get(self, pdb_id, chain_id):
        """
        Abstract method. Needs to be implemented, when any data needs to be
        from pdb data.
        """
        raise NotImplementedError("Please implement this method")
            
# pdb_data = APDBData()
# print(pdb_data.get_a_chain("5sy8", "O"))