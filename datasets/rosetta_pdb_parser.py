import sys
sys.path.append('../FDFAPBG')
import os

class RosettaPDBParser(object):
    def __init__(self):
        super(RosettaPDBParser, self).__init__()

    def parse(self, filename="data/1ail.pdb", out_dir="data/pdbs/", out_file_prefix="sample", n_models=None):
        """
        Each model will be dumped into separate file for RAM usage minimization.
        Input parameters:
            filename: a .pdb file contains rosetta created 1 to multiple models
            out_dir: the directory where the output models will be dumped
            out_file_prefix: i.e pdb_id
            n_models: n number of models to dump. If None then all models will be dumped.
        Returns:
            The output file should look like "output_file_prefix_i.pdb". 
            "i" means i-th model.
        """
        out_file_ids_handle = open("data/models/" + out_file_prefix + ".txt", "w")
        rosetta_file_content = open(filename, "r")
        ith_model = 1
        output_file_handle, out_file_id = self.update_out_file_handle(ith_model, out_dir, out_file_prefix)
        for i, line in enumerate(rosetta_file_content):
            if "END MODEL" in line :
                output_file_handle.write(line)
                out_file_ids_handle.write(out_file_id+"\n")
                print("saved {}th model".format(ith_model))
                if n_models is not None and ith_model==n_models:
                    break
                ith_model += 1
                output_file_handle, out_file_id = self.update_out_file_handle(ith_model, out_dir, out_file_prefix)

            else:
                output_file_handle.write(line) 

        os.remove(out_dir + out_file_id + ".pdb")
    
    def update_out_file_handle(self, model_id, out_dir="data/pdbs/", out_file_prefix="sample"):
        out_file_id = out_file_prefix + "_" + str(model_id)
        filename = out_dir + out_file_id + ".pdb"
        file_handle = open(filename, "w")
        return file_handle, out_file_id

rosetta_pdb_parser = RosettaPDBParser()
rosetta_pdb_parser.parse(filename="data/1AIL_Output_RosettaDecoys_SubDirs0-509999-002.pdb", out_dir="data/pdbs/", out_file_prefix="1ail", n_models=100)
