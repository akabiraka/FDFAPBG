# import numpy as np
# import pandas as pd
# from Bio.PDB import *
# from Bio.SVDSuperimposer import SVDSuperimposer
# bb_atoms = pd.DataFrame(np.tile(["CA", "CB", "N", "O"], 64))

# dcnn_predicted = np.loadtxt("PyRosetta/outputs/1c8c_1A_coords_dcnn_predicted.txt").reshape(256, 3)
# out = pd.concat([bb_atoms, pd.DataFrame(dcnn_predicted*3)], axis=1)
# out.to_csv("outputs/test_dcnn.xyz", header=None, index=None, sep=" ")

# ground_truth = np.loadtxt("PyRosetta/outputs/1c8c_1A_coords_ground_truth_normalized.txt").reshape(256, 3)
# out = pd.concat([bb_atoms, pd.DataFrame(ground_truth)], axis=1)
# out.to_csv("outputs/test_ground_truth.xyz", header=None, index=None, sep=" ")
# https://biopython.org/DIST/docs/api/Bio.SVDSuperimposer.SVDSuperimposer-class.html
# sup = SVDSuperimposer()
# sup.set(ground_truth, dcnn_predicted)
# sup.run()
# rms = sup.get_rms()
# print(rms)

# parser = PDBParser(QUIET=True) 
# pdb_ground_truth = parser.get_structure("x", "PyRosetta/1c8c_1A_bb.pdb")
# pdb_dcnn_predicted = parser.get_structure("x", "PyRosetta/1c8c_1A_coords_dcnn_predicted_bb.pdb")

# fixed = Selection.unfold_entities(Polypeptide.PPBuilder().build_peptides(pdb_ground_truth, aa_only=True)[0], 'A')
# moving = Selection.unfold_entities(Polypeptide.PPBuilder().build_peptides(pdb_dcnn_predicted, aa_only = True)[0], 'A')

# sup = Superimposer()
# sup.set_atoms(fixed , moving)
# print(sup.rms)
# sup = Superimposer()
# sup.set_atoms(fixed , moving)
# print(sup.rms)


# def scale(X, x_min, x_max):
#     nom = (X-X.min(axis=0))*(x_max-x_min)
#     denom = X.max(axis=0) - X.min(axis=0)
#     denom[denom==0] = 1
#     return x_min + nom/denom

# import numpy as np
# from Bio.SVDSuperimposer import SVDSuperimposer
# from sklearn.preprocessing import normalize 

# x = np.random.uniform(low=-5.0, high=10.5, size=(256,3))
# normalized_x = normalize(x, norm="l2", axis=1)

# xmax, xmin = x.max(), x.min()
# normalized_x = (x - xmin)/(xmax - xmin)

# # normalized_x = scale(x, 0, 1)
# print(x, normalized_x)
