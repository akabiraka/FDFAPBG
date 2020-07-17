import sys
sys.path.append('../FDFAPBG')

import os
import numpy as np
import time
import pyrosetta
import rosetta
pyrosetta.init()

import configs.general_config as CONFIGS

AMINO_ACID = 'ACDEFGHIKLMNPQRSTVWY'
BACKBONE_ATOMS = ['CA', 'CB', 'N', 'O']
# http://faculty.washington.edu/dimaio/files/HW_2_2020.pdf
#                          GLY         ASP/ASN     ILE/VAL     PRO         OTHERS
angle_distributions = [[[87, 7], [-140, 165], [-132, 153], [-64, 145], [-136, 153]],
                        [[-66, -35], [-78, 141], [-86, 127], [-60, -29], [-76, 143]],
                        [[70, -144], [-108, 103], [-118, 125], [-60, -29], [-112, 119]],
                        [[105, 170], [-97, 5], [-91, -9], [-77, 161], [-91, -9]],
                        [[-171, 177], [-64, -39], [-63, -42], [-77, 161], [-63, -42]],
                        [[-87, 163], [57, 39], [57, 39], [-84, -2], [57, 39]]]


class RelaxProtocolRunner(object):
    def __init__(self):
        super(RelaxProtocolRunner, self).__init__()

    def filter_aa_residues(self, chain):
        """
        A chain can be heteroatoms(water, ions, etc; anything that 
        isn't an amino acid or nucleic acid)
        so this function get rid of atoms excepts amino-acids
        """
        seq = ""
        for i, ch in enumerate(chain):
            if ch in AMINO_ACID:
                seq += ch
        return seq
    
    def init_ramachandran_angles(self, pose, angles):
        n_residues = pose.total_residue()
        for i in range(1, n_residues+1):
            ith_residue = pose.residue(i).name()
            if "GLY" in ith_residue:
                pose.set_phi(i, angles[0][0])
                pose.set_psi(i, angles[0][1])
            elif "ASP" in ith_residue or "ASN" in ith_residue:
                pose.set_phi(i, angles[1][0])
                pose.set_psi(i, angles[1][1])
            elif "ILE" in ith_residue or "VAL" in ith_residue:
                pose.set_phi(i, angles[2][0])
                pose.set_psi(i, angles[2][1])
            elif "PRO" in ith_residue:
                pose.set_phi(i, angles[3][0])
                pose.set_psi(i, angles[3][1])
            else:
                pose.set_phi(i, angles[4][0])
                pose.set_psi(i, angles[4][1])
                
        return pose

    def get_atom_pair_ids(self, pose, i=0, ith_atom_idx="CA", j=0, jth_atom_idx="CA"):
        res_name_1 = pose.residue(i).name()
        res_name_2 = pose.residue(j).name()
        
        if ith_atom_idx=="CB" and 'GLY' in res_name_1:
            ith_atom_idx = "CA"
        
        if jth_atom_idx=="CB" and "GLY" in res_name_2 :
            jth_atom_idx = "CA"
        
        id_i = pyrosetta.AtomID(pose.residue(i).atom_index(ith_atom_idx), i)
        id_j = pyrosetta.AtomID(pose.residue(j).atom_index(jth_atom_idx), j)
        # print(id_i, id_j)
        return id_i, id_j
        
    def add_constraint_from_4n4n_distance_map(self, pose, dist_mat):
        """[Add distance constraints to each residue]

        Args:
            pose (pyrosetta pose): 
            dist_mat (2d matrix): (4n, 4n) distance matrix where n is the length of the sequence in pose
        """
        # dist_mat: (4n, 4n)
        n_residues = pose.total_residue()
        for i in range(1, n_residues+1):
            for j in range(1, n_residues+1):
                for k, atom_1 in enumerate(BACKBONE_ATOMS):
                    for l, atom_2 in enumerate(BACKBONE_ATOMS):
                        distance_value = dist_mat[4*(i-1)+k][4*(j-1)+l]
                        distance_lower_bound = distance_value-1 if distance_value-1 > 0 else 0
                        distance_upper_bound = distance_value+1
                        id_i, id_j = self.get_atom_pair_ids(pose, i, atom_1, j, atom_2)
                        ij_func = rosetta.core.scoring.constraints.BoundFunc(distance_lower_bound, distance_upper_bound, 1.0, "cst1")
                        ij_cnstrnt = rosetta.core.scoring.constraints.AtomPairConstraint(id_i, id_j, ij_func)
                        pose.add_constraint(ij_cnstrnt)
        return pose

   
    def modify_pose(self, pdb_id, chain_id):
        print("Modifying existing pdb ... ...")
        pose = pyrosetta.pose_from_file(CONFIGS.PDB_DIR+pdb_id+CONFIGS.DOT_CIF)
        pose.dump_pdb("PyRosetta/outputs/{}.pdb".format(pdb_id+chain_id))
        for r in range(1, pose.total_residue() + 1 ):
            for atom in ["CA", "CB", "N", "O"]:
                if atom=="CB" and 'GLY' in pose.residue(r).name():
                    atom = "CA"
                    continue
                
                xyz = pose.residue(r).xyz(atom)
                # print("previous: {}".format(xyz))
                if xyz[0] < 1e-100 and xyz[1] < 1e-100 and xyz[2] <1e-100:
                    continue
                xyz = xyz + np.random.uniform(low=-1.0, high=1.0, size=(1, 3))[0]
                xyz_modified = pyrosetta.rosetta.numeric.xyzVector_double_t(xyz[0], xyz[1], xyz[2])
                pose.residue(r).set_xyz(atom, xyz_modified)
                # print("modified: {}".format(pose.residue(r).xyz(atom)))
        pose.dump_pdb(CONFIGS.PDB_DIR+pdb_id+CONFIGS.DOT_CIF)
        pose.dump_pdb("PyRosetta/outputs/{}_modified.pdb".format(pdb_id+chain_id))        
            # print("previous: {}".format(pose.residue(r).xyz("N")))
            # CA_xyz_modified = pose.residue(r).xyz("CA") + np.random.uniform(low=-5.0, high=5.0, size=(1, 3))[0]
            # CA_xyz_modified = pyrosetta.rosetta.numeric.xyzVector_double_t(CA_xyz_modified[0], CA_xyz_modified[1], CA_xyz_modified[2])
            # pose.residue(r).set_xyz("CA", CA_xyz_modified)
            # print("modified: {}".format(pose.residue(r).xyz("N")))
            
        
    def run(self, pdb_id, chain_id, dist_mat):
        native_pose_pdb = pyrosetta.pose_from_file(CONFIGS.PDB_DIR+pdb_id+CONFIGS.DOT_CIF)
        prediction_out_dir_name = pdb_id+chain_id
        # os.makedirs("PyRosetta/outputs/"+prediction_out_dir_name)
        native_pose_pdb.dump_pdb("PyRosetta/outputs/{}.pdb".format(pdb_id+chain_id))
        seq = native_pose_pdb.sequence()
        filtered_seq = self.filter_aa_residues(seq)
        print(seq,"\n",filtered_seq)
        
        scorefxn_scores = []
        rmsd_scores = []
        start_time = time.time()
        
        best_pose_by_rmsd_score = None
        best_pose_by_energy_score = None
        best_rmsd_score = np.inf
        best_energy_score = np.inf
        for i, angle in enumerate(angle_distributions):
            print("Running relax protocol for {}:{} with {}th angle distribution".format(pdb_id, chain_id, i+1))
            pose = pyrosetta.pose_from_sequence(filtered_seq)
            print("dist-mat: {}, residues: {}".format(dist_mat.shape, pose.total_residue()))
            # raise
            
            # initializing pose with ramachandran angle distribution
            pose = self.init_ramachandran_angles(pose, angle)
            # pose.dump_pdb("{}_init.pdb".format(pdb_id+chain_id))

            # adding distance constrains for all pair of residues
            pose = self.add_constraint_from_4n4n_distance_map(pose, dist_mat)
            
            # defining scoring function        
            scorefxn = pyrosetta.get_fa_scorefxn()
            scorefxn.set_weight(rosetta.core.scoring.atom_pair_constraint, 1.0)

            # defining FastRelax protocol
            print("estimating pose using relax protocol ... ...")
            fastrelax = rosetta.protocols.relax.FastRelax(scorefxn, 5)
            fastrelax.apply(pose)
            
            # pose.dump_pdb("PyRosetta/outputs/{}/{}_using_real_dist_angle_{}.pdb".format(prediction_out_dir_name, pdb_id+chain_id, i+1))
            # pose = pyrosetta.pose_from_file("PyRosetta/outputs/{}/{}_using_real_dist_angle_{}.pdb".format(prediction_out_dir_name, pdb_id+chain_id, i+1))
            rmsd_score = pyrosetta.rosetta.core.scoring.all_atom_rmsd(native_pose_pdb, pose)
            energy_score = scorefxn(pose)
            
            scorefxn_scores.append(energy_score)
            rmsd_scores.append(rmsd_score)
            
            if rmsd_score < best_rmsd_score:
                best_rmsd_score = rmsd_score
                best_pose_by_rmsd_score = pose
                
            if energy_score < best_energy_score:
                best_energy_score = energy_score
                best_pose_by_energy_score = pose
                
            if i==0: break
            
        best_pose_by_rmsd_score.dump_pdb("PyRosetta/outputs/{}_best_pose_by_rmsd_score_pyrosetta.pdb".format(prediction_out_dir_name, pdb_id+chain_id))  
        best_pose_by_energy_score.dump_pdb("PyRosetta/outputs/{}_best_pose_by_energy_score_pyrosetta.pdb".format(prediction_out_dir_name, pdb_id+chain_id))            
        
        return scorefxn_scores, rmsd_scores, (time.time() - start_time)/60, filtered_seq
    
# relaxProtocolRunner = RelaxProtocolRunner()
# relaxProtocolRunner.run(pdb_id="1ail_2", chain_id="A", dist_mat=np.random.randn(292, 292))