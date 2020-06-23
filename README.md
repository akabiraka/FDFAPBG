# Project Title
This project is started from the paper titled "FULLY DIFFERENTIABLE FULL-ATOM PROTEIN BACKBONE GENERATION"

## What it does?
1. Generates [distance-matrix, 3d-coordinates] pair of some fixed window-size and stride. 
2. Distance-matrix can be contact-map based on given threshhold.
3. Distance-matrix and 3d-coordinates are normalized between 0 and 1. 
4. ProteinDataset class is an implementation of pytorch Dataset class which can be used with DataLoader class. 
5. ProteinDataset accesses each data on demand which makes sure the reduction of RAM usage.

## Directories, modules and packages
```
configs/
    __init_.py
    general_config.py
data/
    c_map_vs_coord_pairs/
    contact_maps/
    molecule_coordinates/
    pdbs/
datasets/
    __init_.py
    a_pdb_data.py
    contact_map.py
    data_generator.py
    data_spliter.py
    molecule_coordinates.py
    protein_dataset.py
metrics/
    __init_.py
modules/
    __init_.py
outputs/
    images/
    logs/
    models/
utils
    __init_.py
    clean_slate.py
    data_utils.py
vizualization
    __init_.py
    data_viz.py
    output_viz.py
run.ipnyb
run.py
tester.ipnyb
tester.py
```
## Requirements
Python 3

## How to run?
Makes sure you are in the root directory.
root = "path/to/FDFAPBG/"

**To generate data from scratch:**
```
python datasets/data_generator.py
```
It does five distinct tasks per pdb-id (i.e. 5sy8O). Each of this task can be done separately.
1. Download data from PDB using Biopython.
2. Conpute distance-matrix/contact-map.
3. Prepare 3D-coordinates.
4. For given window-size and stride, it generates fixed size [distance-matrix, 3D-coordinates] pairs by convolving over distance-matrix.
5. It also separates out the corrupted proteins if some molecule does not have some atom.


**To generate full-size contact-map or distance-matrix:**
```
c_map = ContactMap(mat_type="norm_dist", map_type='4N4N')
dist_matrix = c_map.get("5sy8", "O")
```
Some other uses are described in the ```datasets/contact_map.py``` file.

**To generate full-size 3d molecular coordinates:**
```
coords = MoleculeCoordinates(normalized=True)
d3_coords = coords.get("5sy8", "O", ["CA", "CB"])
```
Some other uses are described in the ```datasets/molecule_coordinates.py``` file.

**To split a list of ids (or ids from file):**
```
ds = DataSpliter()
ds.split_from_file(CONFIGS.RECORD_IDS)
```

**To use ```ProteinDataset``` with pytorch DataLoader class:**
```
pd = ProteinDataset(file=CONFIGS.TRAIN_FILE)
pd.__getitem__(0)
```
    
