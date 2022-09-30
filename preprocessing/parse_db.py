import glob
import os
import numpy as np
from ase.io.vasp import read_vasp_xdatcar, read_vasp_out

import sys
sys.path.insert(0, '../schnetpack_active/src/')
from schnetpack import AtomsLoader, AtomsData
from schnetpack.environment import TorchEnvironmentProvider

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cutoff', type=float, default=6.0, help='Cutoff value for nearest neighbor collection') 
args = parser.parse_args()



#### Create DFT databases

pristine = glob.glob("/people/pang248/potential_files/2-DFT/1-*/*/")
defect_diagonal = glob.glob("/people/pang248/potential_files/2-DFT/2-*/*/")
defect_planar = glob.glob("/people/pang248/potential_files/2-DFT/3-*/*/")
two_defect = glob.glob("/people/pang248/potential_files/2-DFT/4-*/*/")

p = sorted(pristine+defect_diagonal+defect_planar+two_defect)


for data_path in p:
    dbpath = data_path.replace("/people/pang248/potential_files/",
                          "/qfs/projects/sppsi/spru445/schnet_dbs/full_dataset_precomputed/databases/")[:-1] + ".db"
    csvpath = dbpath.replace('.db','.csv')


    if(os.path.exists(dbpath)):
        os.remove(dbpath)


    dataset = AtomsData(dbpath, available_properties=['energy','forces'])

    atrefs=[[0] for x in range(28)] + [[-0.350]] + [[-0.223]]
    dataset.metadata = {'atref_labels':['energy'], 'atomrefs':atrefs}


    messy_outcars = glob.glob(data_path + "**", recursive = True)
    
    paths = []
    for pa in messy_outcars:
        if '.OUTCAR' in pa and not('.tar' in pa):
            idx = pa.rfind('.')
            paths.append(pa[:idx])

    outcars = []
    xdatcars = []
    # Setup paths for all XDATCAR and OUTCAR files and ensure they're properly associated
    for p in sorted(paths):# sorting to ensure xdatcar matches outcar
        xdatcars.append(p + '.XDATCAR')
        outcars.append(p + '.OUTCAR')
    n_files = len(outcars)

    if(os.path.exists(csvpath)):
        os.remove(csvpath)

    # Loop through all files
    for i in range(n_files):
        print("Adding data from " + outcars[i])
        X = read_vasp_xdatcar(xdatcars[i])
        n_steps = len(X)
        atoms = []
        # Read in data from all steps in file to a list of atoms type
        atoms = read_vasp_out(outcars[i],slice(0,n_steps,1))
        # Add new properties to object
        property_list = []
        for at in atoms:
            energy = at.get_potential_energy()

            forces = at.get_forces()
            theory = 'DFT'
            neighbors, _ = torch_env.get_environment(at)
            property_list.append(
                {'energy': np.array([energy]),
                    'forces': forces}) 

        #Add objects from file to database
        dataset.add_systems(atoms, property_list)