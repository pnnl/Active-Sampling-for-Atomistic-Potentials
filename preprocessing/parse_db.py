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
parser.add_argument('--device', type=str, default='cpu', help='Device on which to preform preprocessing')
args = parser.parse_args()

device = 'cuda'
torch_env = TorchEnvironmentProvider(cutoff = args.cutoff, device = args.device)



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

    with open(csvpath,'a') as f:
        i = 'id'
        e = 'energy'
        t = 'theory'
        s = 'shear_val'
        sv = 'shear_vector'
        sd = 'shear_direction'
        va = 'vacancy'
        f.writelines(f'{i},{e},{t},{s},{sv},{sd},{va}\n')

    w = 1

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
            
            #Calculate shear
            if 'a3' in outcars[i]:
                sv = 'a3'
                if '100' in outcars[i]:
                    sd = '100'
                    shear = at.cell[2][0] / at.cell[0][0] * 100
                elif '010' in outcars[i]:
                    sd = '010'
                    shear = at.cell[2][1] / at.cell[0][0] * 100
                else: #Assume 110 direction
                    sd = '110'
                    shear = np.sqrt(at.cell[2][0]**2 + at.cell[2][1]**2) / at.cell[2][2] * 100
            elif 'a2' in outcars[i]:
                sv = 'a2'
                if '100' in outcars[i]:
                    sd = '100'
                    shear = at.cell[1][0] / at.cell[0][0] * 100
                elif '001' in outcars[i]:
                    sd = '001'
                    shear = at.cell[1][2] / at.cell[0][0] * 100
                else:#assume 101 direction
                    sd = '101'
                    shear = np.sqrt(at.cell[1][0]**2 + at.cell[1][1]**2) / at.cell[1][1] * 100
            else:#Assume a1
                sv = 'a1'
                if '010' in outcars[i]:
                    sd = '010'
                    shear = at.cell[0][1] / at.cell[0][0] * 100
                elif '001' in outcars[i]:
                    sd = '001'
                    shear = at.cell[0][2] / at.cell[0][0] * 100
                else:#assume 011 direction
                    sd = '011'
                    shear = np.sqrt(at.cell[0][1]**2 + at.cell[0][2]**2) / at.cell[0][0] * 100
            
            #determine the type of vacancy
            if 'pristine' in outcars[i]:
                va = 'pristine'
            elif 'N1' in outcars[i]:
                va = 'N1'
            elif 'N2' in outcars[i]:
                va = 'N2'
            elif 'N3' in outcars[i]:
                va = 'N3'
            elif 'C1' in outcars[i]:
                va = 'C1'
            elif 'C2' in outcars[i]:
                va = 'C2'    
            elif 'C3' in outcars[i]:
                va = 'C3' 
            elif '2va' in outcars[i]:
                va = '2va'

            forces = at.get_forces()
            theory = 'DFT'
            neighbors, _ = torch_env.get_environment(at)
            property_list.append(
                {'energy': np.array([energy]),
                    'forces': forces})
            with open(csvpath,'a') as f:            
                f.writelines(f'{w},{energy},{theory},{shear},{sv},{sd},{va}\n')
            w+=1 

        #Add objects from file to database
        dataset.add_systems(atoms, property_list)
        

        
        
        
#### Create AIMD databases

p = sorted(glob.glob("/people/pang248/potential_files/1-AIMD/*/"))


for data_path in p:
    dbpath = data_path.replace("/people/pang248/potential_files/",
                          "/qfs/projects/sppsi/spru445/schnet_dbs/full_dataset_precomputed/databases/")[:-1] + ".db"
    csvpath = dbpath.replace('.db','.csv')

    print(dbpath)

    if(os.path.exists(dbpath)):
        os.remove(dbpath)


    dataset = AtomsData(dbpath, available_properties=['energy','forces'])

    atrefs=[[0] for x in range(28)] + [[-0.350]] + [[-0.223]]
    dataset.metadata = {'atref_labels':['energy'], 'atomrefs':atrefs}


    messy_outcars = glob.glob(data_path + "**", recursive = True)
    
    paths = []
    for pa in messy_outcars:
        if 'OUTCAR' in pa and not('.tar' in pa):
            idx = pa.rfind('O')
            paths.append(pa[:idx])

    outcars = []
    xdatcars = []
    # Setup paths for all XDATCAR and OUTCAR files and ensure they're properly associated
    for p in sorted(paths):# sorting to ensure xdatcar matches outcar
        xdatcars.append(p + 'XDATCAR')
        outcars.append(p + 'OUTCAR')
    n_files = len(outcars)

    if(os.path.exists(csvpath)):
        os.remove(csvpath)

    with open(csvpath,'a') as f:
        i = 'id'
        e = 'energy'
        t = 'theory'
        s = 'shear_val'
        sv = 'shear_vector'
        sd = 'shear_direction'
        va = 'vacancy'
        f.writelines(f'{i},{e},{t},{s},{sv},{sd},{va}\n')

    w = 1

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
            
            #Calculate shear
            if 'a3' in outcars[i]:
                sv = 'a3'
                if '100' in outcars[i]:
                    sd = '100'
                    shear = at.cell[2][0] / at.cell[0][0] * 100
                elif '010' in outcars[i]:
                    sd = '010'
                    shear = at.cell[2][1] / at.cell[0][0] * 100
                else: #Assume 110 direction
                    sd = '110'
                    shear = np.sqrt(at.cell[2][0]**2 + at.cell[2][1]**2) / at.cell[2][2] * 100
            elif 'a2' in outcars[i]:
                sv = 'a2'
                if '100' in outcars[i]:
                    sd = '100'
                    shear = at.cell[1][0] / at.cell[0][0] * 100
                elif '001' in outcars[i]:
                    sd = '001'
                    shear = at.cell[1][2] / at.cell[0][0] * 100
                else:#assume 101 direction
                    sd = '101'
                    shear = np.sqrt(at.cell[1][0]**2 + at.cell[1][1]**2) / at.cell[1][1] * 100
            else:#Assume a1
                sv = 'a1'
                if '010' in outcars[i]:
                    sd = '010'
                    shear = at.cell[0][1] / at.cell[0][0] * 100
                elif '001' in outcars[i]:
                    sd = '001'
                    shear = at.cell[0][2] / at.cell[0][0] * 100
                else:#assume 011 direction
                    sd = '011'
                    shear = np.sqrt(at.cell[0][1]**2 + at.cell[0][2]**2) / at.cell[0][0] * 100
            
            #determine the type of vacancy
            if 'pristine' in outcars[i]:
                va = 'pristine'
            elif 'N1' in outcars[i]:
                va = 'N1'
            elif 'N2' in outcars[i]:
                va = 'N2'
            elif 'N3' in outcars[i]:
                va = 'N3'
            elif 'C1' in outcars[i]:
                va = 'C1'
            elif 'C2' in outcars[i]:
                va = 'C2'    
            elif 'C3' in outcars[i]:
                va = 'C3' 
            elif '2va' in outcars[i]:
                va = '2va'
            
            
            forces = at.get_forces()
            theory = 'AIMD'
            neighbors, _ = torch_env.get_environment(at)
            property_list.append(
                {'energy': np.array([energy]),
                    'forces': forces})
            with open(csvpath,'a') as f:            
                f.writelines(f'{w},{energy},{theory},{shear},{sv},{sd},{va}\n')
            w+=1 

        #Add objects from file to database
        dataset.add_systems(atoms, property_list)
        
