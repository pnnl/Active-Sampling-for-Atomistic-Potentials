import os
import os.path as op
import sys
import glob
import torch

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ase.db import connect
from ase.io.vasp import read_vasp_xdatcar, read_vasp_out
from ase.optimize import LBFGS
from ase.io import write, read
from ase.io.trajectory import Trajectory

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher

from scipy.optimize import linear_sum_assignment

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, type=str, help='full path to best_model')
parser.add_argument('--code_path', default='/qfs/projects/sppsi/spru445/schnetpack_builds/schnetpack_16_fast/src', type=str, help='base path to schnet model')
parser.add_argument('--data_path', required=True, type=str, help='path to VASP output')
parser.add_argument('--save_path', required=True, type=str, help='path to save output')
parser.add_argument('--cutoff', default=6., type=float, help='NN cutoff distance')
parser.add_argument('--device', default='cuda', type=str, help='device to run on (cuda/cpu)')
parser.add_argument('--save_structures', default=False, type=bool, help='whether or not to save final structures')
args = parser.parse_args()


sys.path.insert(0, args.code_path)


from schnetpack.interfaces import SpkCalculator


from schnetpack.environment import AseEnvironmentProvider as EnvironmentProvider
from schnetpack.utils import load_model


def convert_to_model(model_path, cutoff, device='cuda'):    
    model = load_model(model_path)
    calcs = SpkCalculator(model, device=device, energy="energy", forces="forces",  
                          environment_provider=EnvironmentProvider(cutoff=cutoff))
    return calcs

def get_data_paths(vasp_path):
    return sorted(glob.glob(op.join(vasp_path,'*')))

def get_CAR_files(data_path):#takes path to one directory containing outcars and xdatcars
    return sorted(glob.glob(op.join(data_path,'*OUTCAR'))), sorted(glob.glob(op.join(data_path,'*XDATCAR')))

def grab_data(data_path):
    print(data_path)
    outcars, xdatcars = get_CAR_files(data_path)
    

    for i, x in enumerate(xdatcars):
        X = read_vasp_xdatcar(x)
        n_steps = len(X)
        base_traj = read_vasp_out(outcars[i],slice(0,n_steps,1))
        final_atoms = base_traj[-1].copy()
        
        if i==0:
            initial_atoms = base_traj[0].copy() #must copy the atoms object
            
        final_energy = base_traj[-1].get_potential_energy()
    
    return initial_atoms, final_atoms, final_energy
        


def extract_info(vasp_path): #Takes a singular path
    X = read_vasp_xdatcar(vasp_path+".XDATCAR")
    n_steps = len(X)
    base_traj = read_vasp_out(vasp_path+".OUTCAR",slice(0,n_steps,1))
    
    return base_traj
        

def calc_rmsd(path_A, path_B): 
    return float(os.popen("calculate_rmsd --reorder " + path_A + " " + path_B).read())



def write_xyz(at, filename=''):
    coords = [' '.join([str(i) for i in x])+'\n' for x in at.get_positions()]
    symbs  = at.get_chemical_symbols()
    
    with open(filename,'a') as f:
        f.writelines([str(len(symbs))+'\n']+[str(at.get_potential_energy())+'\n']+
                     [' '.join(x) for x in list(zip(symbs,coords))])

    
    
    
    
    
def shear_validation_ase(model_path, vasp_path, device = 'cuda', keep_traj = False, fmax = 0.02, save_structures = None, cutoff=6.0):
    
    calc = convert_to_model(model_path, cutoff, device=device)
    data_paths = get_data_paths(vasp_path)
    
    nn_energy = []
    base_energy = []
    shears = []
    structure_differences = []
    #difference_distributions = []
    
    for i, p in enumerate(data_paths):
        initial_atoms, final_atoms, final_energy = grab_data(p)
        base_energy.append(final_energy)
        
        final_atoms.set_calculator(calc) #Set calculator because you need to for some reason?
        
        if not op.isdir(op.join(save_structures, 'nn-traj','')):
            os.mkdir(op.join(save_structures, 'nn-traj',''))
        
        shear_displacement = initial_atoms.cell[0,1]
        traj_path = op.join(save_structures,'nn-traj',str(shear_displacement)+'.traj')
        
        #Set up the atoms object for the calculation
        if i == 0:
            ats = initial_atoms.copy()
            ats.set_calculator(calc)#Set the calculator to nn potential
            a,b,c = ats.cell
                        
        
        #perturb cell
        a = (a[0], shear_displacement, shear_displacement)
        ats.set_cell([a, b, c], scale_atoms=True)
        print(ats.cell)
        print(ats.pbc)
        
        #Calculate shear
        L = np.linalg.norm(ats.cell[0][1:])
        H = ats.cell[0][0]
        shear = 100*L/H
        shears.append(shear)
        
        opt = LBFGS(ats, trajectory=traj_path)
        opt.run(fmax=float(fmax))
        
        traj = Trajectory(traj_path, 'r')
        nn_energy.append(traj[-1].get_potential_energy())
        
        
        #Compare final structures
        #ats_nn = traj[-1].copy()
        #ats_base = final_atoms.copy()
        
        #write('./structures.xyz',[traj[-1].copy(),final_atoms.copy()])
        #write('./dft_structure.xyz',final_atoms.copy())
        
        big_ats = traj[-1].copy()
        big_ats.extend(final_atoms.copy())
        
        system_size = int(len(big_ats.numbers) / 2)
        #print(system_size)
        
        D = big_ats.get_all_distances(mic=True)
        
        d = D[:int(D.shape[0]/2), int(D.shape[0]/2):]
        #print(d.shape)
        
        row, col = linear_sum_assignment(d)
        
        diffs = d[row,col]
        #print(diffs)
        
        #difference_distributions.append(diffs.tolist())
        
        rmsd = np.sqrt(np.mean(diffs**2))
        stdv_rmsd = np.std(diffs)
            
        structure_differences.append(rmsd)
        
        print(str(shear) +': ' + str(structure_differences[-1]))
        print('stdv: ' + str(stdv_rmsd))

        print('\n--------------------------------------\n')
        
        if not keep_traj:
            os.remove(traj_path)
           
        if save_structures:
            if not op.isdir(op.join(save_structures, 'dft-xyz', '')):
                os.mkdir(op.join(save_structures, 'dft-xyz', ''))
            if not op.isdir(op.join(save_structures, 'nn-xyz', '')):
                os.mkdir(op.join(save_structures, 'nn-xyz', ''))
                
            
            write( op.join( save_structures, 'dft-xyz', str(shear_displacement)+'.xyz' ),final_atoms.copy())
            write( op.join( save_structures, 'nn-xyz', str(shear_displacement)+'.xyz' ),traj[-1].copy())
            

                
        
    return nn_energy, base_energy, shears, structure_differences





model_name = args.model_path.split('/')[-2]

if not op.isdir(args.save_path):
    os.mkdir(args.save_path)





if 'pristine' in args.data_path:
    data_name_fldr = 'pristine'
elif 'N1' in args.data_path:
    data_name_fldr = 'N1'

if 'new' in args.data_path:
    data_name_fldr += '-new'
elif 'old' in args.data_path:
    data_name_fldr += '-old'


save_structures = op.join(args.save_path, f'{data_name_fldr}','')

if not op.isdir(save_structures):
    os.mkdir(save_structures)


nn_energy, base_energy, shears, structure_differences = shear_validation_ase(args.model_path, 
                                                                               args.data_path, 
                                                                               device=args.device,
                                                                               keep_traj=True,
                                                                               fmax = 0.005,
                                                                               save_structures=save_structures)

data = {'nn_energy':nn_energy,  'base_energy':base_energy, 'shears':shears, 
        'structure_differences':structure_differences}

df = pd.DataFrame.from_dict(data)

df.to_csv(op.join(args.save_path, f'{data_name_fldr}.csv'))

# with open(op.join(args.save_path, model_name, f'{data_name_fldr}.txt'),'w') as f:
#     f.write(str(data))  # set of numbers & a tuple
