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
from ase.optimize import BFGS
from ase.io import write, read
from ase.io.trajectory import Trajectory

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher

from scipy.optimize import linear_sum_assignment

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, type=str, help='full path to best_model')
parser.add_argument('--code_path', default='/people/pope044/SPPSi/NN_potential/schnetpack_base/src', type=str, help='base path to schnet model')
parser.add_argument('--data_path', required=True, type=str, help='path to VASP output')
parser.add_argument('--save_path', required=True, type=str, help='path to save output')
parser.add_argument('--cutoff', default=6., type=float, help='NN cutoff distance')
parser.add_argument('--device', default='cuda', type=str, help='device to run on (cuda/cpu)')
args = parser.parse_args()


sys.path.insert(0, args.code_path)


from schnetpack.interfaces import SpkCalculator


from schnetpack.environment import AseEnvironmentProvider as EnvironmentProvider
from schnetpack.utils import load_model


def convert_to_model(model_path, device = 'cuda'): #Takes a list of model paths
    
    model = load_model(model_path)
    calcs = SpkCalculator(model, device=device, energy='energy', forces='forces')
    
    return calcs

def get_data_paths(vasp_path):
    return sorted(glob.glob(vasp_path+'*/'))

def get_CAR_files(data_path):#takes path to one directory containing outcars and xdatcars
    return sorted(glob.glob(data_path+'*OUTCAR')), sorted(glob.glob(data_path+'*XDATCAR'))

def grab_trajectory(data_path):
    outcars, xdatcars = get_CAR_files(data_path)
    traj = []
    base_energies = []
    for i, x in enumerate(xdatcars):
        energy_list = []
        X = read_vasp_xdatcar(x)
        n_steps = len(X)
        traj += read_vasp_out(outcars[i],slice(0,n_steps,1))
    
    base_energies = [t.get_potential_energy() for t in traj]
    base_forces = [t.get_forces() for t in traj]
    return traj, base_energies, base_forces #returns as a trajectory object
        


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


def multiplot_from_generator(g, num_columns, figsize_for_one_row=None):
    # call 'next(g)' to get past the first 'yield'
    next(g)
    # default to 15-inch rows, with square subplots
    if figsize_for_one_row is None:
        figsize_for_one_row = (15, 15/num_columns)
    try:
        while True:
            # call plt.figure once per row
            plt.figure(figsize=figsize_for_one_row)
            for col in range(num_columns):
                ax = plt.subplot(1, num_columns, col+1)
                next(g)
    except StopIteration:
        pass
    
    
    
    
    
    
def shear_validation_static(model_path, vasp_path, device = 'cuda'):
            
    calc = convert_to_model(model_path, device)
    data_paths = get_data_paths(vasp_path)
    
    nn_energy = []
    nn_forces = []
    base_energy = []
    base_forces = []
    shears = []
    
    for i, p in enumerate(data_paths):
        traj, base_energies_chunk, base_forces_chunk = grab_trajectory(p)
        base_energy += base_energies_chunk
        base_forces += base_forces_chunk
        
        
        for ats in traj:
            ats.set_calculator(calc)#Set the calculator to nn potential
            a,b,c = ats.cell

            #Calculate shear
            L = np.linalg.norm(ats.cell[0][1:])
            H = ats.cell[0][0]
            shear = 100*L/H
            shears.append(shear)

            nn_energy.append(ats.get_potential_energy())
            nn_forces.append(ats.get_forces())
        

    return nn_energy, nn_forces, base_energy, base_forces, shears





model_name = args.model_path.split('/')[-1]

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
    
nn_energy, nn_forces, base_energy, base_forces, shears = shear_validation_static(args.model_path, 
                                                                                 args.data_path, 
                                                                                 device=args.device)


data = {'nn_energy':nn_energy,  'base_energy':base_energy, 'shears':shears, 
        'nn_forces':nn_forces, 'base_forces':base_forces}



np.savez(op.join(args.save_path, f'{data_name_fldr}.npz'),
        nn_energy = np.array(data['nn_energy']),
         base_energy=np.array(data['base_energy']),
         nn_forces=np.array(data['nn_forces']),
         base_forces=np.array(data['base_forces']),
         shears=np.array(data['shears'])
        )

# with open(op.join(args.save_path, model_name, f'{data_name_fldr}_{data_name}.csv'),'w') as f:
#     f.write(str(data))  # set of numbers & a tuple