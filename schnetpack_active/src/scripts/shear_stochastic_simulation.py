## Imports ##
import os
import os.path as op
import sys
import glob
import torch
from datetime import datetime
import numpy as np
import pandas as pd
import json
from ase.db import connect
from ase.io.vasp import read_vasp_xdatcar, read_vasp_out
from ase.optimize import LBFGS
from ase.io import write, read
from ase.io.trajectory import Trajectory
import networkx as nx
from ovito.io import import_file
from ovito.modifiers import PolyhedralTemplateMatchingModifier
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import signal

class TimeoutError(Exception):
    pass

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

## Args ##
parser = argparse.ArgumentParser()

# paths
parser.add_argument('--data', required=True, type=str, help='path to dir with data')
parser.add_argument('--model', required=True, type=str, help='path to trained schnet model')
parser.add_argument('--save_path', required=True, type=str, help='path to save output')
parser.add_argument('--run_name', default='test_run', type=str, help='unique identifier for the run')

# sampling parameters
parser.add_argument('--rattle', type=int, default=10, help='Number of rattle samples')
parser.add_argument('--stdev', type=float, default=0.01, help='Standard deviation for rattle')
parser.add_argument('--base', action='store_true', help='Flag to optimize structure with no rattle')
parser.add_argument('--starting_iteration', type=int, default=0, help='Integer to denote first index of rattle paths. Default 0')

# optimization parameters
parser.add_argument('--num_steps', type=int, default=30, help='Number of steps to take')
parser.add_argument('--shear_vector', type=str, default='011', help='Shear vector to use')
parser.add_argument('--fmax', type=float, default=0.001, help='fmax for optimization convergence')
parser.add_argument('--device', choices=['cuda','cpu'], default='cuda', help='Device to run NNP')
parser.add_argument('--keep_traj', action='store_true', help='Flag to store optimization trajectories')
args = parser.parse_args()


## SchNet Import ##

sys.path.insert(0, '../')
from schnetpack.interfaces import SpkCalculator
from schnetpack.environment import AseEnvironmentProvider as EnvironmentProvider
from schnetpack.utils import load_model


## Functions ##
class train_args:
    """Read train_args from json"""
    def __init__(self, save_dir):
        dictionary = json.load(open(op.join(save_dir,'args.json')))
        for k, v in dictionary.items():
            setattr(self, k, v)

def convert_to_model(model_path, cutoff, device='cuda'):    
    model = load_model(model_path)
    calcs = SpkCalculator(model, device=device, energy="energy", forces="forces",  
                          environment_provider=EnvironmentProvider(cutoff=cutoff))
    return calcs

def get_shear_tensor(name):
    if name == '011':
        return np.array([[0.,0.1,0.1],[0.,0.,0.],[0.,0.,0.]])
    if name == '101':
        return np.array([[0.,0.,0.],[0.1,0.,0.1],[0.,0.,0.]])
    if name == '110':
        return np.array([[0.,0.,0.],[0.1,0.,0.1],[0.1,0.1,0.]])
    else:
        raise(ValueError(f'Unkown shear vector name {name}'))
        
def get_data_paths(vasp_path):
    if len(sorted(glob.glob(op.join(vasp_path,'*.OUTCAR')))) != 0:
        return [vasp_path]
    return sorted(glob.glob(op.join(vasp_path,'*')))

def get_CAR_files(data_path):#takes path to one directory containing outcars and xdatcars
    return sorted(glob.glob(op.join(data_path,'*OUTCAR'))), sorted(glob.glob(op.join(data_path,'*XDATCAR')))

def grab_data(data_path):
    outcars, xdatcars = get_CAR_files(data_path)
    
    for i, x in enumerate(xdatcars):
        X = read_vasp_xdatcar(x)
        n_steps = len(X)
        base_traj = read_vasp_out(outcars[i],slice(0,n_steps,1))
        final_atoms = base_traj[-1].copy()
        
        if i==0:
            initial_atoms = base_traj[0].copy() 
            
        final_energy = base_traj[-1].get_potential_energy()
    
    return initial_atoms, final_atoms, final_energy



##Strcuture Metrics##
def generate_graph(atoms, cutoff=3.1):
    nnA = np.where(atoms.get_all_distances(mic=True)<=cutoff, atoms.get_all_distances(mic=True), 0)
    nnG = nx.from_numpy_matrix(nnA)
    nx.set_node_attributes(nnG, dict(zip(range(len(atoms)),atoms.get_atomic_numbers())), "atomic_number")
    nx.set_node_attributes(nnG, dict(zip(range(len(atoms)),atoms.get_positions())), "position")
    
    # set shear as graph attribute
    cell = atoms.cell
    L = np.linalg.norm(cell[0][1:])
    H = cell[0][0]
    shear = 100*L/H
    
    nnG.graph['shear'] = shear
    
    return nnG

#Structure encoding for OVITO polyhedral template matching
def structure_encoding(row):
    structure_key={0: "other", 1: "FCC", 2: "HCP", 3: "BCC", 4: "ICO", 
               5: "SC", 6: "cubic diamond", 7: "hexagonal diamond", 8: "graphene"}
    return structure_key[row.structure_type]


def get_structure_data(xyz_path, ideal_neighbors=12):
    #structure types
    #OVITO Read-in
    pipeline = import_file(xyz_path)
    pipeline.modifiers.append(PolyhedralTemplateMatchingModifier())
    data = pipeline.compute()
    
    d={'idx': list(range(data.particles.count)),
           'atom_type': list(data.particles['Particle Type']),
           'structure_type': list(data.particles['Structure Type'])}
    df=pd.DataFrame(d)
    
    #Intermixing
    #ASE Read in
    atoms = read(xyz_path)
    G = generate_graph(atoms)

    # atom type bonding
    bond_matrix = nx.attr_matrix(G, node_attr='atomic_number', rc_order=[28, 29])
    n_NiNi = int(bond_matrix[0,0])
    n_CuCu = int(bond_matrix[1,1])
    n_NiCu = int(bond_matrix[0,1])

    # choose one of the two below
    intermix = 100*(n_NiCu)/(n_NiCu+n_NiNi+n_CuCu)
    df_translated = df.apply(lambda row: structure_encoding(row), axis=1)
    structure_counts = df_translated.value_counts().to_dict()
    
    for label in ["other", "FCC", "HCP", "BCC", "ICO", "SC", "cubic diamond", "hexagonal diamond", "graphene"]:
        if label not in structure_counts.keys():
            structure_counts[label] = 0
    
        
    return structure_counts, intermix
    
        


def shear_deformation(model_path, 
                         ats,
                         cutoff,
                         device = 'cuda', 
                         keep_traj = False, 
                         fmax = 0.002, 
                         save_structures = None,
                         rattle = False,
                         stdev = 0.001,
                         seed = 42,
                         shear_displacement = np.array([[0.,0.1,0.1],[0.,0.,0.],[0.,0.,0.]]),
                         num_steps = 30
                        ):
    
    calc = convert_to_model(model_path, cutoff, device=device)
    
    nn_energy = []
    shears = []
    nn_other = []
    nn_fcc = []
    nn_hcp = []
    nn_bcc = []
    nn_ico = []
    nn_sc = []
    nn_cubic_diamond = []
    nn_hexagonal_diamond = []
    nn_graphene = []
    nn_intermix = []
    
    ats = ats.copy()
    ats.set_calculator(calc)
    
    if not op.isdir(op.join(save_structures)):
        if save_structures:
            os.mkdir(op.join(save_structures))
    if not op.isdir(op.join(save_structures, 'nn-traj')):
        if save_structures and keep_traj:
            os.mkdir(op.join(save_structures, 'nn-traj'))
    
    magnitude_steps = int(np.log10(num_steps)) + 1
    shear_step=0
    for i in range(num_steps):
        try:
            # set up paths for trajectory
            if keep_traj:
                traj_path = op.join(save_structures,'nn-traj',str(shear_step).zfill(magnitude_steps)+'.traj')
            else:
                now = str(datetime.now())
                clean_time = '-'.join(now.split(' '))
                traj_path = clean_time + '-run.traj'


            #Calculate shear
            L = np.linalg.norm(ats.cell[0][1:])
            H = ats.cell[0][0]
            shear = 100*L/H
            
            # Apply jitter 
            if rattle:
                ats.rattle(stdev=stdev, seed=seed)

            # Relax the structure using the NNP
            opt = LBFGS(ats, trajectory=traj_path)
            
            with timeout(seconds=2700):
                opt.run(fmax=float(fmax))
                
            traj = Trajectory(traj_path, 'r')
            
            if save_structures:
                if not op.isdir(op.join(save_structures, 'nn-xyz', '')):
                    os.mkdir(op.join(save_structures, 'nn-xyz', ''))
                xyz_path = op.join(save_structures, 'nn-xyz', str(shear_step).zfill(magnitude_steps)+'.xyz' )
                traj[-1].write(op.join(save_structures, 'nn-xyz', str(shear_step).zfill(magnitude_steps)+'.xyz' ))
            else:
                now = str(datetime.now())
                clean_time = '-'.join(now.split(' '))
                xyz_path = op.join('.', clean_time, str(shear_step).zfill(magnitude_steps)+'.xyz' )
                traj[-1].write(op.join(save_structures, 'nn-xyz', str(shear_step).zfill(magnitude_steps)+'.xyz' ))

            #Compare final structures        
            # Percent FCC 
            structure_values, nn_intermix_value = get_structure_data(xyz_path)
            
            if not save_structures:
                os.remove(xyz_path)

            # Append all to lists
            shears.append(shear)
            nn_energy.append(traj[-1].get_potential_energy())
            nn_other.append(structure_values['other'])
            nn_fcc.append(structure_values['FCC'])
            nn_hcp.append(structure_values['HCP'])
            nn_bcc.append(structure_values['BCC'])
            nn_ico.append(structure_values['ICO'])
            nn_sc.append(structure_values['SC'])
            nn_cubic_diamond.append(structure_values['cubic diamond'])
            nn_hexagonal_diamond.append(structure_values['hexagonal diamond'])
            nn_graphene.append(structure_values['graphene'])
            nn_intermix.append(nn_intermix_value)

            print(shear)

            print('\n--------------------------------------\n')

            if not keep_traj:
                os.remove(traj_path)

            if save_structures:
                if not op.isdir(op.join(save_structures, 'nn-xyz', '')):
                    os.mkdir(op.join(save_structures, 'nn-xyz', ''))
                    
                traj[-1].write(op.join(save_structures, 'nn-xyz', str(shear_step).zfill(magnitude_steps)+'.xyz' ))

            # Apply shear by perturbing the cell
            ats.set_cell(ats.cell + shear_displacement, scale_atoms=True)
            shear_step += 1

        except TimeoutError:
            data = {'nn_energy':nn_energy,
                'shears':shears, 
                'nn_other':nn_other, 
                'nn_fcc':nn_fcc,
                'nn_hcp':nn_hcp,
                'nn_bcc':nn_bcc,
                'nn_ico':nn_ico,
                'nn_sc':nn_sc,
                'nn_cubic_diamond':nn_cubic_diamond,
                'nn_hexagonal_diamond':nn_hexagonal_diamond,
                'nn_graphene':nn_graphene,
                'nn_intermix':nn_intermix,
                'complete':0}
            df = pd.DataFrame.from_dict(data)
            return df
    
        except BaseException as err:
            if not keep_traj:
                if os.path.exists(traj_path):
                    os.remove(traj_path)
            raise(err)
        
    data = {'nn_energy':nn_energy,
        'shears':shears, 
        'nn_other':nn_other, 
        'nn_fcc':nn_fcc,
        'nn_hcp':nn_hcp,
        'nn_bcc':nn_bcc,
        'nn_ico':nn_ico,
        'nn_sc':nn_sc,
        'nn_cubic_diamond':nn_cubic_diamond,
        'nn_hexagonal_diamond':nn_hexagonal_diamond,
        'nn_graphene':nn_graphene,
        'nn_intermix':nn_intermix,
        'complete':1}
    df = pd.DataFrame.from_dict(data)
    
    return df

# extract name of model 
model_name = args.model.split('/')[-2]

# read in training args
train_args = train_args('/'.join(args.model.split('/')[:-1]))

## prepare save path 
if not op.isdir(args.save_path):
    os.mkdir(args.save_path)

# make a folder for the final structures and/or traj files
save_structures_path = op.join(args.save_path, f'{args.run_name}')

#get shear_displacement
shear_displacement = get_shear_tensor(args.shear_vector)


if not op.isdir(save_structures_path):
    os.mkdir(save_structures_path)
        
        
if '.xyz' in args.data:
    ats = read(args.data)
else:
    try:
        data_paths = get_data_paths(args.data)
        ats, _, _ = grab_data(data_paths[0])
    except BaseException as err:
        raise(f'Something is wrong with the data path given. Err: {err}')

    
# direct comparison against DFT
if args.base:
    df = shear_deformation(args.model, 
                             ats, 
                             train_args.cutoff,
                             device=args.device,
                             keep_traj = args.keep_traj,
                             fmax = args.fmax,
                             save_structures=op.join(save_structures_path, 'base'),
                             shear_displacement=shear_displacement,
                             rattle=False,
                             num_steps = args.num_steps)

    df.to_csv(op.join(args.save_path, f'{args.run_name}.csv'), index=False)


# incorporate slight perturbation in the starting positions
for i in range(0,args.rattle):
    df = shear_deformation(args.model,
                             ats, 
                             train_args.cutoff,
                             device=args.device,
                             keep_traj = args.keep_traj,
                             fmax = args.fmax,
                             save_structures=op.join(save_structures_path, f'rattle{str(args.starting_iteration+i).zfill(2)}-{args.stdev}'),
                             shear_displacement=shear_displacement,
                             rattle=True,
                             stdev=args.stdev,
                             seed=i,
                             num_steps = args.num_steps)

    df.to_csv(op.join(args.save_path, f'{args.run_name}-rattle{str(args.starting_iteration+i).zfill(2)}-{args.stdev}.csv'), index=False)
