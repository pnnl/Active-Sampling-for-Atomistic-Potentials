import sys
sys.path.insert(0,'../schnetpack/src/')

import math
import logging
import numpy as np
import torch
from ase.neighborlist import neighbor_list
import itertools
from collections import Counter
import os.path as op
import os
from ase.db import connect
from schnetpack import Properties

flatten = lambda x: [item for sublist in x for item in sublist]

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='Path .db with data')
parser.add_argument('--preprocess_path', type=str, help='Directory to save preprocess data')
parser.add_argument('--cutoff', type=float, default=6.0, help='Cutoff value for nearest neighbor collection')
parser.add_argument('--no_mic', action='store_false', default=6.0, help='Cutoff value for nearest neighbor collection')
parser.add_argument()
args = parser.parse_args()
    

def torchify_dict(data):
    """
    Transform np.ndarrays to torch.tensors.

    """
    torch_properties = {}
    for pname, prop in data.items():

        if prop.dtype in [np.int, np.int32, np.int64]:
            torch_properties[pname] = torch.LongTensor(prop)
        elif prop.dtype in [np.float, np.float32, np.float64]:
            torch_properties[pname] = torch.FloatTensor(prop.copy())
        else:
            print(
                "Invalid datatype {} for property {}!".format(type(prop), pname)
            )

    return torch_properties


def collate_input(atoms, data, inputs={}):

    # Elemental composition
    inputs[Properties.Z] = atoms.numbers.astype(np.int)
    positions = atoms.positions.astype(np.float32) 
    # center
    #positions -= get_center_of_mass(atoms)
    inputs[Properties.R] = positions

    # get atom environment
    nbh_idx = data['neighbors']
    offsets = np.zeros((1), dtype=np.float32)

    inputs[Properties.neighbors] = nbh_idx.astype(np.int)
    
    # apply neighbor masks
    mask = inputs[Properties.neighbors] >= 0
    inputs[Properties.neighbor_mask] = mask.astype(np.float)
    inputs[Properties.neighbors] = (
        inputs[Properties.neighbors] * inputs[Properties.neighbor_mask].astype(np.int)
    )

    # Get cells
    inputs[Properties.cell] = np.array(atoms.cell.array, dtype=np.float32)
    inputs[Properties.cell_offset] = offsets.astype(np.float32)
    
    return inputs


db=connect(args.data_path) 

def prepro(i):
    row = db.get(i+1)
    energy = np.array(row.data['energy'], dtype=np.float32)
    forces = np.array(row.data['forces'], dtype=np.float32)
    
    
    atoms = row.toatoms()
    n_atoms = len(atoms)
    distance_matrix = atoms.get_all_distances(not args.no_mic)
    distances = np.nonzero(np.where(distance_matrix <= args.cutoff, distance_matrix, 0))
    neighborhood_idx = [list(distances[1][np.argwhere(distances[0] == i).flatten()]) for i in range(n_atoms)]
    n_max_nbh = np.max([len(x) for x in neighborhood_idx])
    neighbors = np.array([np.pad(x, (0,n_max_nbh-len(x)), mode='constant', constant_values=-1) for x in neighborhood_idx], dtype=np.int)

    
    data = {'neighbors': neighbors}
    input_data = torchify_dict(collate_input(atoms, data, inputs={'energy': energy, 'forces': forces}))
    
    torch.save(input_data, op.join(args.preprocess_path, f'{i}.pt'))
    
    
from multiprocessing import Pool

if __name__ == '__main__':
    with Pool(8) as p:
        p.map(prepro, list(range(0,len(db))))
        
        