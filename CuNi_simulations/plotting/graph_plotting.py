import os
import os.path as op
import glob

import numpy as np
import pandas as pd

import networkx as nx

from ase.io import read
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap

from time_graph import TimeGraph

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_paths', required=True, nargs="+", type=str, help='full path to best_model')
parser.add_argument('--save_path', required=True, type=str, help='path to save output')
args = parser.parse_args()

fig, ax = plt.subplots(1,1, figsize = (4,3))

plt.sca(ax)

TG = TimeGraph(metric_tolerance = 1e-3)

node_norm = None
edge_norm = None

paths = sorted(args.data_paths)

for i, p in enumerate(paths):
    TG.add_shear_series(p)


draw_ax = plt.gca()


norms = TG.draw(ax=ax, 
            node_norm=node_norm, 
            edge_norm=edge_norm,
            node_size=15,
            edge_size=1.5)
draw_ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)


ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel('$\Delta$ Energy (eV)')
    

plt.savefig(args.save_path, dpi=300)