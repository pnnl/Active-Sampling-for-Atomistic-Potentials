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

def calculate_rmsd(ats1, ats2, permute = True):
    big_ats = ats1.copy()
    big_ats.extend(ats2.copy())
    
    system_size = int(len(big_ats.numbers) / 2)
    
    D = big_ats.get_all_distances(mic=True)
        
    d = D[:int(D.shape[0]/2), int(D.shape[0]/2):]
    
    row, col = linear_sum_assignment(d)
        
    diffs = d[row,col]
    
    return (np.sqrt(np.mean(diffs**2)), np.std(diffs))

_all_structures = ['nn_other', 'nn_fcc', 'nn_hcp', 'nn_bcc',
       'nn_ico', 'nn_sc', 'nn_cubic_diamond', 'nn_hexagonal_diamond',
       'nn_graphene']
        
class TimeGraph:
    def __init__(self, metric='nn_energy',
                 metric_tolerance = 1e-5,
                 time_attribute = 'shears'):
        self.G = nx.DiGraph()
        self.metric_type = metric#string value which determines type of metric to use
        self.metric_tolerance = metric_tolerance
        self.time_attribute = time_attribute
    
        
    def attribute_metric(self, node, data):
        reference_data = data[self.metric_type]
        return abs( node[1] - reference_data ) <= self.metric_tolerance
    
    
    def rmsd_metric(self, time_idx, node, data):
        assert(data is str)
        assert(self._call_data(node) is list and self._call_data(node)[time_idx] is str)
        reference_data = self._call_data(node)[time_idx]
        if '.xyz' not in data:
            #convert the path to xyz
            pass
        if '.xyz' not in reference_data:
            #convert the path to xyz
            pass
        return calculate_rmsd(read(data), read(reference_data))[0] < self.metric_tolerance
    
    
    def metric(self, node, data):
        if self.metric_type in data.keys():
            return self.attribute_metric(node, data)
        elif self.metric_type == 'rmsd': #do RMSD type
            raise(NotImplementedError(f'The rmsd metric type is not implemented'))
            #return self.rmsd_metric(node, data)
        else:
            raise(NotImplementedError(f'Metric type {self.metric_type} not supported for {data}'))
            
            
    
    def create_node(self, data, rattle=None, num_atoms=None):
        time, metric_val = (data[self.time_attribute], data[self.metric_type])
        
        attr = dict()
        for k, v in data.items():
            if k != self.time_attribute:
                attr[k] = [v]
                if k in _all_structures and (num_atoms is not None or num_atoms !=0):
                    attr[k+'_percent']= [data[k]/num_atoms]
        if rattle is not None:
            attr['rattle'] = [rattle]
        #nx.set_node_attributes(self.G, {(time, metric_val): attr})
        (self.G).add_nodes_from([ ((time, metric_val), attr) ])
        
        return (time, metric_val)
    
    
    def add_shear_series(self, tseries):
        if type(tseries) == str:
            rattle = tseries.split('rattle')[-1].split('-')[0]
            tseries = pd.read_csv(tseries)
        else:
            rattle = None
        
        current_node = None
        num_atoms = 0
        for structure_key in _all_structures:
            num_atoms += tseries.iloc[0][structure_key]
            
        for i, t in enumerate(tseries[self.time_attribute]):
            data = tseries.iloc[i]
            found = False
            for node in [n for n in self.G.nodes if n[0]==t]:
                if self.metric(node, data):
                    for k in set(self.G.nodes()[node].keys()).intersection(set(data.keys())):
                        self.G.nodes()[node][k].append(data[k])
                        if k in _all_structures:
                            self.G.nodes()[node][k+'_percent'].append(data[k]/num_atoms)
                    if rattle is not None:
                        self.G.nodes()[node]['rattle'].append(rattle)
                    found = True
                    break

            if not found:
                node = self.create_node(data, rattle, num_atoms=num_atoms)

            if current_node is not None:

                if self.G.has_edge(current_node, node):
                    self.G[current_node][node]['weight'] += 1
                else:
                    self.G.add_edge(current_node, node,
                                    weight=1)
            current_node = node
                        
                                    
    def plot(self, value_key, ax = None, fig=None):
        if fig is not None:
            plt.figure(fig)
        if value_key in _all_structures:
            total_atoms = 0
        else:
            total_atoms = None
        nodes_list = self.G.nodes
        for n in nodes_list:
            if total_atoms == 0:
                for k in _all_structures:
                    total_atoms += self.G.nodes()[n][k][0]
            for s in self.G.successors(n):
                #edge_data = self.G.get_edge_data(n, s) TODO: Get color and frequency data from here
                shears = [n[0], s[0]]
                values = np.array([nodes_list[n][value_key][0], nodes_list[s][value_key][0]])
                if ax is None:
                    plt.plot(shears, values, '-x', alpha = 0.6)
                else:
                    if value_key in _all_structures:
                        ax.plot(shears, values/total_atoms*100, 'x-', alpha=1, color='#7f7f7f', markersize=4)
                    else:
                        ax.plot(shears,values, 'x-', alpha=1, color='#7f7f7f', markersize=4)
                    
    def get_rattles_idx(self, idx):
        time = self.times()[idx]
        return self.get_rattles_shear(time)
    
    
    def get_rattles_shear(self, time):
        all_times = self.times()
        idx = np.argmin(np.abs([t - time for t in all_times]))
        closest_time = all_times[idx]
        print(f'closest_time={closest_time}, idx:{idx}')
        filt = lambda tup: abs(tup[0] - closest_time) < 0.00001
        nodes = list(filter(filt, self.G.nodes))
        for n in reversed(sorted(nodes, key=lambda tup:tup[1])):
            rat = self.G.nodes()[n]['rattle']
            print(f'node: {n}, rattle: {rat}')
    
    def get_nodes_idx(self, idx):
        time = self.times()[idx]
        return self.get_nodes_shear(time)
    
    def get_nodes_shear(self, time):
        all_times = self.times()
        closest_time = all_times[np.argmin(np.abs([t - time for t in all_times]))]
        filt = lambda tup: tup[0] == time
        return list(filter(filt, self.G.nodes))
        
    
    
    def times(self):
        return list(set([tup[0] for tup in self.G.nodes]))
        

                
    def draw(self,
             ax=None,
             fig=None,
             edge_norm=None,
             node_norm=None,
             cb=False,
             node_size=25,
             edge_size=2,
             intermix_name='nn_intermix',
             fcc_name='nn_fcc_percent'):
        min_energy = self.get_nodes_idx(0)[0][1]
        nodes_list = self.G.nodes
        pos = dict()
        y_number = 0
        previous_shear = None
        for t in self.times():
            nodes_list = self.get_nodes_shear(t)
            for i, n in enumerate(sorted(nodes_list, key = lambda tup: tup[-1])):
                pos[n] = (n[0]/100,n[1]-min_energy)
        
        #edge_colors = [int(self.G.get_edge_data(u, v)['weight'])/10 for u, v in self.G.edges()]
        edge_colors = [self.G.nodes[v][intermix_name][0] - self.G.nodes[u][intermix_name][0] for u, v in self.G.edges()]
        edges = [(u,v) for u, v in self.G.edges()]
        #edge_colors = [c/max([abs(val) for val in edge_colors]) for c in edge_colors]
        
        node_colors = [self.G.nodes[n][fcc_name][0] for n in self.G.nodes()]
        
        
        edge_cmap = plt.get_cmap('coolwarm')
        a=0.75
        my_cmap = plt.cm.coolwarm(np.arange(plt.cm.coolwarm.N))
        my_cmap[:,0:3] *= a 
        edge_cmap = ListedColormap(my_cmap)
        if edge_norm is None:
            edge_vmin = -1*np.abs(np.array(edge_colors)).max()
            edge_vmax = 1*np.abs(np.array(edge_colors)).max()
            edge_norm = matplotlib.colors.Normalize(vmin=edge_vmin,
                                            vmax=edge_vmax)
            
            print(f'edge_norm: vmin={edge_vmin}, vmax={edge_vmax}')
        else:
            edge_vmin = edge_norm.vmin
            edge_vmax = edge_norm.vmax
            edge_norm = matplotlib.colors.Normalize(vmin=edge_vmin,
                                            vmax=edge_vmax)
            print(f'edge_norm: vmin={edge_vmin}, vmax={edge_vmax}')
            if cb:#colorbar
                edge_fig, edge_ax = plt.subplots(figsize=(1, 6))
                edge_fig.subplots_adjust(bottom=0.5)
                cb1 = matplotlib.colorbar.ColorbarBase(edge_ax, norm=edge_norm, cmap=edge_cmap)
                edge_ax.set_ylabel(r'$\Delta$ Intermixing (%)')
        
        node_cmap = plt.get_cmap('plasma')
        if node_norm is None:
            node_vmin = np.abs(np.array(node_colors)).min()
            node_vmax = np.abs(np.array(node_colors)).max()
            node_norm = matplotlib.colors.Normalize(vmin=node_vmin,
                                            vmax=node_vmax)
            print(f'node_norm: vmin={node_vmin}, vmax={node_vmax}')
        else:
            node_vmin = node_norm.vmin
            node_vmax = node_norm.vmax
            node_norm = matplotlib.colors.Normalize(vmin=node_vmin,
                                            vmax=node_vmax)
            print(f'node_norm: vmin={node_vmin}, vmax={node_vmax}')
            if cb:#colorbar
                node_fig, node_ax = plt.subplots(figsize=(1, 6))
                node_fig.subplots_adjust(bottom=0.5)
                cb2 = matplotlib.colorbar.ColorbarBase(node_ax, norm=node_norm, cmap=node_cmap)
                node_ax.set_ylabel('FCC (%)')
        #G_undirected = self.G.to_undirected(as_view=True)
        nx.draw_networkx_nodes(self.G, ax=ax,
                                       pos=pos, 
                                       node_size = node_size, 
                                       node_color=node_colors, 
                                       edgecolors='black',
                                       linewidths=0.5,
                                       cmap='plasma',
                                       vmin=node_vmin,
                                       vmax = node_vmax)
        nx.draw_networkx_edges(self.G, ax=ax,
                                       pos=pos, 
                                       width=edge_size, 
                                       edge_color=edge_colors, 
                                       edge_cmap=edge_cmap, 
                                       edge_vmin = edge_vmin, 
                                       edge_vmax=edge_vmax,
                                       arrows=False,
                                       node_size = node_size)

        return {'edge_norm':(edge_vmin, edge_vmax), 'node_norm': (node_vmin, node_vmax)}
        
    def plot_rattles(self, rattles, rattles_colors = None, ax=None, fig=None):
        min_energy = self.get_nodes_idx(0)[0][1]
        if rattles_colors is None:
            colors = ['green','red','blue']
            rattles_colors = dict()
            for i, r in enumerate(rattles):
                rattles_colors[r] = colors[i]
                
        find_color = lambda node: rattles_colors[[r for r in self.G.nodes()[node]['rattle'] if r in rattles][0]]
        get_color = lambda node: 'black' if all_filter(node) else find_color(node)
        rattle_filter = lambda node: any([r in self.G.nodes()[node]['rattle'] for r in rattles])
        all_filter = lambda node: all([r in self.G.nodes()[node]['rattle'] for r in rattles])
        nodes_list = [n for n in self.G.nodes if rattle_filter(n)]
        for n in nodes_list:
            for s in self.G.predecessors(n):
                if rattle_filter(s):
                    shears = [n[0], s[0]]
                    values = np.array([self.G.nodes()[n]['nn_energy'][0], self.G.nodes()[s]['nn_energy'][0]]) - min_energy
                    if ax is None:
                        if get_color(n) == 'black' and get_color(s) == 'black':
                            color = 'black'
                        else:
                            color = get_color(n) if get_color(n) != 'black' else get_color(s)
                        plt.plot(shears, values, '-', alpha = 1., c=color)
                        color = get_color(n)
                        plt.scatter(shears[0], values[0], c=color)
                        color = get_color(s)
                        plt.scatter(shears[1], values[1], c=color)
                    else:
                        ax.plot(shears,values, 'x-', alpha=1, color='#7f7f7f', markersize=4)
        
        
    def save(self, name):
        nx.write_gpickle(self.G, name+'.gpickle')
        
        
    def _gather_data():
        return 
    
    def _plot_data(ax, data):
        return 