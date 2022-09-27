import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from ase.io import read, write
import random
import string

def random_string(length):
    pool = string.ascii_letters + string.digits
    return ''.join(random.choice(pool) for i in range(length))


flatten = lambda regular_list: [item for sublist in regular_list for item in sublist]


def generate_graph(path, cutoff=3.1, index=-1):
    atoms = read(path, index=index)

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

def generate_component_subgraph(path, ideal_NN=12):
    G=generate_graph(path, cutoff=3.1, index=-1)
    
    # remove atoms with 12 neighbors
    G.remove_nodes_from([x[0] for x in G.degree if x[1]==ideal_NN])
    
    return G

def neighbor_ratios(G):
    G_full = G.copy()
    
    # remove atoms with 12 neighbors
    G.remove_nodes_from([x[0] for x in G.degree if x[1]==12])
    
    # get atom IDs
    vacancy_atoms = list(G.nodes())
    first_neighbors = set(flatten([list(G_full.neighbors(atom)) for atom in vacancy_atoms])) - set(vacancy_atoms)
    second_neighbors = set(flatten([list(G_full.neighbors(atom)) for atom in first_neighbors])) - set(vacancy_atoms) - first_neighbors
    
    # get neighbor info
    dfa = pd.DataFrame.from_dict([G_full.nodes[n] for n in vacancy_atoms])
    dfa['NN'] = 0
    dfb = pd.DataFrame.from_dict([G_full.nodes[n] for n in first_neighbors])
    dfb['NN'] = 1
    dfc = pd.DataFrame.from_dict([G_full.nodes[n] for n in second_neighbors])
    dfc['NN'] = 2
    df = pd.concat([dfa,dfb,dfc], ignore_index=True, sort=False)
    
    # get counts
    tmp_file = random_string(10)+'.csv'
    dd=df.groupby('atomic_number')['NN'].value_counts()
    dd.to_csv(tmp_file)
    dd = pd.read_csv(tmp_file)
    dd['count']=dd['NN.1']
    dd.drop(['NN.1'],axis=1,inplace=True)

    # get ratios
    Ni_ratios=[x[0] if len(x)!=0 else 0 for x in [dd.loc[(dd.atomic_number==28)&(dd.NN==NN)]['count'].tolist() for NN in [0,1,2]]]
    Cu_ratios=[x[0] if len(x)!=0 else 0 for x in [dd.loc[(dd.atomic_number==29)&(dd.NN==NN)]['count'].tolist() for NN in [0,1,2]]]

    os.remove(tmp_file)
    return list(zip(Ni_ratios,Cu_ratios))
 
    
def plot_all_component_subgraphs_neighbors(all_G, all_species_ratios, saveas='fig.png', n=6, m=7):    
    """
    Draw subgraph(s) of present vacancies vacancies.
    all_species_ratios (list): list of nearest neighbor ratios
    saveas (str): filename for image
    n (int): number of columns in figure
    m (int): number of rows in figure
    """
    fig,ax=plt.subplots(n,m, figsize=(12,10))

    colors=plt.get_cmap("Dark2").colors

    c=0
    for row in range(0,n):
        for col in range(0,m):
            G = all_G[c]
            pos = nx.spring_layout(G, center=(0,0), seed=42, weight='weight')
            nx.draw_networkx_edges(G, pos, ax=ax[row][col], alpha=0.7)
            int_node_ids = list(G.nodes()) 
            

            node_colors=[]
            for node in list(G.nodes()):
                if G.nodes[node]['atomic_number'] ==28:
                    node_colors+= [colors[0]]
                else:
                    node_colors+= [colors[1]]

            
            nx.draw_networkx_nodes(G, pos, ax=ax[row][col], node_size=40, node_color=node_colors)

            ax[row][col].set_title(c)
            c+=1

    plt.tight_layout()
    plt.savefig(saveas, dpi=300)
    plt.close()


def draw_pies(all_species_ratios, step, ax):
    """
    Draw pie charts of atom types for first, second, and third nearest neighbors to vacancies.
    all_species_ratios (list): list of nearest neighbor ratios
    step (int): step in series (title of plot) 
    ax (matplotlib.pyplot.axis): axis on which to plot figure
    """
    sizes_inner  = [all_species_ratios[step, 0, 0], all_species_ratios[step, 0, 1]]
    sizes_middle = [all_species_ratios[step, 1, 0], all_species_ratios[step, 1, 1]]
    sizes_outer  = [all_species_ratios[step, 2, 0], all_species_ratios[step, 2, 1]]

    colors_inner  = plt.get_cmap("Dark2").colors
    colors_middle = plt.get_cmap("Set2").colors
    colors_outer  = plt.get_cmap("Pastel2").colors

    outside_donut = ax.pie(sizes_outer, labels=None, colors=colors_outer, radius=0.9,
                            startangle=90, autopct=None, wedgeprops=dict(width=0.6, edgecolor='black'),
                            textprops=dict(fontsize=18, color='black'))

    middle_donut = ax.pie(sizes_middle, labels=None, colors=colors_middle, radius=0.7,
                           startangle=90, autopct=None, wedgeprops=dict(width=0.6, edgecolor='black'))

    inside_donut = ax.pie(sizes_inner, labels=None, colors=colors_inner, radius=0.5,
                           startangle=90, autopct=None, wedgeprops=dict(width=0.6, edgecolor='black'))


    centre_circle = plt.Circle((0, 0), 0.25, facecolor='white', linewidth=1.25, edgecolor='black')
    ax.add_patch(centre_circle)


def NN_pie_plots(all_species_ratios, saveas='vacancyNN.png', n=6, m=7):
    """
    Draw pie charts of atom types for first, second, and third nearest neighbors to vacancies.
    all_species_ratios (list): list of nearest neighbor ratios 
    saveas (str): filename for image
    n (int): number of columns in figure
    m (int): number of rows in figure
    """
    fig,ax=plt.subplots(n,m, figsize=(12,10))
    step=0
    for row in range(0,n):
        for col in range(0,m):
            draw_pies(all_species_ratios, step, ax[row][col])
            ax[row][col].set_title(step, pad=-1)
            step+=1
    plt.tight_layout()
    plt.savefig(saveas, dpi=300)
    plt.show()
