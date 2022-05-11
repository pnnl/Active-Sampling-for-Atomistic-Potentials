import os
import os.path as op
import numpy as np
import tqdm

import utils

datadir='/qfs/projects/sppsi/spru445/NNs/active_learning/active_anneal_N1_high_cutoff/experiments-for-paper-011/best_model/'
sample='N3'
#savedir='/qfs/projects/sppsi/sppsi_graphs/images/vacancy_NNs'
savedir='./'
# collect all rattles
runs = [op.join(datadir, sample, run, 'nn-xyz') for run in os.listdir(op.join(datadir, sample))]
runs = runs[:1]
print(f'{len(runs)} rattle plots to generate')


if not op.isdir(op.join(savedir,sample)):
    os.mkdir(op.join(savedir,sample))


for run in tqdm.tqdm(runs):
    steps = sorted(os.listdir(run))
    all_species_ratios=[]
    all_G=[]
    figname = op.join(savedir, sample, f"{run.split('/')[-2]}")
    
    i=0
    for step in range(len(steps)):
        G=utils.generate_graph(op.join(run,steps[step]), cutoff=3.1, index=-1)
        species_ratios=utils.neighbor_ratios(G)
        if i == 0:
            all_species_ratios=species_ratios
        else:
            all_species_ratios = np.vstack([all_species_ratios,species_ratios])
            
        # remove atoms with 12 neighbors
        G.remove_nodes_from([x[0] for x in G.degree if x[1]==12])       
        all_G.append(G)
        i+=1
        
    try:
        # reshape to n_steps, NN collections, N atom types
        all_species_ratios=all_species_ratios.reshape(-1,3,2)
        utils.plot_all_component_subgraphs_neighbors(all_G, all_species_ratios, saveas=figname+'-subgraphs.png')     
        utils.NN_pie_plots(all_species_ratios, saveas=figname+'-NN.png')
    except:
        continue
