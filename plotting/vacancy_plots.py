import os
import os.path as op
import numpy as np
import tqdm
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--sample', required=True, type=str, help="sample ID")
parser.add_argument('--datadir', required=True, type=str, help="path where output of shear tests is stored")
parser.add_argument('--savedir', default='./', type=str, help="path to save images")
args = parser.parse_args()


# collect all rattles
runs = [op.join(args.datadir, args.sample, run, 'nn-xyz') for run in os.listdir(op.join(args.datadir, args.sample))]
runs = runs[:1]
print(f'{len(runs)} rattle plots to generate')


if not op.isdir(op.join(args.savedir, args.sample)):
    os.mkdir(op.join(args.savedir, args.sample))


for run in tqdm.tqdm(runs):
    steps = sorted(os.listdir(run))
    all_species_ratios=[]
    all_G=[]
    figname = op.join(args.savedir, args.sample, f"{run.split('/')[-2]}")
    
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
