#!/usr/bin/env python
import sys
sys.path.insert(0, '/qfs/projects/sppsi/spru445/schnetpack_builds/schnetpack_16_fast_copy/src')

import numpy as np
import random
import os
import os.path as op
import torch
import logging
import schnetpack as spk
import csv  
from schnetpack.utils import (
    get_dataset,
    get_metrics,
    get_loaders,
    get_statistics,
    get_model,
    #get_trainer,
    ScriptError,
    evaluate,
    setup_run,
    get_divide_by_atoms
)
from schnetpack.utils.script_utils.training import (
    get_loss_fn,#Import these for get_trainer
    simple_loss_fn,
    tradeoff_loss_fn,
    get_metrics
)
from schnetpack.data import AtomsDataError, AtomsDataSubset
from schnetpack.data.atoms import PreprocessedData
from schnetpack.environment import SimpleEnvironmentProvider, MBEnvironmentProvider 
from schnetpack.utils.script_utils.active_parsing import build_parser
from ase.db import connect
from scipy.special import erfinv
from shutil import copyfile

import os
import glob

import schnetpack as spk
import torch
from torch.optim import Adam

__all__ = ["get_trainer", "simple_loss_fn", "tradeoff_loss_fn", "get_metrics"]

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def get_statistics_active(
    args, split_path, train_loader, atomref, divide_by_atoms=False, logging=None, dataset=None
):
    """
    Get statistics for molecular properties. Use split file if possible.

    Args:
        args (argparse.Namespace): parsed script arguments
        split_path (str): path to the split file
        train_loader (spk.data.AtomsLoader): dataloader for training set
        atomref (dict): atomic references
        divide_by_atoms (dict or bool): divide mean by number of atoms if True
        logging: logger

    Returns:
        mean (dict): mean values for the selected properties
        stddev (dict): stddev values for the selected properties
    """
    # check if split file exists
    if not os.path.exists(split_path):
        raise ScriptError("No split file found at {}".format(split_path))
    split_data = np.load(split_path)

    # check if split file contains statistical data
    if "mean" in split_data.keys():
        mean = {args.property: torch.from_numpy(split_data["mean"])}
        stddev = {args.property: torch.from_numpy(split_data["stddev"])}
        if logging is not None:
            logging.info("cached statistics was loaded...")

    # calculate statistical data
    else:
        energies = []
        with connect(args.datapath) as db:
            for row in db.select():
                #at = row.toatoms()
                atomic_numbers = row.numbers
                num_Ni = np.sum(atomic_numbers[atomic_numbers == 28])
                num_Cu = np.sum(atomic_numbers[atomic_numbers == 29])
                energies.append((row.energy + num_Ni * 0.350 + num_Cu * 0.223) / (num_Ni+num_Cu))
        mean = {args.property:torch.from_numpy(np.array([np.mean(energies)]))}
        stddev = {args.property:torch.from_numpy(np.array([np.std(energies, ddof=1)]))}
        np.savez(
            split_path,
            train_idx=split_data["train_idx"],
            val_idx=split_data["val_idx"],
            test_idx=split_data["test_idx"],
            mean=mean[args.property].numpy(),
            stddev=stddev[args.property].numpy(),
        )
    logging.info(f'mean is: {mean[args.property].numpy()[0]}')
    logging.info(f'stddev is: {stddev[args.property]}.numpy()[0]')
    return mean, stddev


def get_examine_set(args, dataset, iteration):
    split_file = os.path.join(args.modelpath, f"split_{iteration}.npz")
    if split_file is None or os.path.exists(split_file)==False:
        raise AtomsDataError(f"{split_file} does not exist")

    # load splits
    S = np.load(split_file)
    train_idx = S["train_idx"].tolist()
    val_idx = S["val_idx"].tolist()   
    test_idx = S["test_idx"].tolist()

    # sample subset
    random.shuffle(test_idx)
    examine_idx = test_idx[:args.n_to_examine]
    test_idx = test_idx[args.n_to_examine:]

    # get loader for examine set
    data_examine = AtomsDataSubset(dataset, examine_idx)

    examine_loader = spk.data.AtomsLoader(
            data_examine, batch_size=args.batch_size, num_workers=1, pin_memory=True
        )
        
    return examine_idx, examine_loader


def create_new_split(train_args, idx_to_add, iteration):
    split_file = os.path.join(train_args.modelpath, f"split_{iteration}.npz")
    if split_file is None or os.path.exists(split_file)==False:
        raise AtomsDataError(f"{split_file} does not exist")

    # load previous splits
    S = np.load(split_file)
    train_idx = S["train_idx"].tolist()
    val_idx = S["val_idx"].tolist()
    test_idx = S["test_idx"].tolist()

    # add indices to train and away from test
    train_idx += idx_to_add
    test_idx = list(set(test_idx)-set(idx_to_add))

    # write new split path
    new_iteration = str(int(iteration)+1).zfill(2)
    new_split_file = os.path.join(train_args.modelpath, f"split_{new_iteration}.npz")    
    np.savez(new_split_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)



def generate_loaders(train_args, dataset, iteration):
    # get dataset
    split_path = os.path.join(train_args.modelpath, f"split_{iteration}.npz")
    train_loader, val_loader, test_loader = get_loaders(
        train_args, dataset=dataset, split_path=split_path, logging=logging
    )

    # get statistics
    # TODO: get bulk stats at beginning and apply throughout training
    atomref = dataset.get_atomref(train_args.property)
    mean, stddev = get_statistics_active(
        args=train_args,
        split_path=split_path,
        train_loader=train_loader,
        atomref=atomref,
        divide_by_atoms=get_divide_by_atoms(train_args),
        logging=logging,
        dataset=dataset
    )

    return (train_loader, val_loader, test_loader), (atomref, mean, stddev)


def new_get_metrics(args):
    #make 2 sets of metrics, one for training and one for validation
    metrics = []
    for name in ['train','val']:
        # setup property metrics
        metrics += [
            spk.train.metrics.MeanAbsoluteError(args.property, args.property, name='MAE_Energy_'+name),
            spk.train.metrics.RootMeanSquaredError(args.property, args.property, name='RMSE_Energy_'+name),
        ]

        # add metrics for derivative
        derivative = spk.utils.get_derivative(args)
        if derivative is not None:
            metrics += [
                spk.train.metrics.MeanAbsoluteError(
                    derivative, derivative, element_wise=True, name='MAE_Forces_'+name
                ),
                spk.train.metrics.RootMeanSquaredError(
                    derivative, derivative, element_wise=True, name='RMSE_Forces_'+name
                ),
            ]

        # Add stress metric
        stress = spk.utils.get_stress(args)
        if stress is not None:
            metrics += [
                spk.train.metrics.MeanAbsoluteError(stress, stress, element_wise=True),
                spk.train.metrics.RootMeanSquaredError(stress, stress, element_wise=True),
            ]

    return metrics

def get_trainer(args, model, train_loader, val_loader, metrics):
    # setup optimizer
    # filter for trainable parameters (https://github.com/pytorch/pytorch/issues/679)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params, lr=args.lr)

    # setup hook and logging
    hooks = [spk.train.MaxEpochHook(args.max_epochs)]
    if args.max_steps:
        hooks.append(spk.train.MaxStepHook(max_steps=args.max_steps))
    schedule = spk.train.ReduceLROnPlateauHook(
        optimizer=optimizer,
        patience=args.lr_patience,
        factor=args.lr_decay,
        min_lr=args.lr_min,
        window_length=1,
        stop_after_min=True,
    )
#     schedule = spk.train.WarmRestartHook( 
#         T0=args.n_epochs,
#         Tmult=1,
#         each_step=False,
#         lr_min=1e-6,
#         lr_factor=0.8,
#         patience=160
#     )
    hooks.append(schedule)

    if args.logger == "csv":
        logger = spk.train.CSVHook(
            os.path.join(args.modelpath, "log"),
            metrics,
            every_n_epochs=args.log_every_n_epochs,
        )
        hooks.append(logger)
    elif args.logger == "tensorboard":
        logger = spk.train.TensorboardHook(
            os.path.join(args.modelpath, "log"),
            metrics,
            every_n_epochs=args.log_every_n_epochs,
        )
        hooks.append(logger)

    # setup loss function
    loss_fn = get_loss_fn(args)

    # setup trainer
    trainer = spk.train.Trainer(
        args.modelpath,
        model,
        loss_fn,
        optimizer,
        train_loader,
        val_loader,
        checkpoint_interval=args.checkpoint_interval,
        keep_n_checkpoints=args.keep_n_checkpoints,
        hooks=hooks,
    )
    return trainer

def get_error_distribution(args):
 
    csvfilename = os.path.join(args.modelpath, 'log', 'log.csv')
    data = []
    with open(csvfilename, "r", encoding="utf-8", errors="ignore") as scraped:
        reader = csv.reader(scraped, delimiter=',')
        row_index=0
        for row in reader:
            if row:  # avoid blank lines
                row_index += 1
                columns = [str(row_index), row[6], row[7]] #6-MAE Forces Train 7-RMSE Forces Train
                data.append(columns)
    

    last_row = data[-1]
    mae = float(last_row[1])
    mse = float(last_row[2])**2
    var = mse - mae**2
    return mae, np.sqrt(var)



def main(args):
    # setup
    train_args = setup_run(args)

    device = torch.device("cuda" if args.cuda else "cpu")

    # define metrics
    metrics = new_get_metrics(train_args)

    # generate dataset
    logging.info(f"datapath: {args.datapath}")
    if train_args.derivative=='forces':
        available_properties=['energy','forces']
    else:
        available_properties=['energy']
    dataset = PreprocessedData(train_args, available_properties=available_properties,
                              centering_function=None)
    #No precomp:
#     # get dataset
#     environment_provider = MBEnvironmentProvider(train_args.cutoff)
#     dataset = get_dataset(train_args, environment_provider=environment_provider)


    # old active learning params, remove when you know this works
#     train_args.iterations = 50
#     train_args.n_to_examine = 5000
#     train_args.percent_add = 0.3 # add up to top 30% examples to training set
#     train_args.error_tolerance = 0.08
#     train_args.eval_target = "forces"
    start_at = train_args.start_at

    logging.info("training...")
    if args.mode == "train":
        #handle starting from the end
        if start_at == -1:
            best_models = sorted(glob.glob(os.path.join(train_args.modelpath, "best_model_*")))
            if best_models == []:
                start_at = 0
            else:
                final_best_model = best_models[-1]
                start_at = int(best_models[-1].split('_')[-1])

        start_at_name = str(start_at).zfill(2)

        if start_at == 0:
            # get dataloader
            # get dataloaders
            split_path = os.path.join(args.modelpath, "split_01.npz")
            train_loader, val_loader, test_loader = get_loaders(
            args, dataset=dataset, split_path=split_path, logging=logging
            )

            
            atomref = dataset.get_atomref(args.property)
            mean, stddev = get_statistics(
            args=args,
            split_path=split_path,
            train_loader=train_loader,
            atomref=atomref,
            divide_by_atoms=get_divide_by_atoms(args),
            logging=logging,
            )
            model = get_model(train_args, None, mean, stddev, atomref, logging=logging)

        else:
            if os.path.isfile(os.path.join(train_args.modelpath, f"best_model_{start_at_name}")):
                logging.info(f'Reloading from best_model_{start_at_name}')
                model = spk.utils.load_model(
                    os.path.join(args.modelpath, f"best_model_{start_at_name}"), map_location=device)
                loaders, stats = generate_loaders(train_args, dataset,
                                                 str(start_at+1).zfill(2))
                train_loader, val_loader, test_loader = loaders
            else:
                logging.warning(f"best_model_{start_at_name} doesn't exist")
                sys.exit(f"best_model_{start_at_name} doesn't exist")

   
        for i in range(start_at+1, train_args.num_iterations+1):
            i = str(i).zfill(2)
            # switch to train mode
            model.train()

            # build trainer
            trainer = get_trainer(train_args, model, train_loader, val_loader, metrics)

            # run training
            trainer.train(device, n_epochs=args.n_epochs)

            # save best_model of the iteration
            torch.save(model, os.path.join(train_args.modelpath, f"best_model_{i}"))

            # extract random subset of test set to examine 
            examine_idx, examine_loader = get_examine_set(train_args, dataset, i)

            # perform inference, get error, choose top percent_add
            model.eval()
            for metric in metrics:
                metric.reset()

            eval_error = []
            for batch in examine_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                result = model(batch)

                #MAE for energy and forces; for RMSE change to 1 and 3
                if train_args.eval_target == "energy":
                    eval_error += metrics[0].give_diffs(batch, result).detach().cpu().squeeze().tolist()
                elif train_args.eval_target == 'forces':
                    eval_error += metrics[2].give_diffs(batch, result).detach().cpu().squeeze().tolist()



            logging.info(f'len eval_error {len(eval_error)}')
            eval_error = [np.mean([np.linalg.norm(F) for F in ats]) for ats in eval_error]
            #eval_error = np.linalg.norm(eval_error, axis=2) #norm the force vectors
            #eval_error = np.mean(eval_error, axis=1) #average over atoms

            # get index of sort high to low
            # calulate a cutoff for when to not add item
            mae, stdvae = get_error_distribution(args)
            cutoff = erfinv(1-args.error_tolerance) * stdvae + mae
            logging.info(f'error tolerance {cutoff} for mean AE {mae} and stdv AE {stdvae}')

            # TODO: add some kind of logging or csv print to get sample errors
            n_samples_to_add = int(len(eval_error)*train_args.max_to_add)
            idx_highest_errors = np.argsort(eval_error)[-n_samples_to_add:]
            idx_to_add = [examine_idx[idx] for idx in idx_highest_errors if eval_error[idx]>=cutoff]

            logging.info(f'adding {len(idx_to_add)} examples to training set')
            # add indexes to new split file 
            create_new_split(train_args, idx_to_add, i)
        
        logging.info("Done with active learning. Beginning annealing...")
        
        #load the best model
        model = spk.utils.load_model(
                    os.path.join(args.modelpath, f"best_model"), map_location=device)
        
        #move the name of the best model to keep the old one
        copyfile(os.path.join(train_args.modelpath, f"best_model"), os.path.join(train_args.modelpath, f"active_best_model"))
                 
        #save the current model as best model
        torch.save(model, os.path.join(train_args.modelpath, f"best_model"))
                 
        #train the best model
        train_args.n_epochs = 500
        # build trainer
        trainer = get_trainer(train_args, model, train_loader, val_loader, metrics)

        # run training
        trainer.train(device, n_epochs=args.n_epochs)
        
        #endfor
        logging.info("...training done!")

    elif args.mode == "eval":

        # remove old evaluation files
        evaluation_fp = os.path.join(args.modelpath, "evaluation.txt")
        if os.path.exists(evaluation_fp):
            if args.overwrite:
                os.remove(evaluation_fp)
            else:
                raise ScriptError(
                    "The evaluation file does already exist at {}! Add overwrite flag"
                    " to remove.".format(evaluation_fp)
                )

        # load model
        logging.info("loading trained model...")
        model = spk.utils.load_model(
            os.path.join(args.modelpath, "best_model"), map_location=device
        )

        # run evaluation
        logging.info("evaluating...")
        if spk.utils.get_derivative(train_args) is None:
            with torch.no_grad():
                evaluate(
                    args,
                    model,
                    train_loader,
                    val_loader,
                    test_loader,
                    device,
                    metrics=metrics,
                )
        else:
            evaluate(
                args,
                model,
                train_loader,
                val_loader,
                test_loader,
                device,
                metrics=metrics,
            )
        logging.info("... evaluation done!")

    else:
        raise ScriptError("Unknown mode: {}".format(args.mode))


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "from_json":
        args = spk.utils.read_from_json(args.json_path)

    main(args)
