import os
import csv
import numpy


__all__ = ["evaluate", "evaluate_dataset"]

#THIS EVALUATION FUNCTION HAS BEEN EDITED FROM THE ORIGINAL
def evaluate(
    args,
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    metrics,
    custom_header=None,
):

    paths = []

    loaders = dict(train=train_loader, validation=val_loader, test=test_loader)
    for datasplit in args.split:
        
        paths += [os.path.join( args.modelpath, "energies_{}.csv".format(datasplit) )]

        try:
           derivative = model.output_modules[0].derivative
        except:
           derivative = None

        if derivative is not None:
            paths += [os.path.join( args.modelpath, 'forces_{}.csv'.format(datasplit) )]

        for p in paths:
            if os.path.exists(p):
                os.remove(p)
        
        evaluate_dataset(metrics, model, loaders[datasplit], device, paths)


def evaluate_dataset(metrics, model, loader, device, paths_list):
    model.eval()

    energies = []

    for metric in metrics:
        metric.reset()

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        result = model(batch)

        energy = metrics[0].give_diffs(batch, result).detach().cpu().numpy()
        
        print(energy)

        for e in energy:
            energies.append(e[0])

    #with open(paths_list[0], 'w', newline='') as f:
     #   writer = csv.writer(f, delimiter = ',')

    numpy.savetxt(paths_list[0], numpy.array(energies), delimiter=',')








