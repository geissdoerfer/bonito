import click
import numpy as np
import h5py
import matplotlib.pyplot as plt

from neslab.bonito import ExponentialDistribution
from neslab.bonito import NormalDistribution
from neslab.bonito import GaussianMixtureModel

model_map = {"norm": NormalDistribution, "exp": ExponentialDistribution, "gmm": GaussianMixtureModel}


def learn_trace(sig: np.array, dist_cls):
    """Iterates trace of charging times and learns parameters of distribution model.

    Args:
        sig: observations of a device's charging times
        model_type: model of charging time distribution
    Returns:
        array of model parameters after every model update step
    """

    dist = dist_cls()
    params = np.empty((len(sig), dist.nparams))
    for i, x in enumerate(sig[:]):
        dist.sgd_update(x)
        # store current model parameters
        params[i, :] = dist._mp.flatten()

    return params


@click.command()
@click.option("--input", "-i", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="path to hdf file with charging time traces")
@click.option("--model", "-m", type=click.Choice(["norm", "exp", "gmm"]))
@click.option("--node", "-n", type=int)
def cli(input_path, model, node):
    with h5py.File(input_path, "r") as hf:
        if node == 0:
            grp = hf[f"(0, 1)"]
            ds = grp["node0"]
        else:
            grp = hf[f"(0, {node})"]
            ds = grp["node1"]

        if model is None:
            model = ds.attrs["model"]

        dist_cls = model_map[model]

        params = learn_trace(ds[:], dist_cls)

        if model == "gmm":
            click.echo("plotting not supported for gaussian mixture model")
            return
        elif model == "exp":
            model_mean = 1.0 / params[:, 0]
        else:
            model_mean = params[:, 0]

        f, ax = plt.subplots()
        plt.plot(ds[:], label="observations")
        plt.plot(model_mean, label="model mean")
        plt.xlabel("Sample number")
        plt.ylabel("Charging time [s]")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    cli()
