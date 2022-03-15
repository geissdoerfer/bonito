import h5py
import numpy as np
import click
import matplotlib.pyplot as plt
from collections import namedtuple

from neslab.bonito import bonito
from neslab.bonito import modest
from neslab.bonito import greedy

from neslab.bonito import NormalDistribution
from neslab.bonito import ExponentialDistribution
from neslab.bonito import GaussianMixtureModel

model_map = {"norm": NormalDistribution, "exp": ExponentialDistribution, "gmm": GaussianMixtureModel}


def run_protocol(grp_pair: h5py.Dataset, protocol: str, bonito_target_probability=0.99):
    tchrg_node0 = grp_pair["node0"][:]
    tchrg_node1 = grp_pair["node1"][:]

    if protocol == "greedy":
        pro_gen = greedy((tchrg_node0, tchrg_node1))
    elif protocol == "modest":
        pro_gen = modest((tchrg_node0, tchrg_node1))
    elif protocol == "bonito":
        dist_cls_node0 = model_map[grp_pair["node0"].attrs["model"]]
        dist_cls_node1 = model_map[grp_pair["node1"].attrs["model"]]
        pro_gen = bonito((tchrg_node0, tchrg_node1), (dist_cls_node0, dist_cls_node1), bonito_target_probability)
    else:
        raise NotImplementedError

    connection_intervals = np.empty((len(grp_pair["time"]),))

    for i, (ci, success) in enumerate(pro_gen):
        if success:
            connection_intervals[i] = ci
        else:
            connection_intervals[i] = np.nan

    return connection_intervals


@click.command()
@click.option("--input", "-i", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="path to hdf file with charging time traces")
@click.option("--pair", "-p", type=(int, int), help="pair of nodes")
@click.option("--target-probability", "-t", type=float, help="Bonito target probability", default=0.99)
def cli(input_path, pair, target_probability):
    with h5py.File(input_path, "r") as hf:
        try:
            grp = hf[str(pair)]
        except KeyError:
            grp = hf[str(tuple(reversed(pair)))]

        f, ax = plt.subplots()
        ax.plot(grp["time"][:], grp["node0"][:], color="gray", linestyle="--", label="charging time node0")
        ax.plot(grp["time"][:], grp["node1"][:], color="gray", linestyle="-.", label="charging time node1")

        for protocol in ["greedy", "modest", "bonito"]:
            cis = run_protocol(grp, protocol, target_probability)
            ax.plot(grp["time"][:], cis, label=f"connection interval {protocol}")
            success_rate = np.count_nonzero(~np.isnan(cis)) / len(grp["time"])
            delay = np.nanmedian(cis)
            print(f"{protocol}: success_rate={success_rate*100:.2f}% delay={delay:.3f}s")

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Time [s]")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    cli()
