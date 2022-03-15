import numpy as np
import click
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


@click.command()
@click.option("--input", "-i", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="path to hdf file with power traces")
@click.option("--pair", "-p", type=(int, int), help="pair of nodes")
@click.option("--sampling-rate", "-s", type=int, default=100, help="sampling rate of plotted data")
def cli(input_path, pair, sampling_rate):
    if sampling_rate > 100_000:
        click.UsageError("sampling rate cannot be larger than original sampling rate of 100kSps")
    ds_factor = 100_000 // sampling_rate

    with h5py.File(input_path, "r") as hf:
        f, axarr = plt.subplots(2, 1, sharex=True)
        for i in range(2):
            # downsample to 100 Hz
            axarr[i].plot(hf["time"][::ds_factor], hf["data"][f"node{pair[i]}"][::ds_factor] * 1e6)
            axarr[i].set_ylabel("Power [uW]")
        axarr[1].set_xlabel("Time [s]")
        plt.show()


if __name__ == "__main__":
    cli()
