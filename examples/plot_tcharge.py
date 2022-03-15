import matplotlib.pyplot as plt
import numpy as np
import h5py
import click
import seaborn as sns


def extract_moments(sig: np.ndarray, window_size: int = 500):
    """Extracts first and second order moments of a signal using a sliding window.

    Args:
        sig: input signal
        window_size: size of the sliding window in number of samples
    """
    moments = np.empty((len(sig), 2))
    moments[0, 0] = sig[0]
    moments[:window_size, 1] = np.var(sig[:window_size])

    for i in range(1, window_size):
        moments[i, 0] = np.mean(sig[:i])

    for i in range(window_size, len(sig)):
        moments[i, 0] = np.mean(sig[i - window_size : i])
        moments[i, 1] = np.var(sig[i - window_size : i])

    return moments


def plot_time(grp: h5py.Group):
    """Plots charging times of two nodes over time.

    Args:
        grp: hdf5 group with traces from two nodes
    """
    f, axarr = plt.subplots(2, 1, sharex=True)
    for i in range(2):
        axarr[i].plot(grp["time"][:], grp[f"node{i}"][:])
    plt.show()


def plot_hist(grp, normalize: float = 30.0):
    """Plots distribution of charging times of two nodes.

    Args:
        grp: hdf5 group with traces from two nodes
        normalize: if provided, specifies duration of window for normalization of first two moments
    """
    data = np.vstack((grp["node0"][:], grp[f"node1"][:])).T
    if normalize is not None:
        for i in range(2):
            # approximate window size from normalization period
            ws = int(normalize / np.mean(data[:, i]))
            moments = extract_moments(data[:, i], ws)
            data[:, i] = (data[:, i] - moments[:, 0]) / moments[:, 0]

    sns.jointplot(x=data[:, 0], y=data[:, 1])
    plt.show()


@click.command()
@click.option("--input", "-i", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="path to hdf file with charging time traces")
@click.option("--pair", "-p", type=(int, int), help="pair of nodes")
@click.option("--hist", is_flag=True, help="plot histogram")
def cli(input_path, pair, hist):
    with h5py.File(input_path, "r") as hf:
        try:
            grp = hf[str(pair)]
        except KeyError:
            grp = hf[str(tuple(reversed(pair)))]
        if hist:
            plot_hist(grp)
        else:
            plot_time(grp)


if __name__ == "__main__":
    cli()
