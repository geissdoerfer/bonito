import numpy as np
import h5py
import click
from multiprocessing import Pool
from multiprocessing import cpu_count
from itertools import combinations


class CachedDataset(object):
    """Wrapper around default h5py Dataset that accelerates single index access to the data.

    hdf5 loads data 'lazily', i.e. only loads data into memory that is explicitly indexed. This incurs high
    delays when accessing single values successively. Chunk caching should improve this, but we couldn't observe
    a gain. Instead, this class implements a simple cached dataset, where data is read into memory in blocks
    and single index access reads from that cache.

    Args:
        dataset: underlying hdf5 dataset
        cache_size: number of values to be held in memory
    """

    def __init__(self, dataset: h5py.Dataset, cache_size: int = 10_000_000):
        self._ds = dataset
        self._istart = 0
        self._iend = -1
        self._cache_size = cache_size

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._ds[key]
        elif isinstance(key, int):
            return self.get_cached(key)

    def __len__(self):
        return len(self._ds)

    def update_cache(self, idx):
        self._istart = (idx // self._cache_size) * self._cache_size
        self._iend = min(len(self._ds), self._istart + self._cache_size)
        self._buf = self._ds[self._istart : self._iend]

    def get_cached(self, idx):
        if idx >= self._istart and idx < self._iend:
            return self._buf[idx - self._istart]
        else:
            self.update_cache(idx)
            return self.get_cached(idx)


class DataReader(object):
    """Convenient and cached access to an hdf5 database with power traces from multiple nodes."""

    def __init__(self, path, cache_size=10_000_000):
        self.path = path
        self.cache_size = cache_size
        self._datasets = dict()

    def __enter__(self):
        self._hf = h5py.File(self.path, "r")
        self.nodes = list(self._hf["data"].keys())

        self.time = self._hf["time"]
        for node in self._hf["data"].keys():
            self._datasets[node] = CachedDataset(self._hf["data"][node], self.cache_size)

        return self

    def __exit__(self, *exc):
        self._hf.close()

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._datasets[f"node{key}"]
        else:
            return self._datasets[key]

    def __len__(self):
        return len(self.time)

    def pairs(self):
        """Returns all unique combinations between the nodes in this database."""
        return combinations(range(len(self.nodes)), 2)


class BatteryfreeDevice(object):
    """Simple simulation model of battery-free device.

    Args:
        capacity: energy storage capacity in farad
        v_on: turn-on threshold in volts
        v_off: turn-off threshold in volts
    """

    def __init__(self, capacity, v_on, v_off):
        self._capacity = capacity
        self._von = v_on
        self._voff = v_off
        self.energy_per_cycle = 0.5 * self._capacity * (self._von**2 - self._voff**2)

        self.reset()

    @property
    def charged(self):
        return self._estored >= self.energy_per_cycle

    def harvest(self, energy):
        """Charges the capacitor by given amount of energy."""
        self._estored += energy

    def reset(self):
        self._estored = 0
        self._charged = False


def pwr2time(input_path: click.Path, pair: tuple, capacity: float = 17e-6, von: float = 3.0, voff: float = 2.4):
    """Calculates synchronized charging times from two power traces.

    In order to obtain realistic 'bivariate' samples of the charging times of two nodes, we have to
    make sure that a sample from one node is taken at the same time as the corresponding sample
    of the second node. To this end, we run a simple simulation, where two nodes are charged
    from their corresponding time-synchronized power traces. We note the time until the first
    node reaches the turn-on threshold and continue the simulation until the second node also
    reaches the turn-on threshold. Now we have one 'bivariate' sample and restart the simulation
    from the current time.

    Args:
        dr: wraps underlying hdf database with power traces
        pair: the pair of nodes for which to compute synchronized charging times
        capacity: simulated energy storage capacity in farad
        v_on: simulated turn-on threshold in volts
        v_off: simulated turn-off threshold in volts
    """

    # Stores charging times
    csi = [list(), list()]
    # Stores times
    tsi = list()

    nodes = [BatteryfreeDevice(capacity, von, voff), BatteryfreeDevice(capacity, von, voff)]

    with DataReader(input_path) as dr:

        # Assume that sampling interval is constant
        Ts = np.mean(np.diff(dr.time[:1000000]))

        # Stores last time index at which both nodes were fully charged
        last_charged = 0

        for i in range(len(dr.time)):
            for j, node in enumerate(nodes):

                if not node.charged:
                    node.harvest(Ts * dr[j][i])
                    if node.charged:
                        csi[j].append(i - last_charged)

            # When both nodes are fully charged
            if all([node.charged for node in nodes]):
                # Log the timestamp when both nodes are charged
                tsi.append(i)
                last_charged = i

                for node in nodes:
                    node.reset()

            if i % (len(dr.time) // 100) == 0:
                print(f"{(100 * i)//len(dr.time)}% of pair {pair} done")

        ts = dr.time[tsi]

    # Cut the charging times to equal lengths
    for node_idx in range(2):
        csi[node_idx] = csi[node_idx][: len(tsi)]

    return ts, np.array(csi[0]) * Ts, np.array(csi[1]) * Ts


@click.command(help="Converts pairwise power traces to charging times")
@click.option("--output", "-o", "output_path", type=click.Path(dir_okay=False), required=True, help="path for output hdf file with charging time traces")
@click.option("--input", "-i", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="path to hdf file with power traces")
@click.option("--capacity", "-c", type=float, default=17e-6, help="simulated capacitor voltage")
@click.option("--von", type=float, default=3.0, help="simulated turn-on threshold")
@click.option("--voff", type=float, default=2.4, help="simulated turn-off threshold")
def cli(input_path, output_path, capacity, von, voff):

    with DataReader(input_path) as dr:
        pairs = dr.pairs()

    args = list()
    for pair in pairs:
        args.append((input_path, pair, capacity, von, voff))

    with Pool(cpu_count()) as pool:
        results = pool.starmap(pwr2time, args)

    with h5py.File(output_path, "w") as hf_out, h5py.File(input_path, "r") as hf_in:
        for arg, (ts, cs0, cs1) in zip(args, results):

            grp = hf_out.create_group(str(arg[1]))
            ds_time = grp.create_dataset("time", data=ts, dtype="f8")
            ds_time.attrs["unit"] = "second"

            ds = grp.create_dataset("node0", data=cs0, dtype="f8")
            ds = grp.create_dataset("node1", data=cs1, dtype="f8")
            for i in range(2):
                ds.attrs["model"] = hf_in["data"][f"node{arg[1][i]}"].attrs["model"]
                ds.attrs["host"] = hf_in["data"][f"node{arg[1][i]}"].attrs["host"]
                ds.attrs["unit"] = "second"


if __name__ == "__main__":
    cli()
