[![Build Status](https://app.travis-ci.com/geissdoerfer/bonito.svg?branch=master)](https://app.travis-ci.com/geissdoerfer/bonito)

# Bonito connection protocol

This directory contains a Python implementation of the Bonito protocol presented in our [NSDI 2022 paper](https://nes-lab.org/pubs/2022-Geissdoerfer-Bonito.pdf).
It enables users to compare the performance of Bonito to baseline approaches using the real-world energy-harvesting traces provided TODO.


## Installation

Install the python package together with the requirements for the examples using either

```
pip install neslab.bonito[examples]
```

or

```
pipenv install neslab.bonito[examples]
```

## Data

We provide 32h of time-synchronized energy-harvesting traces from 5 different scenarios involving solar panels and piezoelectric harvesters. The data was recorded with [Shepherd](https://nes-lab.org/pubs/2019-Geissdoerfer-Shepherd.pdf), a measurement tool that records time-synchronized voltage and current traces from one or more energy-harvesting nodes with high rate and resolution.

- The *jogging* dataset comprises traces from two participants, each equipped with two piezoelectric harvester at the ankles and a solar panel at the left wrist. The two participants run together for an hour in a public park, including short walking and standing breaks.
- For the *stairs* dataset, we recorded traces from six solar panels that are embedded into the surface of an outdoor stair in front of a lecture hall. Over the course of one hour, numerous students pass the stairs, leading to temporary shadowing effects on some or all of the solar panels.
- The *office* dataset comprises traces from five solar panels mounted on the doorframe and walls of an office with fluorescent lights. During the one-hour recording, people enter and leave the office and operate the lights.
- The *cars* dataset contains traces from two cars. Each car is equipped with three piezoelectric harvesters mounted on the windshield, the dashboard, and in the trunk. The cars drive for two hours in convoy over a variety of roads.
- The *washer* dataset includes five traces from piezoelectric harvesters mounted on a WPB4700H industrial washing machine, while the machine runs a washing program with maximum load for 45 minutes.

The data is publicly available at [Zenodo](https://doi.org/10.5281/zenodo.6383042). The data is provided as one hdf5 file per dataset containing time-synchronized power traces with a sampling rate of 100kSps and the following format:

```bash
	   .
	   ├── time			# Common time base in seconds
	   ├── data
	       ├── node0	# Power samples of node0
	       ├── node1	# Power samples of node1
	       ├── node2	# Power samples of node2
	       └── ...

```

Download the data from [Zenodo](https://doi.org/10.5281/zenodo.6383042) to `[DATA_PATH]` on your local machine. Most of the example code in this repository works with sequences of charging times. These can be computed from the provided power traces by simulating the charging behavior of a battery-free device. To convert the power traces to charging time traces of a simulated node with a capacity of 17uF, a turn-on voltage of 3V and a turn-off voltage of 2.4V, use the provided command line utility `pwr2time` that gets installed with the Python package:

```
pwr2time -i [DATA_PATH]/pwr_stairs.h5 -o [DATA_PATH]/tchrg_stairs.h5
```

The resulting hdf5 file has the following structure

```bash
.
├── (0, 1)	    # Group for 'synchronized' charging times of node0 and node1
│   ├── time	# Common timebase for the two charging time traces
│   ├── node0	# Power samples of node0
│   └── node1	# Power samples of node1
├── (0, 2)
│   ├── time
│   ├── node0	# Power samples of node0
│   └── node1	# Power samples of node2
├── ...
├── (4, 5)
│   ├── time
│   ├── node0	# Power samples of node4
│   └── node1	# Power samples of node5
└── ...

```

This conversion can take multiple hours per dataset. To spare you from the long wait, we provide pre-computed charging time traces for the default configuration (17uF, 2.4V-3V) together with the power traces. For example, `tchrg_stairs.h` contains the charging times of all possible combinations of the power traces from the dataset `pwr_stairs.h5`.


## Usage

Plot the first 10 minutes of harvesting power trace of two nodes:

```python
import h5py
import matplotlib.pyplot as plt

with h5py.File("pwr_stairs.h5", "r") as hf:
    ptimes = hf["time"][:60_000_000]
    pwr1 = hf["data"]["node0"][:60_000_000]
    pwr2 = hf["data"]["node4"][:60_000_000]

plt.plot(ptimes, pwr1)
plt.plot(ptimes, pwr2)
plt.show()
```

Convert the power traces to sequences of charging times and plot the results (this can take a few minutes):

```python
from neslab.bonito import pwr2time

ctimes, tchrg1, tchrg2 = pwr2time(ptimes, pwr1, pwr2)

plt.plot(ctimes, tchrg1)
plt.plot(ctimes, tchrg2)
plt.show()
```

Learn the parameters of a normal distribution from one of the charging time traces using stochastic gradient descent and plot the results.

```python
import numpy as np
from neslab.bonito import NormalDistribution as NrmDst

means = np.empty((len(ctimes),))

dist_model = NrmDst()
for i, c in enumerate(tchrg1):
    dist_model.sgd_update(c)
    means[i] = dist_model._mp[0]

plt.plot(ctimes, tchrg1)
plt.plot(ctimes, means)
plt.show()
```

Run the Bonito protocol on the two charging time traces and print the resulting connection interval for every successful encounter:

```python
from neslab.bonito import bonito

cis = np.empty((len(ctimes),))
for i, (ci, success) in enumerate(bonito((tchrg1, tchrg2), (NrmDst, NrmDst))):
    if success:
        cis[i] = ci
    else:
        cis[i] = np.nan

plt.plot(ctimes, tchrg1)
plt.plot(ctimes, tchrg2)
plt.plot(ctimes, cis)
plt.show()
```



## Examples

We provide more involved example scripts in the [examples](./examples) directory.

To plot the power traces of nodes 0 and 2 of the *jogging* dataset downsampled to 100Hz:

```
python examples/plot_power.py -i [DATA_PATH]/pwr_jogging.h5 -p 0 2 -s 100
```

To plot the charging times of nodes 2 and 3 of the *stairs* dataset:

```
python examples/plot_tcharge.py -i [DATA_PATH]/tchrg_stairs.h5 -p 2 3
```

To plot the histograms of the charging times of nodes 2 and 3 of the *stairs* dataset:

```
python examples/plot_tcharge.py -i [DATA_PATH]/tchrg_stairs.h5 -p 2 3 --hist
```

To learn the parameters of the charging time distribution of node 4 of the *washer* dataset:

```
python examples/learning.py -i [DATA_PATH]/tchrg_washer.h5 -n 4
```

To plot the connection interval and compare the success rate and communication delay of Bonito (with a target probability of 90%), Modest and Greedy on node 1 and 3 of the *cars* dataset:

```
python examples/protocols.py -i [DATA_PATH]/tchrg_cars.h5 -p 1 3 -t 0.9
```

