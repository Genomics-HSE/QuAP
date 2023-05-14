# inference-adxmiture-fraction
Method prototype for estimating admixture parameters using admixture fractions.

# Installation

For performance reasons, we use cython to speed up calculations. For this, you need a working toolchain for building C
code (gcc and clang are known to work). Since you are going to build Python extensions, you will need python development headers (e.g. on ubuntu linux the package name is python-dev).

To build the package use

```
$ cd lamom
$ python setup.py build_ext -i
```


## Quick start
To run this method k-statistics k2 and k3 of admixture proportions are required.
To estimate parameters simply run

```
python run.py -s 0.2 -k2 0.0025 -k3 0.00035 -N 1000 -L 1
```

# Parameters
`-N` specifies an effective population size.

`-L` chromosome lengths. If several chromosomes are used put length for each one.

`-k2` k2 statistics.

`-k3` k3 statistics.

`-s` total admixture proportion of the population that admixed (or k1 statistics)
