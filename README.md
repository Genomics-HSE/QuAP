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
To run this method k-statistics k1, k2 and k3 of admixture proportions for each chromosome of
population that admixed only in founding generation are required.
To estimate parameters simply run

```
python run.py --file data.txt
```

You may test this script using `dummy_data.txt` file, that included in the repository. True parameters for this file are `Te=0` and `Td=10`.

# Parameters
`-N` specifies an effective population size.
`-Te0` and `-Td0` sets initial point for the optimisation algorithm.
