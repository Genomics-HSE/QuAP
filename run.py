#!/usr/bin/env python3

import argparse
import numpy as np

from lamom.proportions import kMoment


parser = argparse.ArgumentParser(description='...')

# input
parser.add_argument('--file', '-f', nargs=1, type=str, default='',
                    help=
'File with k1, k2 and k3 - statistics and chromosome lengths of the following format:\n\
chr_length_1\tchr_length_2\t...\tchr_length_n\n\
k1stat_1\tk1stat_2\t...\tk1stat_n\n\
k2stat_1\tk2stat_2\t...\tk2stat_n\n\
k3stat_1\tk3stat_2\t...\tk3stat_n\n')

# parameters
parser.add_argument('--Te0', '-Te0', nargs=1, type=float, default=5,
                    help='Initial point of optimization algorithm for parameter: time end')
parser.add_argument('--Td0', '-Td0', nargs=1, type=float, default=5,
                    help='Initial point of optimization algorithm for parameter: duration')
parser.add_argument('--N', '-N', nargs=1, type=float, default=10000,
                    help='Effective population size, default is 10000')



clargs = parser.parse_args()

if isinstance(clargs.file, list):
    clargs.file = clargs.file[0]
if isinstance(clargs.Te0, list):
    clargs.Te0 = clargs.Te0[0]
if isinstance(clargs.Td0, list):
    clargs.Td0 = clargs.Td0[0]
if isinstance(clargs.N, list):
    clargs.N = clargs.N[0]


print('Starting point estimate...')

sample = np.loadtxt(clargs.file)

lengths = sample[0]
sample_k1 = sample[1]
sample_k2 = sample[2]
sample_k3 = sample[3]

exp = kMoment(clargs.N)
exp.sample(sample_k1, sample_k2, sample_k3, lengths)
x = exp.estimate(x0=[clargs.Te0, clargs.Td0])

s = exp.model.get_prop_per_gen(x.x[1])
print(s, *x.x, end=' ')
print(x.cost)

estimate_k3 = exp.model.get_k3(s, *x.x)
estimate_k2 = exp.model.get_k2(s, *x.x)

print(*estimate_k2)
print(*estimate_k3)
