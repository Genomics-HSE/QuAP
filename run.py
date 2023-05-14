#!/usr/bin/env python3

import argparse

from lamom.proportions import kMoment


parser = argparse.ArgumentParser(description='...')

#brackets
parser.add_argument('--k2', '-k2', nargs=1, type=float, default=0,
                    help='Obtained k2 statistics')
parser.add_argument('--k3', '-k3', nargs=1, type=float, default=0,
                    help='Obtained k3 statistics')

#parameters
parser.add_argument('--T', '-T', nargs=1, type=float, default=None,
                    help='Time to the admixture')
parser.add_argument('--Td', '-d', nargs=1, type=float, default=None,
                    help='The duration of admixture')
parser.add_argument('--s', '-s', nargs=1, type=float, default=None,
                    help='Total admixture proportion of the first ancestral population.')
parser.add_argument('--N', '-N', nargs=1, type=float, default=1000,
                    help='Effective population size, default is 1000')

parser.add_argument('--lengths', '-L', default=[], nargs='+',
                    help='Length of chromosome in M., for multiple chromosomes put several values (default: 1)')


clargs = parser.parse_args()

if isinstance(clargs.k2, list):
    clargs.k2 = clargs.k2[0]
if isinstance(clargs.k3, list):
    clargs.k3 = clargs.k3[0]
if isinstance(clargs.T, list):
    clargs.T = clargs.T[0]
if isinstance(clargs.Td, list):
    clargs.Td = clargs.Td[0]
if isinstance(clargs.s, list):
    clargs.s = clargs.s[0]
if isinstance(clargs.N, list):
    clargs.N = clargs.N[0]


print('Starting point estimate...')

exp = kMoment(clargs.N, lengths=clargs.lengths)
exp.set_k([clargs.k1], [clargs.k2], [clargs.k3], clargs.lengths)
exp.estimate(x0=[5, 5], batchsize=0, silence=False)
