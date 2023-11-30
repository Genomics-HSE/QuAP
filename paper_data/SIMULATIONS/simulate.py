import sys
import msprime
from scipy.stats import kstat
import numpy as np

from lamom.proportions import kMoment
from sim import const_gen_flow

T_start = int(sys.argv[1])
duration = int(sys.argv[2])
seed = int(sys.argv[3])


total_s = 0.3
replicate_num = 22

lengths=np.linspace(1, 2, sim_num)

sample_k1 = np.zeros(sim_num)
sample_k2 = np.zeros(sim_num)
sample_k3 = np.zeros(sim_num)

for sim_i in range(replicate_num):
    reps = const_gen_flow(g_start=T_start + 1, g_end=T_start+duration, total_s=total_s, # +1 is for index
                        seed=T_start+duration + 1 + seed + sim_i, N_haploid = [1000, 1000, 1000],
                        num_replicates=1, length_m=lengths[sim_i])
    exp = kMoment(1000)
    exp.sample_from_simulations(lengths[sim_i], reps=reps, time=T_start+duration+1)
    sample_k1[sim_i] = exp.sample_k1[0]
    sample_k2[sim_i] = exp.sample_k2[0]
    sample_k3[sim_i] = exp.sample_k3[0]

print(*sample_k1)
print(*sample_k2)
print(*sample_k3)
