import sys

from lamom.proportions import kMoment
import numpy as np

file = str(sys.argv[1])
seed = int(sys.argv[2])

rng = np.random.default_rng(seed=seed)
x0 = rng.uniform(1, 20, size=2)

sample = np.loadtxt(file)

sample_k1 = sample[0]
sample_k2 = sample[1]
sample_k3 = sample[2]
lengths=np.linspace(1, 2, len(sample_k1))

exp = kMoment(1000)
exp.sample(sample_k1, sample_k2, sample_k3, lengths)
x = exp.estimate(x0=x0)

s = exp.model.get_prop_per_gen(x.x[1])
print(s, *x.x, end=' ')
print(x.cost)

estimate_k3 = exp.model.get_k3(s, *x.x)
estimate_k2 = exp.model.get_k2(s, *x.x)

print(*estimate_k2)
print(*estimate_k3)
