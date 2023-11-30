import sys

from lamom.proportions import kMoment
import numpy as np

file = str(sys.argv[1])
x0 = [1.752574607889148, 6.921785584579151]
sample = np.loadtxt(file)

sample_k1 = sample[0]
sample_k2 = sample[1]
sample_k3 = sample[2]
lengths = np.array([2.86279234, 2.68839622, 2.23361095, 2.14667992118, 2.04089357,
                    1.920175851864947, 1.872205, 1.68003442, 1.66359329, 1.81144,
                    1.5821865, 1.74679023, 1.25706316, 1.20202583, 1.41730338572,
                    1.34037726, 1.28490529, 1.17708923, 1.07733846, 1.08266934,
                    0.62786478, 0.74109562])

exp = kMoment(10000)
exp.sample(sample_k1, sample_k2, sample_k3, lengths)
x = exp.estimate(x0=x0)

s = exp.model.get_prop_per_gen(x.x[1])
print(s, *x.x, end=' ')
print(x.cost)

estimate_k3 = exp.model.get_k3(s, *x.x)
estimate_k2 = exp.model.get_k2(s, *x.x)

print(*estimate_k2)
print(*estimate_k3)
