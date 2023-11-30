All simulations were performed using msprime.

- sim.py contains functions to simulate demographic scenarios
- simulate.py contains script that simulates concrete scenario and estimate k-stat.
- estimate_simulation.py contains script that estimate parameters from k-stat.

- /simulations_kstat contains all k-statistics from simulated populations.
The meaning of a filename in that directory:
s{T_e}d{T_d}_{simulation_id}.log
s{T_e}d{T_d}_{simulation_id}.err

- /simulations_estimate contains estimates for files from /simulations_kstat
The meaning of a filename is the same as in /simulations_kstat.
