import msprime
import functools


def one_pulse(g=15, s0=0.5, sample_sizes=[100, 100, 100], mu=1.25e-8, rho = 1.6e-9,
              num_replicates=1, N_haploid = [1000, 1000, 1000], length_m = 1,
              seed=1, sim_mut=False):
    length=int(lenght_m/rho)

    dem = msprime.Demography()
    dem.add_population(name='A', initial_size=N_haploid[0])
    dem.add_population(name='B', initial_size=N_haploid[1])
    dem.add_population(name='C', initial_size=N_haploid[2])
    dem.add_population(name='old', initial_size=1000)

    dem.add_admixture(time=g + 1, ancestral=["A","B"], derived="C", proportions=[1-s0, s0])
    dem.add_population_split(time=5000, derived=["A", "B"], ancestral="old")
    dem.sort_events()

    reps = msprime.sim_ancestry(
        samples={"A": sample_sizes[0], "B": sample_sizes[1], "C": sample_sizes[2]},
        demography=dem, sequence_length = length, recombination_rate=rho, ploidy=2,
        #model=[msprime.DiscreteTimeWrightFisher(duration=50), msprime.StandardCoalescent(duration=3950)],
        #model=[msprime.DiscreteTimeWrightFisher(duration=3950), msprime.StandardCoalescent(duration=50)],
        #model=msprime.DiscreteTimeWrightFisher(),
        #model=msprime.StandardCoalescent(),
        model='dtwf', num_replicates=num_replicates,
        random_seed=seed)
    if sim_mut:
        mutations_func = functools.partial(msprime.sim_mutations, rate=mu)
        reps = map(mutations_func, reps)
    #mts = msprime.sim_mutations(ts, rate=mu, random_seed=seed)
    return reps

def const_gen_flow(g_start=2, g_end=11, sg=None, Tdiv=4000, total_s = 0.2, num_replicates=1,
                   sample_sizes=[100, 100, 100], N_haploid = [1000, 1000, 1000],
                   mu=1.25e-8, rho = 1.6e-9, length_m = 1, seed=1, sim_mut=False):
    length=int(length_m/rho)

    # print('msprime seed:', seed)

    if sg == None:
        dur = g_end - g_start + 1
        sg = 1 - (1 - total_s)**(1/dur)

    dem = msprime.Demography()
    dem.add_population(name='A', initial_size=N_haploid[0])
    dem.add_population(name='B', initial_size=N_haploid[1])
    dem.add_population(name='C', initial_size=N_haploid[2])
    dem.add_population(name='old', initial_size=1000)

    dem.add_migration_rate_change(time=g_start, rate=sg, source='C', dest='B')
    dem.add_migration_rate_change(time=g_end, rate=0, source='C', dest='B')
    dem.add_admixture(time=g_end, ancestral=["A","B"], derived="C", proportions=[1-sg, sg])
    dem.add_population_split(time=Tdiv, derived=["A", "B"], ancestral="old")
    dem.sort_events()

    reps = msprime.sim_ancestry(
        samples={"A": sample_sizes[0], "B": sample_sizes[1], "C": sample_sizes[2]},
        demography=dem, sequence_length = length, recombination_rate=rho, ploidy=2,
        #model=[msprime.DiscreteTimeWrightFisher(duration=50), msprime.StandardCoalescent(duration=3950)],
        #model=[msprime.DiscreteTimeWrightFisher(duration=3950), msprime.StandardCoalescent(duration=50)],
        #model=msprime.DiscreteTimeWrightFisher(),
        #model=msprime.StandardCoalescent(),
        model='dtwf', num_replicates=num_replicates,
        random_seed=seed)
    if sim_mut:
        mutations_func = functools.partial(msprime.sim_mutations, rate=mu)
        reps = map(mutations_func, reps)
    return reps
