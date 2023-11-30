import numpy as np
from scipy.stats import kstat
from scipy.optimize import least_squares
from scipy import integrate

from lamom.calcmom import RegEst


class kMoment:
    def __init__(self, N=1000):
        self.N = N
        self.model = RegEst(N)

    def estimate(self, x0=[5, 5], weights=[1]):
        return self.model.estimate(np.array(x0, dtype=float), weights=weights)

    def sample_from_simulations(self, lengths, reps=None, time=50):
        self.lengths = np.array([lengths], dtype=float)
        if type(lengths) == list:
            self.lengths = np.array(lengths, dtype=float)
        self.sample_k1, self.sample_k2, self.sample_k3, self.sample_k4 = \
            self.get_admixture_moments(reps, 2, 1, time=time)
        self.lengths = np.array([1]*len(self.sample_k1), dtype=float)
        self.model.sample(self.sample_k1, self.sample_k2, self.sample_k3, self.lengths)

    def sample(self, sample_k1, sample_k2, sample_k3, lengths):

        sample_k1 = np.array(sample_k1, dtype=float)
        sample_k2 = np.array(sample_k2, dtype=float)
        sample_k3 = np.array(sample_k3, dtype=float)
        lengths = np.array(lengths, dtype=float)

        perm = lengths.argsort()
        sample_k1 = sample_k1[perm]
        sample_k2 = sample_k2[perm]
        sample_k3 = sample_k3[perm]
        lengths = lengths[perm]

        print('Your k-statistics (sorted by length):')
        print(f'Length (M.)\tk_1\tk_2\tk_3')
        for i in range(lengths.shape[0]):
            print(f'{lengths[i]:.2f}\t{sample_k1[i]:.8f}\t{sample_k2[i]:.8f}\t{sample_k3[i]:.8f}')

        self.model.sample(sample_k1, sample_k2, sample_k3, lengths)
    # msprime
    def get_individuals_nodes(self, ts, population, sampled=False):
        nodes = [] # nodes of samples
        for node in ts.nodes(): # they are contained in return of ts.nodes()
            if node.population == population: # population that we need
                if not sampled or node.individual != -1: # TRUE/FALSE + sampled_conditon = TRUE/sampled_conditon
                    nodes.append(node.id)
        return nodes

    def get_admixture_proportions(self, ts, admixed_population, source_population, maxtime):
            adm_nodes = self.get_individuals_nodes(ts, admixed_population, sampled=True)
            props = np.zeros(len(adm_nodes))
            LA = np.zeros((len(adm_nodes), ts.num_trees))
            segment_length = np.zeros(ts.num_trees)
            for tree_i, tree in enumerate(ts.trees()):
                segment_length[tree_i] = tree.interval[1] - tree.interval[0]
                for sample_i, node in enumerate(adm_nodes):
                    parent = tree.parent(node)
                    while tree.time(node) < maxtime and parent != -1:
                        node = parent
                        parent = tree.parent(node)
                    LA[sample_i, tree_i] = tree.population(node) # 0 - first, 1 - second src
                props[:] += segment_length[tree_i]*LA[:, tree_i]
            return props / ts.sequence_length

    def get_admixture_moments(self, replicates, admixed_population, source_population,
                                length_m=1, rho=1.6e-9, time=50):

        k1_list = []
        k2_list = []
        k3_list = []
        k4_list = []
        for replicate_i, ts in enumerate(replicates):
            props = self.get_admixture_proportions(ts, admixed_population,
                source_population, time)
            k1_list.append(kstat(props, 1))
            k2_list.append(kstat(props, 2))
            k3_list.append(kstat(props, 3))
            k4_list.append(kstat(props, 4))
        return np.array(k1_list, dtype=float), np.array(k2_list, dtype=float), \
               np.array(k3_list, dtype=float), np.array(k4_list, dtype=float)

    def set_k(self, k1_list, k2_list, k3_list, lengths):
        self.sample_k1 = np.array(k1_list, dtype=float)
        self.sample_k2 = np.array(k2_list, dtype=float)
        self.sample_k3 = np.array(k3_list, dtype=float)
        self.lengths = np.array(lengths, dtype=float)

    def estimate_old(self, silence=True, x0=[5, 5], batchsize=0):

        if batchsize != 0:
            dtype = [('k1', float), ('k2', float), ('k3', float), ('length', float)]
            values = [(self.k1_list[i],
                       self.k2_list[i],
                       self.k3_list[i],
                       self.lengths[i]) for i in range(len(self.lengths))]
            a = np.array(values, dtype=dtype)
            b = np.sort(a, order='length')
            r = [[b[i][0], b[i][1], b[i][2], b[i][3]] for i in range(len(self.lengths))]
            r = np.array(r, dtype=float)
            k1_sampled_batch, k2_sampled_batch, k3_sampled_batch, sampled_batch_lengths \
                = self.make_batch(
                    r[:, 0], r[:, 1], r[:, 2], r[:, 3],
                    batchsize=batchsize)
        else:
            k1_sampled_batch, k2_sampled_batch, k3_sampled_batch, sampled_batch_lengths \
                = self.make_batch(
                    self.k1_list, self.k2_list, self.k3_list, self.lengths,
                    batchsize=batchsize)

        self.res = np.zeros((len(k1_sampled_batch), 4))
        for i, (k1, k2, k3, lengths) in enumerate(zip(k1_sampled_batch,
                                                      k2_sampled_batch,
                                                      k3_sampled_batch,
                                                      sampled_batch_lengths)):
            tmp = self.estimate_cont(k1, k2, k3, lengths, silence=silence, x0=x0)
            self.res[i] = tmp

        print('estimation results:')
        print('batch_i\tT_start\tT_end\ts\tcost_fun_val')
        for i, tmp in enumerate(self.res):
            print(f'{i}\t{tmp[1]}\t{tmp[2]}\t{tmp[0]}\t{tmp[3]}')
        if len(self.res) > 1:
            print(f'mean\t{self.res[:, 1].mean()}\t{self.res[:, 2].mean()}\t{self.res[:, 0].mean()}\t{self.res[:, 3].mean()}')
            print(f'var\t{self.res[:, 1].var()}\t{self.res[:, 2].var()}\t{self.res[:, 0].var()}\t{self.res[:, 3].var()}')


    def init_matrixes(self, n, s0): #need to update, legacy
        M4 = np.array([
            [n**2 - 6*n + 6, -2*n, -2*n, -2*n],
            [n - 1, n**2 - n, n, n],
            [n - 1, n, n**2 - n, n],
            [n - 1, n, n, n**2 - n]
        ]) * (n-1) / n**3

        v0 = np.array([1-s0,
                      (1-s0)**2, (1-s0)**2, (1-s0)**2, (1-s0)**2, (1-s0)**2, (1-s0)**2, (1-s0)**2,
                      (1-s0)**3, (1-s0)**3, (1-s0)**3, (1-s0)**3, (1-s0)**3, (1-s0)**3,
                      (1-s0)**4])
        y4 = [1, -1, -1, -1, -1, -1, -1, -1, 2, 2, 2, 2, 2, 2, -6]
        y41 = [0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, -1, 1]
        y42 = [0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, -1, 0, 1]
        y43 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, -1, 0, 0, 1]
        y = np.array([y4, y41, y42, y43], dtype=np.float64)
        return M4, v0, y

    def LD4(self, a, b, c, d, g, *args): #need to update, legacy
        R4 = np.array([
            [np.exp(-2*(np.abs(a-b) + np.abs(b-c) + np.abs(c-d))), 0, 0, 0],
            [0, np.exp(-2*(np.abs(a-b)+np.abs(c-d))), 0, 0],
            [0, 0, np.exp(-2*(np.abs(a-c)+np.abs(b-d))), 0],
            [0, 0, 0, np.exp(-2*(np.abs(a-d)+np.abs(c-b)))]])
        y = args[2]
        v0 = args[1]
        M4 = args[0]
        w0 = y @ v0
        wg = w0
        for i in range(g):
            wg = M4@ (R4 @ w0)
        ld4 = wg[0]
        return ld4

    def eq_k2(self, s, g, L=1, N=10000):
        def integrand(l, g, L):
            return (L-l) * (0.5 + 0.5*np.exp(-2*l))**g
        return 2 * integrate.quad(integrand, 0, L, args=[g, L])[0] * (1-1/2/N)**g * (s - s*s)

    def eq_2(self, g, s, N):
        return (1 - 1 / (2*N))**g * (s - s**2) * 2 / (2 + g)

    def metr_one_pulse(self, g):
        return self.eq_2(g, self.k1_sampled, self.N) - self.k2_sampled

    def metr_cont(self, par):
        prop_per_gen = get_prop_per_gen(self.sample_k1_mean, par[1])
        print(prop_per_gen, par)
        return (
            getk2(prop_per_gen, par[0], par[1],
                  self.lengths, N=self.N) - self.sample_k2_mean,
            getk3(prop_per_gen, par[0], par[1],
                  self.lengths, N=self.N) - self.sample_k3_mean
            )

    def metr_cont_start_end(self, par):
        prop_per_gen = get_prop_per_gen(self.sample_k1, par[1]-par[0]+1)
        if not self.silence:
            print(par)
        return (
            getk2(prop_per_gen, par[0], par[1]-par[0]+1,
                  self.lengths_sampled, N=self.N) - self.k2_sampled,
            getk3(prop_per_gen, par[0], par[1]-par[0]+1,
                  self.lengths_sampled, N=self.N) - self.k3_sampled
            )

    def metr_cont_3(self, par):
        k1 = getk1(par[0], par[2])
        k2 = getk2(prop_per_gen, par[0], par[1], L=self.L, N=self.N)
        k3 = getk3(prop_per_gen, par[0], par[1], L=self.L, N=self.N)
        if not self.silence:
            print(par, k1, k2, k3)
        return (k1 - self.k1_sampled,
                k2 - self.k2_sampled,
                k3 - self.k3_sampled)

    def estimate_one_pulse(self, x0=[5]):
        x = least_squares(self.metr_one_pulse, x0)
        self.res = x
        return x.x

    def estimate_cont(self, x0=[5, 5], opb=False):
        """
            return:
                s - int, prop per gen
                T_start - int, generation, when admixture started
                T_end - int, generation, when admixture ended
                cost - int, value of cost function
        """
        self.sample_k1_mean = self.sample_k1.mean()
        self.sample_k2_mean = self.sample_k2.mean()
        self.sample_k3_mean = self.sample_k3.mean()

        if opb: # dur > opt - g_start + 1
            optime = self.estimate_one_pulse()[0]
            print(optime)
            x0 = [optime*0.5, optime*1.5]
            x = least_squares(self.metr_cont_start_end, x0,
                              bounds=([0, optime], [optime, np.inf]))
            T_end = x.x[1]
            duration = x.x[1] - x.x[0] + 1
        else:
            bounds = ([0, 0.05], [np.inf, np.inf])
            x = least_squares(self.metr_cont, x0,
                              bounds=([0, 0.05], [np.inf, np.inf]))
            T_end = x.x[1] + x.x[0] - 1
            duration = x.x[1]
        T_start = x.x[0]
        return get_prop_per_gen(self.sample_k1, duration), T_start, T_end, x.cost

    def metr_cont_discr_1(self, T_start, dur, ppg):
        T_start = T_start[0]
        k3 = getk3(ppg, T_start, dur, L=self.L, N=self.N)
        return (k3 - k3_sampled)

    def metr_cont_discr_2(self, dur, T_start):
        ppg = get_prop_per_gen(k1_sampled, dur)
        k2 = getk2(ppg, T_start, dur[0], L=self.L, N=self.N)
        return (k2 - k2_sampled)

    def estimate_cont_discr(self, n_it=20, T_start0=10, dur0=5):
        estimates = []
        costs = []
        T_start_next_next = T_start0
        dur_next = dur0
        it = 0
        while it < n_it:
            ores = least_squares(self.metr_cont_discr_2, [dur_next],
                              bounds=(0, np.inf), args=[T_start_next_next])
            dur_next = ores.x[0]
            dur_next_cost = ores.cost
            print([T_start_next_next, dur_next])
            ppg = get_prop_per_gen(k1_sampled, dur_next)
            ores = least_squares(self.metr_cont_discr_1, [T_start_next_next],
                              bounds=(0, np.inf), args=[dur_next, ppg])
            T_start_next_next = ores.x[0]
            T_start_next_next_cost = ores.cost
            estimates.append([dur_next, T_start_next_next])
            costs.append([dur_next_cost, T_start_next_next_cost])
            it+=1
        return estimates, costs

    def moments_equations(self, par):
        set_expectation_vector(par[0], par[1], par[2], self.lengths,
            self.expectation_vector, N=self.N)
        print(par)
        return ((self.expectation_vector - self.sample_vector) * self.weights)

    def moments_equations_fixed_k1(self, par):
        s = get_prop_per_gen(self.sample_k1.mean(), par[1])
        set_expectation_vector(s, par[0], par[1], self.lengths,
            self.expectation_vector, N=self.N)
        print(par)
        return ((self.expectation_vector - self.sample_vector) * self.weights)[self.lengths.shape[0]:]

    def regression_estimator(self):
        self.sample_vector = np.zeros(3*self.lengths.shape[0])
        self.sample_vector[:self.lengths.shape[0]] = self.sample_k1
        self.sample_vector[self.lengths.shape[0]: 2*self.lengths.shape[0]] = self.sample_k2
        self.sample_vector[2*self.lengths.shape[0]: 3*self.lengths.shape[0]] = self.sample_k3
        if not fix_k1:
            x0 = [get_prop_per_gen(self.sample_k1.mean(), 5), 5, 5]
            x = least_squares(self.moments_equations, x0,
                              bounds=([0, 0, 0], [max(self.sample_k1), np.inf, np.inf]))
            self.res = x
            return x.x[0], x.x[1], x.x[2]
        x0 = [5, 5]
        x = least_squares(self.moments_equations_fixed_k1, x0,
                          bounds=([0, 0], [np.inf, np.inf]))
        self.res = x
        s = get_prop_per_gen(self.sample_k1.mean(), x.x[1])
        return s, x.x[0], x.x[1]
