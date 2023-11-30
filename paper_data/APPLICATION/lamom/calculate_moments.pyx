# cython: language_level=3

cimport cython
import numpy as np
from scipy import integrate
from scipy.optimize import least_squares

from libc.math cimport exp, abs, pow, sqrt

cdef class RegEst:
    cdef:
        double[:] expectation_vector, sample_vector, loss_vector, weights

        double[:] sample_lengths

        double[:] expectation_k2, expectation_k3

        double[:, ::1] L_matrix, D_matrix, U_matrix
        double[:] eigv, v0

        double N
        double sample_k1_mean

        int populations_number

    def __init__(self, double N):
        self.v0 = np.zeros(5, dtype=float)
        self.eigv = np.array([1, -1, -1, -1, 2], dtype=float)
        self.D_matrix = np.zeros((5, 5), dtype=float)
        self.U_matrix = np.zeros((5, 5), dtype=float)
        self.N = N
        self.L_matrix = np.zeros((5, 5), dtype=float)
        self.L_matrix[0, 0] = 1.
        self.L_matrix[1, 0] = 1. / 2. / N
        self.L_matrix[1, 1] = (2.*N-1.) / 2. / N
        self.L_matrix[2, 0] = 1. / 2. / N
        self.L_matrix[2, 2] = (2.*N-1.) / 2. / N
        self.L_matrix[3, 0] = 1. / 2. / N
        self.L_matrix[3, 3] = (2.*N-1.) / 2. / N
        self.L_matrix[4, 0] = 1. / 4. / N / N
        self.L_matrix[4, 1] = (2.*N-1.) / 4. / N / N
        self.L_matrix[4, 2] = (2.*N-1.) / 4. / N / N
        self.L_matrix[4, 3] = (2.*N-1.) / 4. / N / N
        self.L_matrix[4, 4] = (2.*N-1.)*(2.*N-2.) / 4. / N / N

    def sample(self, sample_k1, sample_k2, sample_k3, sample_lengths):
        self.populations_number = sample_lengths.shape[0]
        self.expectation_vector = np.zeros(2*self.populations_number, dtype=float)
        self.loss_vector = np.zeros(2*self.populations_number, dtype=float)
        temp = np.zeros(2*self.populations_number, dtype=float)
        temp[:self.populations_number] = sample_k2
        temp[self.populations_number:] = sample_k3
        self.sample_vector = temp
        self.sample_lengths = sample_lengths
        self.sample_k1_mean = sample_k1.mean()

        self.expectation_k2 = np.zeros(self.populations_number, dtype=float)
        self.expectation_k3 = np.zeros(self.populations_number, dtype=float)

    def get_sample(self):
        return np.array(self.sample_vector), np.array(self.sample_lengths)

    def get_prop_per_gen(self, double dur):
        return self.calculate_prop_per_gen(dur)

    def get_k2(self, double s, double gs, double dur):
        self.calculate_expectation_k2(s, gs, dur)
        return np.array(self.expectation_k2)

    def get_k3(self, double s, double gs, double dur):
        self.calculate_expectation_k3(s, gs, dur)
        return np.array(self.expectation_k3)

    @cython.cdivision(True)
    cdef double calculate_prop_per_gen(self, double dur):
        return 1 - pow(self.sample_k1_mean, 1/dur)

    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef calculate_expectation_k2(self, double s, double gs, double dur):
        cdef Py_ssize_t i = 0
        for i in range(self.sample_lengths.shape[0]):
            self.expectation_k2[i] = integrate.dblquad(
                self.integrand_k2, 0, self.sample_lengths[i], lambda x: x, lambda x: self.sample_lengths[i],
                args=[s, gs, dur])[0]*pow(1-1/2./self.N, gs)
            self.expectation_k2[i] *= 2 / pow(self.sample_lengths[i], 2)

    @cython.initializedcheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef calculate_expectation_k3(self, double s, double gs, double dur):

        # cdef double[:, ::1] L_matrix = np.array([
        #         [4*N*N, 0, 0, 0, 0],
        #         [2*N, 2*N*(2*N-1), 0, 0, 0],
        #         [2*N, 0, 2*N*(2*N-1), 0, 0],
        #         [2*N, 0, 0, 2*N*(2*N-1), 0],
        #         [1, 2*N-1, 2*N-1, 2*N-1, (2*N-1)*(2*N-2)]
        #     ]) / 4 / N / N
        #
        # D = np.array([
        #     [1 - s, 0, 0, 0, 0],
        #     [0, pow(1-s, 2), 0, 0, 0],
        #     [0, 0, pow(1-s, 2), 0, 0],
        #     [0, 0, 0, pow(1-s, 2), 0],
        #     [0, 0, 0, 0, pow(1-s, 3)]
        # ])
        #
        # v0 = np.array(
        #     [1-s, pow(1-s, 2), pow(1-s, 2), pow(1-s, 2), pow(1-s, 3)]
        # )

        self.D_matrix[0, 0] = 1 - s
        self.D_matrix[1, 1] = pow(1 - s, 2)
        self.D_matrix[2, 2] = self.D_matrix[1, 1]
        self.D_matrix[3, 3] = self.D_matrix[1, 1]
        self.D_matrix[4, 4] = pow(1 - s, 3)

        self.v0[0] = self.D_matrix[0, 0]
        self.v0[1] = self.D_matrix[1, 1]
        self.v0[2] = self.D_matrix[1, 1]
        self.v0[3] = self.D_matrix[1, 1]
        self.v0[4] = self.D_matrix[4, 4]

        cdef Py_ssize_t i = 0
        cdef double one_pulse_k3_drift_multiplier = pow((1-1/2./self.N)*(1-2/2./self.N), gs)

        self.expectation_k3[0] = integrate.tplquad(
            self.integrand_k3, 0, self.sample_lengths[i], lambda x: x, lambda x: self.sample_lengths[i],
            lambda x, y: y, lambda x, y: self.sample_lengths[i],
            args=[gs, dur])[0] * one_pulse_k3_drift_multiplier

        for i in range(1, self.sample_lengths.shape[0]):
            self.expectation_k3[i] = self.expectation_k3[i-1]

            if (self.sample_lengths[i-1] != self.sample_lengths[i]):
                self.expectation_k3[i] += integrate.tplquad(
                    self.integrand_k3,
                    self.sample_lengths[i-1], self.sample_lengths[i],
                    lambda x: x, lambda x: self.sample_lengths[i],
                    lambda x, y: y, lambda x, y: self.sample_lengths[i],
                    args=[gs, dur])[0] * one_pulse_k3_drift_multiplier

                self.expectation_k3[i] += integrate.tplquad(
                    self.integrand_k3,
                    0, self.sample_lengths[i-1],
                    lambda x: x, lambda x: self.sample_lengths[i-1],
                    lambda x, y: self.sample_lengths[i-1], lambda x, y: self.sample_lengths[i],
                    args=[gs, dur])[0] * one_pulse_k3_drift_multiplier

                self.expectation_k3[i] +=  integrate.tplquad(
                    self.integrand_k3,
                    0, self.sample_lengths[i-1],
                    lambda x: self.sample_lengths[i-1], lambda x: self.sample_lengths[i],
                    lambda x, y: y, lambda x, y: self.sample_lengths[i],
                    args=[gs, dur])[0] * one_pulse_k3_drift_multiplier

            self.expectation_k3[i-1] *= 6 / pow(self.sample_lengths[i-1], 3)
        self.expectation_k3[i] *= 6 / pow(self.sample_lengths[i], 3)

    @cython.cdivision(True)
    cdef double get_coefficient_d(self, double ldl, double sg):
        return (1 - 1/2./self.N + ldl/2./self.N)*(1-sg)**2

    @cython.cdivision(True)
    cdef double get_coefficient_c(self, double ll, double sg):
        return ll*(1-sg)/2./self.N

    cdef double get_coefficient_a(self, double ll, double sg):
        return ll*(1-sg)

    cdef double get_coefficient_b(self, double ldl, double sg):
        return ldl*pow(1-sg, 2)

    cdef double get_discr(self, double a, double b, double c, double d):
        return sqrt(a*a + 4*b*c-2*a*d + d*d)

    cdef double get_root_1(self, double a, double b, double c, double d):
        return 0.5*(a+d-self.get_discr(a, b, c, d))

    cdef double get_root_2(self, double a, double b, double c, double d):
        return 0.5*(a+d+self.get_discr(a, b, c, d))

    @cython.cdivision(True)
    cdef double get_vn_2(self,
            double sg, double g, double length,
            double a, double b, double c, double d,
            double ll, double ldl, double discr, double dr,
            double r1, double r2, double v00, double v01,
            double N):
        return 0.5*(-2.*c*(dr)*v00+((a-d)*dr+discr*(r1**g+r2**g))*v01)/ discr

    @cython.cdivision(True)
    cdef double get_vn_1(self,
            double sg, double g, double length,
            double a, double b, double c, double d,
            double ll, double ldl, double discr, double dr,
            double r1, double r2, double v00, double v01,
            double N):
        return 0.5 * (discr*v00*(r1**g+r2**g) + (d - a)*dr*v00 - 2.*b*v01*dr) / discr

    @cython.cdivision(True)
    cpdef double integrand_k2(self, double l, double ls, double s, double gs, double dur):
        cdef double length = l - ls

        cdef double ll = (1. + exp(-2.*length))/2.
        cdef double ldl = (1. - exp(-2.*length))/2.

        cdef double a = self.get_coefficient_a(ll, s)
        cdef double b = self.get_coefficient_b(ldl, s)
        cdef double c = self.get_coefficient_c(ll, s)
        cdef double d = self.get_coefficient_d(ldl, s)

        cdef double discr = self.get_discr(a, b, c, d)

        cdef double r1 = self.get_root_1(a, b, c, d)
        cdef double r2 = self.get_root_2(a, b, c, d)

        cdef double v00 = 1.-s
        cdef double v01 = (1.-s)*(1.-s)

        cdef double dr = pow(r1, dur-1) - pow(r2, dur-1.)

        return (self.get_vn_1(s, dur-1., length, a, b, c, d, ll,
                       ldl, discr, dr, r1, r2, v00, v01, self.N)
              - self.get_vn_2(s, dur-1., length, a, b, c, d, ll,
                       ldl, discr, dr, r1, r2, v00, v01, self.N)) \
              * pow(ll, gs)

    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef integrand_k3(self, double l, double ls, double lss,
             double gs, double dur):

        cdef double d1 = l - ls
        cdef double d2 = l - lss
        cdef double d3 = ls - lss

        # Haldane map function
        # cdef double lls = (1. + exp(-2.*d1)) / 2.
        # cdef double ldls = (1. - exp(-2.*d1)) / 2.
        # cdef double lsdlss = (1. - exp(-2.*d3)) / 2.
        # cdef double lslss = (1. + exp(-2.*d3)) / 2.
        # cdef double llslss = lls*lslss
        # cdef double llsdlss = lls*lsdlss
        # cdef double ldlslss = ldls*lslss
        # cdef double ldlsdlss = ldls*lsdlss
        # cdef double llss = llslss + ldlsdlss
        # cdef double ldlss = 1. - (llslss + ldls*lsdlss)

        # my
        self.U_matrix[1, 1] = (1. + exp(-2.*d3)) / 2. #lslss
        self.U_matrix[1, 4] = (1. - exp(-2.*d3)) / 2. #lsdlss
        self.U_matrix[3, 3] = (1. + exp(-2.*d1)) / 2. # lls
        self.U_matrix[3, 4] = (1. - exp(-2.*d1)) / 2. # ldls
        self.U_matrix[0, 0] = self.U_matrix[3, 3]*self.U_matrix[1, 1] #llslss
        self.U_matrix[0, 1] = self.U_matrix[3, 4]*self.U_matrix[1, 1] #ldlslss
        self.U_matrix[0, 2] = self.U_matrix[3, 4]*self.U_matrix[1, 4] #ldlsdlss
        self.U_matrix[0, 3] = self.U_matrix[3, 3]*self.U_matrix[1, 4] #llsdlss
        self.U_matrix[2, 2] = self.U_matrix[0, 0] + self.U_matrix[0, 2] #llss
        self.U_matrix[2, 4] = 1. - (self.U_matrix[0, 0] + self.U_matrix[3, 4]*self.U_matrix[1, 4]) #ldlss
        self.U_matrix[4, 4] = 1.

        V, Q = np.linalg.eig(self.L_matrix @ np.matmul(self.U_matrix, self.D_matrix))
        V = np.diag(V**(dur-1))
        Q_inv = np.linalg.solve(Q, np.eye(5))
        return  (self.eigv @ Q) @ V @ (Q_inv @ self.v0) * pow(self.U_matrix[0, 0], gs)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef double[:] calculate_loss_vector(self, double[:] par):
        cdef Py_ssize_t i = 0
        print(par[0], par[1])
        cdef double s = self.calculate_prop_per_gen(par[1])
        self.calculate_expectation_k2(s, par[0], par[1])
        self.calculate_expectation_k3(s, par[0], par[1])
        for i in range(self.populations_number):
            self.loss_vector[i] = self.sample_vector[i] - self.expectation_k2[i]
        for i in range(self.populations_number):
            self.loss_vector[i+self.populations_number] = self.sample_vector[i+self.populations_number] - self.expectation_k3[i]
        return np.array(self.loss_vector)*np.array(self.weights)


    def estimate(self, x0=np.array([5, 5]), weights=[1]):
        if len(weights) == 1:
            self.weights = np.array(weights*self.populations_number*2, dtype=float)
        else:
            self.weights = np.array(weights, dtype=float)
        x = least_squares(self.calculate_loss_vector, x0,
                          bounds=([0, 0.05], [np.inf, np.inf]))
        s = self.calculate_prop_per_gen(x.x[0])
        return x


#
# @cython.wraparound(False)
# @cython.boundscheck(False)
# cdef set_expectation_vector(double s, double gs,
#     double dur, double[:] lengths, double N):
#
#     cdef int i = 0
#
#     for i in range(lengths.shape[0]):
#         expectation_vector[i] = expectation_k1(s, dur)
#     for i in range(lengths.shape[0]):
#         expectation_vector[lengths.shape[0]+i] = expectation_k2(s, gs, dur, lengths[i], N)
#     for i in range(lengths.shape[0]):
#         expectation_vector[2*lengths.shape[0]+i] = expectation_k3(s, gs, dur, lengths[i], N)
#
# cdef double expectation_k1(double sg, double dur):
#     return 1 - pow(1 - sg, dur)
#
# @cython.cdivision(True)
# @cython.wraparound(False)
# @cython.boundscheck(False)
# cdef double expectation_k2(double s, double gs, double dur, double length, double N):
#     cdef double k2 = 0
#     cdef double L = 0
#     cdef Py_ssize_t i = 0
#     k2 = integrate.dblquad(
#         fun, 0, length, lambda x: x, lambda x: length,
#         args=[s, gs, dur, N])[0]*(1-1/2/N)**gs
#     k2 *= 2
#     return k2
#
# @cython.cdivision(True)
# @cython.wraparound(False)
# @cython.boundscheck(False)
# cdef double expectation_k3(double s, double gs, double dur, double length, double N):
#
#     L_matrix[0, 0] = 1.
#     L_matrix[1, 0] = 1. / 2. / N
#     L_matrix[1, 1] = (2.*N-1.) / 2. / N
#     L_matrix[2, 0] = 1. / 2. / N
#     L_matrix[2, 2] = (2.*N-1.) / 2. / N
#     L_matrix[3, 0] = 1. / 2. / N
#     L_matrix[3, 3] = (2.*N-1.) / 2. / N
#     L_matrix[4, 0] = 1. / 4. / N / N
#     L_matrix[4, 1] = (2.*N-1.) / 4. / N / N
#     L_matrix[4, 2] = (2.*N-1.) / 4. / N / N
#     L_matrix[4, 3] = (2.*N-1.) / 4. / N / N
#     L_matrix[4, 4] = (2.*N-1.)*(2.*N-2.) / 4. / N / N
#
#     D[0, 0] = 1 - s
#     D[1, 1] = pow(1 - s, 2)
#     D[2, 2] = pow(1 - s, 2)
#     D[3, 3] = pow(1 - s, 2)
#     D[4, 4] = pow(1 - s, 3)
#
#     v0[0] = 1 - s
#     v0[1] = pow(1 - s, 2)
#     v0[2] = pow(1 - s, 2)
#     v0[3] = pow(1 - s, 2)
#     v0[4] = pow(1 - s, 3)
#
#     cdef double k3 = 0
#     cdef Py_ssize_t i = 0
#
#     k3 = (-1)*integrate.tplquad(
#         fun3, 0, length, lambda x: x, lambda x: length,
#         lambda x, y: y, lambda x, y: length,
#         args=[gs, dur])[0] * pow((1-1/2./N)*(1-2./2./N), gs)
#     k3 *= 6.
#     return k3
#
# def getk1(double sg, double dur):
#     return 1 - pow(1 - sg, dur)
#
# def get_prop_per_gen(double k1, double dur):
#     return 1 - pow(1 - k1, 1/dur)
#
# def getk2(s, gs, dur, double[:] lengths, N=1000):
#     cdef double k2 = 0
#     cdef double L = 0
#     cdef Py_ssize_t i = 0
#     for i in range(lengths.shape[0]):
#         k2 += integrate.dblquad(
#             fun, 0, lengths[i], lambda x: x, lambda x: lengths[i],
#             args=[s, gs, dur, N])[0]*(1-1/2/N)**gs
#         L += lengths[i]
#     k2 /= L**2
#     k2 *= 2
#     return k2
#
#
#
#
#
# cdef double geta(double N, double ll, double ldl, double sg):
#     return ll*(1-sg)
#
# cdef double getb(double N, double ll, double ldl, double sg):
#     return ldl*pow(1-sg, 2)
#
# @cython.cdivision(True)
# cdef double getc(double N, double ll, double ldl, double sg):
#     return ll*(1-sg)/2./N
#
# @cython.cdivision(True)
# cdef double getd(double N, double ll, double ldl, double sg):
#     return (1 - 1/2./N + ldl/2./N)*(1-sg)**2
#
# cdef double getdiscr(double a, double b, double c, double d):
#     return sqrt(a*a + 4*b*c-2*a*d + d*d)
#
# cdef double getroot1(double a, double b, double c, double d):
#     return 0.5*(a+d-getdiscr(a, b, c, d))
#
# cdef double getroot2(double a, double b, double c, double d):
#     return 0.5*(a+d+getdiscr(a, b, c, d))
#
# @cython.cdivision(True)
# cdef double getvn2(
#         double sg, double g, double length,
#         double a, double b, double c, double d,
#         double ll, double ldl, double discr, double dr,
#         double r1, double r2, double v00, double v01,
#         double N):
#     return 0.5*(-2*c*(dr)*v00+((a-d)*dr+discr*(r1**g+r2**g))*v01)/ discr
#
# @cython.cdivision(True)
# cdef double getvn1(
#         double sg, double g, double length,
#         double a, double b, double c, double d,
#         double ll, double ldl, double discr, double dr,
#         double r1, double r2, double v00, double v01,
#         double N):
#     return 0.5 * (discr*v00*(r1**g+r2**g) + (d - a)*dr*v00 - 2*b*v01*dr) / discr
#
#
# # cdef int findchr(double pos, double[:] lengths):
# #     cdef Py_ssize_t i = 0
# #     cdef int length = lengths[0]
# #     while length < pos:
# #         i += 1
# #         length += lengths[i]
# #     return i
#
#
# cdef double fun(double l, double ls, double s, double gs, double dur, double N):
#     length = l - ls
#
#     cdef double ll = (1 + exp(-2*length))/2
#     cdef double ldl = (1 - exp(-2*length))/2
#
#     cdef double a = geta(N, ll, ldl, s)
#     cdef double b = getb(N, ll, ldl, s)
#     cdef double c = getc(N, ll, ldl, s)
#     cdef double d = getd(N, ll, ldl, s)
#
#     cdef double discr = getdiscr(a, b, c, d)
#
#     cdef double r1 = getroot1(a, b, c, d)
#     cdef double r2 = getroot2(a, b, c, d)
#
#     cdef double v00 = 1-s
#     cdef double v01 = (1-s)*(1-s)
#
#     cdef double dr = pow(r1, dur-1) - pow(r2, dur-1)
#
#     return (getvn1(s, dur-1, length, a, b, c, d, ll,
#                    ldl, discr, dr, r1, r2, v00, v01, N)
#           - getvn2(s, dur-1, length, a, b, c, d, ll,
#                    ldl, discr, dr, r1, r2, v00, v01, N)) \
#           * pow(ll, gs)
# cdef double fun(double l, double ls, double s, double gs, double dur):
#     length = abs(l - ls)
#     return (getvn1(s, dur, length) - getvn2(s, dur, length)) * pow(0.5 * (1 + exp(-2*length)), gs)
