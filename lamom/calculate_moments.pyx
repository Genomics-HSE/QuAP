# cython: language_level=3

cimport cython
import numpy as np
from scipy import integrate

from libc.math cimport exp, abs, pow, sqrt


def getk1(double sg, double dur):
    return 1 - pow(1 - sg, dur)

def get_prop_per_gen(double k1, double dur):
    return 1 - pow(1 - k1, 1/dur)

def getk2(s, gs, dur, double[:] lengths, N=1000):
    cdef double k2 = 0
    cdef double L = 0
    cdef Py_ssize_t i = 0
    for i in range(lengths.shape[0]):
        k2 += integrate.dblquad(
            fun, 0, lengths[i], lambda x: x, lambda x: lengths[i],
            args=[s, gs, dur, N])[0]*(1-1/2/N)**gs
        L += lengths[i]
    k2 /= L**2
    k2 *= 2
    return k2


def getk3(s, gs, dur, double[:] lengths, N=1000):

    cdef double[:, ::1] L_matrix = np.array([
            [4*N*N, 0, 0, 0, 0],
            [2*N, 2*N*(2*N-1), 0, 0, 0],
            [2*N, 0, 2*N*(2*N-1), 0, 0],
            [2*N, 0, 0, 2*N*(2*N-1), 0],
            [1, 2*N-1, 2*N-1, 2*N-1, (2*N-1)*(2*N-2)]
        ]) / 4 / N / N

    D = np.array([
        [1 - s, 0, 0, 0, 0],
        [0, pow(1-s, 2), 0, 0, 0],
        [0, 0, pow(1-s, 2), 0, 0],
        [0, 0, 0, pow(1-s, 2), 0],
        [0, 0, 0, 0, pow(1-s, 3)]
    ])

    v0 = np.array(
        [1-s, pow(1-s, 2), pow(1-s, 2), pow(1-s, 2), pow(1-s, 3)]
    )

    cdef double k3 = 0
    cdef double L = 0
    cdef Py_ssize_t i = 0
    cdef double[:] k3_list = np.zeros(len(lengths))

    k3_list[0] = (-1)*integrate.tplquad(
        fun3, 0, lengths[i], lambda x: x, lambda x: lengths[i],
        lambda x, y: y, lambda x, y: lengths[i],
        args=[gs, dur, L_matrix, D, v0])[0] * pow((1-1/2/N)*(1-2/2/N), gs)
    k3 += k3_list[0]
    L += lengths[0]

    for i in range(1, lengths.shape[0]):
        k3_list[i] = k3_list[i-1] + (-1)*integrate.tplquad(
            fun3, lengths[i-1], lengths[i], lambda x: x, lambda x: lengths[i],
            lambda x, y: y, lambda x, y: lengths[i],
            args=[gs, dur, L_matrix, D, v0])[0] * pow((1-1/2/N)*(1-2/2/N), gs)
        k3 += k3_list[i]
        L += lengths[i]

    k3 /= L**3
    k3 *= 6
    return k3


cdef double geta(double N, double ll, double ldl, double sg):
    return ll*(1-sg)

cdef double getb(double N, double ll, double ldl, double sg):
    return ldl*pow(1-sg, 2)

@cython.cdivision(True)
cdef double getc(double N, double ll, double ldl, double sg):
    return ll*(1-sg)/2./N

@cython.cdivision(True)
cdef double getd(double N, double ll, double ldl, double sg):
    return (1 - 1/2./N + ldl/2./N)*(1-sg)**2

cdef double getdiscr(double a, double b, double c, double d):
    return sqrt(a*a + 4*b*c-2*a*d + d*d)

cdef double getroot1(double a, double b, double c, double d):
    return 0.5*(a+d-getdiscr(a, b, c, d))

cdef double getroot2(double a, double b, double c, double d):
    return 0.5*(a+d+getdiscr(a, b, c, d))

@cython.cdivision(True)
cdef double getvn2(
        double sg, double g, double length,
        double a, double b, double c, double d,
        double ll, double ldl, double discr, double dr,
        double r1, double r2, double v00, double v01,
        double N):
    return 0.5*(-2*c*(dr)*v00+((a-d)*dr+discr*(r1**g+r2**g))*v01)/ discr

@cython.cdivision(True)
cdef double getvn1(
        double sg, double g, double length,
        double a, double b, double c, double d,
        double ll, double ldl, double discr, double dr,
        double r1, double r2, double v00, double v01,
        double N):
    return 0.5 * (discr*v00*(r1**g+r2**g) + (d - a)*dr*v00 - 2*b*v01*dr) / discr


# cdef int findchr(double pos, double[:] lengths):
#     cdef Py_ssize_t i = 0
#     cdef int length = lengths[0]
#     while length < pos:
#         i += 1
#         length += lengths[i]
#     return i


cdef double fun(double l, double ls, double s, double gs, double dur, double N):
    length = l - ls

    cdef double ll = (1 + exp(-2*length))/2
    cdef double ldl = (1 - exp(-2*length))/2

    cdef double a = geta(N, ll, ldl, s)
    cdef double b = getb(N, ll, ldl, s)
    cdef double c = getc(N, ll, ldl, s)
    cdef double d = getd(N, ll, ldl, s)

    cdef double discr = getdiscr(a, b, c, d)

    cdef double r1 = getroot1(a, b, c, d)
    cdef double r2 = getroot2(a, b, c, d)

    cdef double v00 = 1-s
    cdef double v01 = (1-s)*(1-s)

    cdef double dr = pow(r1, dur-1) - pow(r2, dur-1)

    return (getvn1(s, dur-1, length, a, b, c, d, ll,
                   ldl, discr, dr, r1, r2, v00, v01, N)
          - getvn2(s, dur-1, length, a, b, c, d, ll,
                   ldl, discr, dr, r1, r2, v00, v01, N)) \
          * pow(ll, gs)
# cdef double fun(double l, double ls, double s, double gs, double dur):
#     length = abs(l - ls)
#     return (getvn1(s, dur, length) - getvn2(s, dur, length)) * pow(0.5 * (1 + exp(-2*length)), gs)


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def fun3(double l, double ls, double lss,
         double gs, double dur,
         double[:, ::1] L, D, v0):
    # check all these probabilities
    cdef double d1 = abs(ls - l)
    cdef double d2 = abs(lss - l)
    cdef double d3 = abs(lss - ls)

    # Haldane map function
    cdef double lls = (1 + exp(-2*d1)) / 2
    cdef double ldls = (1 - exp(-2*d1)) / 2
    cdef double lsdlss = (1 - exp(-2*d3)) / 2
    cdef double lslss = (1 + exp(-2*d3)) / 2
    cdef double llslss = lls*lslss
    cdef double llsdlss = lls*lsdlss
    cdef double ldlslss = ldls*lslss
    cdef double ldlsdlss = ldls*lsdlss
    cdef double llss = llslss + ldlsdlss
    cdef double ldlss = 1 - (llslss + ldls*lsdlss)

    # my
    U = np.array([
        [llslss, ldlslss, ldlsdlss, llsdlss, 0],
        [0, lslss, 0, 0, lsdlss],
        [0, 0, llss, 0, ldlss],
        [0, 0, 0, lls, ldls],
        [0, 0, 0, 0, 1]
    ])

    eigv = np.array([1, -1, -1, -1, 2], dtype=float)

    A = L @ U @ D
    U, Q = np.linalg.eig(A)
    U = np.diag(U**(dur-1))
    Q_inv = np.linalg.solve(Q, np.eye(5))
    return  (eigv @ Q) @ U @ (Q_inv @ v0) * pow(llslss, gs)
