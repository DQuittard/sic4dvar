I = dict
import copy as O, numpy as A
from sic4dvar_classes.sic4dvar_0_defaults import SIC4DVarLowCostDefaults as C
from sic4dvar_functions.helpers.helpers_arrays import iterable_to_flattened_array as X

def P(x_array, y_array):
    B, C = A.polyfit(x_array, y_array, 1)
    return (C, B)

def D(values_in_array, base_in_array, limits=None, check_nan=True, float_atol=C().def_float_atol):
    W = 'lin'
    S = float_atol
    R = base_in_array
    Q = values_in_array
    G = limits
    if any([isinstance(B, A.ma.MaskedArray) for B in [Q, R]]):
        raise NotImplementedError
    B, F = [X(A) for A in [Q, R]]
    if check_nan:
        if A.any(A.isnan(F)):
            raise RuntimeError
    if not G:
        J = I(left=A.nan, right=A.nan)
    elif 'fill' in G.lower():
        J = I()
    elif W in G.lower():
        J = I(left=A.nan, right=A.nan)
    else:
        raise TypeError
    C = A.isfinite(B)
    H = C.nonzero()[0]
    E, D = H[[0, -1]]
    if A.all(C):
        return B
    if A.count_nonzero(C) < 2:
        return B
    if G and W in G.lower():
        J = I(left=B[C][0], right=B[C][-1])
    Y = A.interp(x=F, xp=F[C], fp=B[C], **J)
    B = Y
    if not G or 'fil' in G.lower():
        return B
    if E == 0:
        if D == C.size - 1:
            return B
    if E > 0:
        B[:E] = A.nan
    if D < C.size - 1:
        if D + 1 < C.size - 1:
            B[D + 1:] = A.nan
        else:
            B[D + 1] = A.nan
    C = A.isfinite(B)
    H = C.nonzero()[0]
    E, D = H[[0, -1]]
    if E > 0:
        T = H[1]
        Z = F[[E, T]]
        K = O.deepcopy(B[[E, T]])
        if A.isclose(K[0], K[1], rtol=0.0, atol=S):
            B[:E] = K[0]
        else:
            a, L = P(Z, K)
            B[:E] = a + L * F[:E]
    if D < C.size - 1:
        U = H[-2]
        b = F[[U, D]]
        M = O.deepcopy(B[[U, D]])
        if A.isclose(M[0], M[1], rtol=0.0, atol=S):
            N = M[0]
        else:
            V, L = P(b, M)
            if D + 1 < C.size - 1:
                N = V + L * F[D + 1:]
            else:
                N = V + L * F[D + 1]
        if D + 1 < C.size - 1:
            B[D + 1:] = N
        else:
            B[D + 1] = N
    if A.count_nonzero(A.isnan(B)) > 0:
        raise AssertionError
    return B
if __name__ == '__main__':
    E = 0
    while E < 10:
        B = A.random.random_sample(100)
        B[B < 0.2] = A.nan
        D(B, A.array(range(B.size)), limits='linear')