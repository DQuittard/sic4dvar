p = 'debitance'
o = 'hydraulic_radius'
n = 'wide'
m = IndexError
h = 'should be always increasing'
g = 'perimeter'
f = str
e = range
d = any
Y = 'area'
X = 'z'
V = 'w_up_bound'
U = 'w_low_bound'
T = 'z_up_bound'
S = 'z_low_bound'
R = list
Q = 'bottom'
O = 'top'
N = TypeError
K = 0.0
M = AssertionError
I = 'elevation'
G = property
H = 'width'
F = False
D = True
C = None
import copy as E, pathlib
from typing import Iterable, Literal, Tuple
import numpy as A, pandas as q
from shapely import LineString as s
from sic4dvar_classes.sic4dvar_0_defaults import SIC4DVarLowCostDefaults as i
from sic4dvar_functions.helpers.helpers_arrays import arrays_rmv_nan_pair as j, arrays_check_increase as Z, iterable_to_flattened_array as a, array_fix_next_same as k, arrays_bounds as J
from sic4dvar_functions.helpers.helpers_generic import pairwise as t
from sic4dvar_functions.helpers.helpers_plot import helper_plot_lines as u
from sic4dvar_functions.T14 import D as r

def v(width_array, elevation_array, comp_area=D, comp_per=D, check_nan=D, sort=D, check_increasing=D):
    G = comp_per
    E = comp_area
    C = elevation_array
    B = width_array
    if check_nan:
        B, C, W = j(B, C)
    else:
        B = a(B)
        C = a(C)
    B = B.astype(A.float32)
    C = C.astype(A.float32)
    if B.size < 2:
        raise L()
    if sort:
        C = C[A.argsort(B)]
        B.sort()
    if check_increasing:
        if not Z(C, remove_nan=F):
            raise L()
        if not Z(B, remove_nan=F):
            raise L()
    H, I = ([B[0] / 2], [K])
    for (J, N), (O, P) in t(zip(B / 2, C)):
        if G:
            D = s([[J, N], [O, P]]).length
            if A.isnan(D) or not A.isfinite(D) or D < K:
                raise M
            H.append(D)
        if E:
            S = J
            T = O
            U = P - N
            V = (S + T) * U
            I.append(V)
    if G:
        Q = A.array(H) * 2
    else:
        Q = A.empty(0, dtype=A.float32)
    if E:
        R = A.array(I)
    else:
        R = A.empty(0, dtype=A.float32)
    return (R, Q)

def l(width_array, elevation_array, comp_area=D, comp_per=D, check_nan=D, sort=D, check_increasing=D):
    D = comp_area
    C = width_array
    B = elevation_array
    F, G = v(width_array=C, elevation_array=B, comp_area=D, comp_per=comp_per, check_nan=check_nan, sort=sort, check_increasing=check_increasing)
    E, H = (A.nancumsum(F), A.nancumsum(G))
    if D:
        I = C[-1]
        J = B[-1] - B[0]
        if A.round(E[-1], 2) > A.round(J * I, 2) + 1:
            raise M
    return (E, H)

class L(RuntimeError):

    def __init__(A, *B):
        if B:
            A.message = B[0]
        else:
            A.message = C

    def __str__(A):
        if A.message:
            return f'HydraulicGeomParamsError, {A.message}'
        else:
            return f'HydraulicGeomParamsError'

class b:

    def __init__(B, width, elevation, check_nan=D, sort=D, check_increasing=D, float_atol=i().def_cs_float_atol):
        D = float_atol
        B.__n_sig_dig = int(abs(A.log10(D)))
        B.__float_atol = D
        B._width = E.deepcopy(width)
        B._elevation = E.deepcopy(elevation)
        B._wet_a_array = A.empty(0, dtype=A.float32)
        B._wet_per_array = A.empty(0, dtype=A.float32)
        B._hydro_rad_array = A.empty(0, dtype=A.float32)
        B._debitance_array = A.empty(0, dtype=A.float32)
        B._wet_a = C
        B._wet_per = C
        B._hydro_rad = C
        B._debitance = C
        B._min_width = C
        B._max_width = C
        B._min_elevation = C
        B._max_elevation = C
        B._depth = C
        B.__init_helper(check_nan=check_nan, sort=sort, check_increasing=check_increasing)

    @classmethod
    def from_dataframe(B, df, check_nan=D, sort=D, check_increasing=D, elevation_col=X, width_col='w', float_atol=i().def_cs_float_atol):
        return B(width=df[width_col].to_numpy().astype(A.float32), elevation=df[elevation_col].to_numpy().astype(A.float32), check_nan=check_nan, sort=sort, check_increasing=check_increasing, float_atol=float_atol)

    def __v_close(B, v0, v1):
        return A.isclose(v0, v1, rtol=K, atol=B.__float_atol)

    def __init_helper(B, check_nan, sort, check_increasing):
        if check_nan:
            B._width, B._elevation, D = j(B._width, B._elevation)
        else:
            B._width = a(B._width)
            B._elevation = a(B._elevation)
        if B._width.size != B._elevation.size:
            raise m
        B._width = B._width.astype(A.float32)
        B._elevation = B._elevation.astype(A.float32)
        B._width = A.round(B._width, B.__n_sig_dig)
        B._elevation = A.round(B._elevation, B.__n_sig_dig)
        if B._width.size < 2:
            raise L()
        if sort:
            B._elevation = B._elevation[A.argsort(B._width)]
            B._width = A.sort(B._width)
        if check_increasing:
            if not Z(B._elevation, remove_nan=F):
                raise L()
            if not Z(B._width, remove_nan=F):
                raise L()
        B._width[B.__v_close(B._width, K)] = K
        if A.any(B._width < K):
            raise L()
        B._width = k(B._width, float_atol=B.__float_atol)
        B._elevation = k(B._elevation, float_atol=B.__float_atol)
        B._wet_a_array = A.empty(0, dtype=A.float32)
        B._wet_per_array = A.empty(0, dtype=A.float32)
        B._hydro_rad_array = A.empty(0, dtype=A.float32)
        B._debitance_array = A.empty(0, dtype=A.float32)
        B._wet_a = C
        B._wet_per = C
        B._hydro_rad = C
        B._debitance = C
        B._min_width = B._width[0]
        B._max_width = B._width[-1]
        B._min_elevation = B._elevation[0]
        B._max_elevation = B._elevation[-1]
        B._depth = B.elevation - B.min_elevation

    def __get_var1_by_w_z(G, var0_value, var0_ref, var1_ref):
        k = 'perim'
        j = 'radius'
        i = 'comp_per'
        h = 'comp_area'
        f = 'elev'
        c = 'debit'
        U = var0_ref
        R = 'var0_ref'
        P = var0_value
        L = var1_ref
        J = 'var1_ref'
        if U not in [I, H]:
            raise N
        if L in [Y, g]:
            L = f'wetted_{L}'
        if U == L:
            return P
        try:
            P[0]
        except N:
            V = D
            O = A.array([P], dtype=A.float32)
        except m:
            V = D
            O = A.array([P], dtype=A.float32)
        else:
            V = F
            O = A.array(E.deepcopy(P), dtype=A.float32)
        if U == H:
            W = A.concatenate([G.width, O], axis=0)
            X = A.concatenate([G.elevation, A.full_like(O, fill_value=A.nan)], axis=0)
            S = E.deepcopy(W)
            Z = I
        else:
            W = A.concatenate([G.width, A.full_like(O, fill_value=A.nan)], axis=0)
            X = A.concatenate([G.elevation, O], axis=0)
            S = E.deepcopy(X)
            Z = H
        O = (A.round(O, G.__n_sig_dig) * 10 ** G.__n_sig_dig).astype(A.int32)
        S = (A.round(S, G.__n_sig_dig) * 10 ** G.__n_sig_dig).astype(A.int32)
        try:
            B = q.DataFrame({R: S, H: W, I: X})
        except ValueError as n:
            raise n
        B.drop_duplicates(R, keep='first', inplace=D)
        B[R] = B[R].astype(A.int32)
        B.set_index(R, inplace=D)
        B.sort_index(inplace=D)
        B[Z] = r(values_in_array=B[Z].to_numpy(), base_in_array=B.index, limits='linear', check_nan=F, float_atol=G.__float_atol)
        C = B[H].to_numpy() >= G.min_width
        if d([A in L for A in [f, H]]):
            if f in L:
                B[J] = B[I]
            else:
                B[J] = B[H]
            if A.any(~C):
                B.loc[~C, J] = A.nan
        else:
            B[J] = A.full(B.shape[0], A.nan)
            a = {h: F, i: F}
            if d([A in L for A in [c, j, Y]]):
                a[h] = D
            if d([A in L for A in [c, j, k]]):
                a[i] = D
            b, T = l(width_array=B[H].to_numpy()[C], elevation_array=B[I].to_numpy()[C], check_nan=F, sort=F, check_increasing=F, **a)
            o = max(G.__float_atol, G.min_width)
            T = A.maximum(T, o)
            if Y in L:
                B.loc[C, J] = b
                if A.any(~C):
                    B.loc[~C, J] = K
            elif k in L:
                B.loc[C, J] = T
                if A.any(~C):
                    B.loc[~C, J] = K
            else:
                B.loc[C, J] = b / T
                if A.any(~C):
                    B.loc[~C, J] = K
                if c in L:
                    p = B.loc[C, J]
                    s = b * p ** (2.0 / 3.0)
                    B.loc[C, J] = s
        e = B.loc[O].index
        if e.size != O.size:
            raise M
        Q = B.loc[e, J].values
        if Q.size != O.size:
            raise M
        Q = A.round(Q, G.__n_sig_dig)
        if V:
            return Q[0]
        return Q

    def __ext_shrink_by_var(B, var_ref, value, loc_modif=C):
        V = 'adapt'
        U = value
        T = var_ref
        S = 'shrink'
        R = 'extend'
        K = loc_modif
        if T == I:
            E = U
        else:
            E = B.__get_var1_by_w_z(U, T, I)
            if A.isnan(E):
                raise L()
        if E >= B.max_elevation or B.__v_close(E, B.max_elevation):
            K = O
        elif E <= B.min_elevation or B.__v_close(E, B.min_elevation):
            K = Q
        else:
            if O in K.lower():
                K = O
            elif 'bot' in K.lower():
                K = Q
            if K is C:
                raise N
        if T == H:
            G = U
        else:
            G = B.__get_var1_by_w_z(U, T, H)
            if B.__v_close(G, B.min_width):
                G = B.min_width
        if A.isnan(G):
            J = R
            G = B.min_width
        elif B.min_width < G < B.max_width:
            J = S
        elif B.__v_close(G, B.min_width):
            if E < B.min_elevation:
                J = R
            elif B.min_elevation < E < B.max_elevation:
                J = S
            elif E > B.max_elevation:
                raise M
            else:
                J = V
        elif B.__v_close(G, B.max_width):
            if E > B.max_elevation:
                J = R
            elif B.min_elevation < E < B.max_elevation:
                J = S
            elif E < B.min_elevation:
                raise M
            else:
                J = V
        elif G > B.max_width:
            J = R
        else:
            raise M
        if K == O and J == R:
            B._width = A.append(B._width, G)
            B._elevation = A.append(B._elevation, E)
        elif K == Q and J == R:
            B._elevation = A.append(E, B._elevation)
            B._width = A.append(G, B._width)
        elif K == O and J == S:
            P = (B._width <= G) & (B._elevation < E)
            if A.any(P):
                B._elevation = B._elevation[P]
                B._width = B._width[P]
                if not B.__v_close(B._elevation[-1], E) or not B.__v_close(B._width[-1], G):
                    B._elevation = A.sort(A.append(B._elevation, E))
                    if G > B._width[0]:
                        B._width = A.append(B._width, G)
                    else:
                        B._width = A.append(B._width, B._width[-1])
            else:
                raise M
        elif K == Q and J == S:
            P = (B._width >= G) & (B._elevation > E)
            if A.any(P):
                B._elevation = B._elevation[P]
                B._width = B._width[P]
                if not B.__v_close(B._elevation[0], E) or not B.__v_close(B._width[0], G):
                    B._elevation = A.sort(A.append(E, B._elevation))
                    if G < B._width[0]:
                        B._width = A.append(G, B._width)
                    else:
                        B._width = A.append(B._width[0], B._width)
            else:
                raise M
        elif K == O and J == V:
            B._width[-1] = G
            B._elevation[-1] = E
        elif K == Q and J == V:
            B._width[0] = G
            B._elevation[0] = E
        B.__init_helper(check_nan=F, sort=F, check_increasing=D)

    def compute(B):
        B._wet_a_array, B._wet_per_array = l(B._width, B._elevation, check_nan=F, sort=F, check_increasing=F)
        B._wet_a, B._wet_per = (B._wet_a_array[-1], B._wet_per_array[-1])
        C = E.deepcopy(B._wet_per_array)
        D = max(B.__float_atol, B.min_width)
        C = A.maximum(C, D)
        B._hydro_rad_array = B._wet_a_array / C
        B._hydro_rad = B._hydro_rad_array[-1]
        B._debitance_array = B._wet_a_array * B._hydro_rad_array ** (2.0 / 3.0)
        B._debitance = B._debitance_array[-1]

    def plot(B, show=D, plot_title='', fig_width=15, fig_height=5, y_axis_title='elevation (m)', x_axis_title='width (m)', y_lim=C, x_max_lim=C, output_file=C):
        D = x_max_lim
        H = E.deepcopy(-B._width[::-1] / 2)
        I = E.deepcopy(B._width / 2)
        J = A.append(H, I)
        K = E.deepcopy(B._elevation[::-1])
        L = E.deepcopy(B._elevation)
        M = A.append(K, L)
        if D is not C:
            G = (-D / 2, D / 2)
        else:
            G = C
        u(xs=[J], ys=[M], show=show, title=plot_title, x_axis_title=x_axis_title, y_axis_title=y_axis_title, fig_width=fig_width, fig_height=fig_height, y_lim=y_lim, x_lim=G, add_legend=F, output_file=output_file)

    def combine_with_other(I, other, self_z_up_bound_array=C, self_z_low_bound_array=C, self_w_up_bound_array=C, self_w_low_bound_array=C, other_z_up_bound_array=C, other_z_low_bound_array=C, other_w_up_bound_array=C, other_w_low_bound_array=C, preference=n):
        A3 = 'w_up_other'
        A2 = 'w_low_other'
        A1 = 'z_up_other'
        A0 = 'z_low_other'
        z = 'w_up_self'
        y = 'w_low_self'
        x = 'z_up_self'
        w = 'z_low_self'
        p = 'self'
        o = 'other'
        n = other_w_low_bound_array
        m = other_w_up_bound_array
        l = other_z_low_bound_array
        k = other_z_up_bound_array
        j = self_w_low_bound_array
        i = self_w_up_bound_array
        g = self_z_low_bound_array
        d = self_z_up_bound_array
        c = 'w_other'
        b = 'w_self'
        P = other
        g, d = J(ref_array=I.elevation, value0_low_bound_array=g, value0_up_bound_array=d)
        j, i = J(ref_array=I.width, value0_low_bound_array=j, value0_up_bound_array=i)
        l, k = J(ref_array=P.elevation, value0_low_bound_array=l, value0_up_bound_array=k)
        n, m = J(ref_array=P.width, value0_low_bound_array=n, value0_up_bound_array=m)
        s = A.round(E.deepcopy(I.elevation) * 10 ** I.__n_sig_dig, 0).astype(A.int64)
        t = A.round(E.deepcopy(P.elevation) * 10 ** I.__n_sig_dig, 0).astype(A.int64)
        u = A.concatenate([E.deepcopy(s), E.deepcopy(t)])
        K = A.full_like(u, fill_value=A.nan, dtype=P.width.dtype)
        B = q.DataFrame({X: E.deepcopy(u), b: K, c: K, w: K, x: K, y: K, z: K, A0: K, A1: K, A2: K, A3: K})
        B.drop_duplicates(subset=X, inplace=D, ignore_index=D)
        B.sort_values(X, inplace=D)
        B.set_index(X, drop=D, inplace=D)
        Q = B.index[A.isin(B.index.to_numpy(), s).nonzero()[0]]
        R = B.index[A.isin(B.index.to_numpy(), t).nonzero()[0]]
        B.loc[Q, b] = E.deepcopy(I.width)
        B.loc[Q, w] = E.deepcopy(g)
        B.loc[Q, x] = E.deepcopy(d)
        B.loc[Q, y] = E.deepcopy(j)
        B.loc[Q, z] = E.deepcopy(i)
        B.loc[R, c] = E.deepcopy(P.width)
        B.loc[R, A0] = E.deepcopy(l)
        B.loc[R, A1] = E.deepcopy(k)
        B.loc[R, A2] = E.deepcopy(n)
        B.loc[R, A3] = E.deepcopy(m)
        for N in B.columns:
            if A.any(A.isfinite(B[N].to_numpy())):
                B[N] = r(values_in_array=B[N].to_numpy(), base_in_array=B.index.to_numpy(), limits=C, check_nan=F, float_atol=I.__float_atol)
            elif 'up' in N:
                B[N] = A.inf
            else:
                B[N] = -A.inf
        O, W, Y, Z, a = ([], [], [], [], [])
        for G in e(B.shape[0]):
            if A.isnan(B.iloc[G][b]):
                H = o
            elif A.isnan(B.iloc[G][c]):
                H = p
            elif 'wi' in preference:
                if B.iloc[G][b] >= B.iloc[G][c]:
                    H = p
                else:
                    H = o
            elif B.iloc[G][b] <= B.iloc[G][c]:
                H = p
            else:
                H = o
            if G == 0:
                O.append(B.iloc[G][f'w_{H}'])
                W.append(B.iloc[G][f'z_low_{H}'])
                Y.append(B.iloc[G][f'z_up_{H}'])
                a.append(B.iloc[G][f'w_low_{H}'])
                Z.append(B.iloc[G][f'w_up_{H}'])
            elif B.iloc[G][f'w_{H}'] >= O[-1]:
                O.append(B.iloc[G][f'w_{H}'])
                W.append(B.iloc[G][f'z_low_{H}'])
                Y.append(B.iloc[G][f'z_up_{H}'])
                a.append(B.iloc[G][f'w_low_{H}'])
                Z.append(B.iloc[G][f'w_up_{H}'])
            else:
                O.append(O[-1])
                W.append(W[-1])
                Y.append(Y[-1])
                a.append(a[-1])
                Z.append(Z[-1])
        B.index = B.index.astype(I.elevation.dtype)
        B.index = B.index * 10 ** (-I.__n_sig_dig)
        I._elevation = B.index.to_numpy(dtype=I.elevation.dtype)
        I._width = A.array(O, dtype=I.width.dtype)
        try:
            I.__init_helper(check_nan=F, sort=F, check_increasing=D)
        except L as v:
            A4 = f(v.args[0])
            if h in A4:
                raise M
            else:
                raise v
        return {S: W, T: Y, U: a, V: Z}

    def add_wet_bathy_from_other(B, other, self_z_up_bound_array=C, self_z_low_bound_array=C, self_w_up_bound_array=C, self_w_low_bound_array=C, other_z_up_bound_array=C, other_z_low_bound_array=C, other_w_up_bound_array=C, other_w_low_bound_array=C):
        c = other_w_low_bound_array
        b = other_w_up_bound_array
        a = other_z_low_bound_array
        Z = other_z_up_bound_array
        Y = self_w_low_bound_array
        X = self_w_up_bound_array
        O = self_z_low_bound_array
        N = self_z_up_bound_array
        C = other
        O, N = J(ref_array=B.elevation, value0_low_bound_array=O, value0_up_bound_array=N)
        Y, X = J(ref_array=B.width, value0_low_bound_array=Y, value0_up_bound_array=X)
        a, Z = J(ref_array=C.elevation, value0_low_bound_array=a, value0_up_bound_array=Z)
        c, b = J(ref_array=C.width, value0_low_bound_array=c, value0_up_bound_array=b)
        if B.min_elevation < C.min_elevation:
            return {S: O, T: N, U: Y, V: X}
        P, G = ([], [])
        Q, W, H, I = ([], [], [], [])
        for E in e(C.elevation.size):
            if C.elevation[E] >= B.min_elevation:
                break
            if C.width[E] >= B.min_width:
                break
            P.append(C.elevation[E])
            G.append(C.width[E])
            Q.append(a[E])
            W.append(Z[E])
            H.append(c[E])
            I.append(b[E])
        if len(P) == 0:
            return {S: O, T: N, U: Y, V: X}
        d = C.get_width_by_elevation(B.min_elevation)
        g = B.min_width
        if d < g:
            P.append(B.min_elevation)
            G.append(d)
        else:
            P.append(B.min_elevation)
            G.append(g)
        Q.append(-A.inf)
        W.append(A.inf)
        H.append(-A.inf)
        I.append(A.inf)
        K = (B._width > G[-1]).nonzero()[0]
        if K.size > 0:
            Q += R(O[K])
            W += R(N[K])
            H += R(Y[K])
            I += R(X[K])
            B._elevation = A.append(A.array(P), B._elevation[K])
            B._width = A.append(A.array(G), B._width[K])
        else:
            Q = [*Q, O[-1]]
            W = [*W, N[-1]]
            H = [*H, H[-1]]
            I = [*I, I[-1]]
            B._elevation = A.append(A.array(P), B._elevation[-1])
            B._width = A.append(A.array(G), G[-1])
        try:
            B.__init_helper(check_nan=F, sort=F, check_increasing=D)
        except L as i:
            j = f(i.args[0])
            if h in j:
                raise M
            else:
                raise i
        return {S: A.array(Q, dtype=B._elevation.dtype), T: A.array(W, dtype=B._elevation.dtype), U: A.array(H, dtype=B._elevation.dtype), V: A.array(I, dtype=B._elevation.dtype)}

    def add_dry_bathy_from_other(B, other, self_z_up_bound_array=C, self_z_low_bound_array=C, self_w_up_bound_array=C, self_w_low_bound_array=C, other_z_up_bound_array=C, other_z_low_bound_array=C, other_w_up_bound_array=C, other_w_low_bound_array=C):
        c = other_w_low_bound_array
        b = other_w_up_bound_array
        a = other_z_low_bound_array
        Z = other_z_up_bound_array
        O = self_w_low_bound_array
        N = self_w_up_bound_array
        K = self_z_low_bound_array
        I = self_z_up_bound_array
        C = other
        K, I = J(ref_array=B.elevation, value0_low_bound_array=K, value0_up_bound_array=I)
        O, N = J(ref_array=B.width, value0_low_bound_array=O, value0_up_bound_array=N)
        a, Z = J(ref_array=C.elevation, value0_low_bound_array=a, value0_up_bound_array=Z)
        c, b = J(ref_array=C.width, value0_low_bound_array=c, value0_up_bound_array=b)
        if B.max_elevation > C.max_elevation:
            return {S: K, T: I, U: O, V: N}
        P, E = ([], [])
        Q, W, X, Y = ([], [], [], [])
        for G in e(C.elevation.size):
            d = C.elevation[G]
            g = C.width[G]
            if d <= B.max_elevation:
                continue
            if g <= B.max_width:
                continue
            P.append(d)
            E.append(g)
            Q.append(a[G])
            W.append(Z[G])
            X.append(c[G])
            Y.append(b[G])
        if len(P) == 0:
            return {S: K, T: I, U: O, V: N}
        i = C.get_width_by_elevation(B.max_elevation)
        j = B.max_width
        P = [B.max_elevation] + P
        if i > j:
            E = [i] + E
        else:
            E = [j] + E
        Q.append(-A.inf)
        W.append(A.inf)
        X.append(-A.inf)
        Y.append(A.inf)
        H = B._width < E[0]
        Q = R(K[H]) + Q
        W = R(I[H]) + W
        X = R(O[H]) + X
        Y = R(N[H]) + Y
        B._elevation = A.append(B._elevation[H], P)
        B._width = A.append(B._width[H], E)
        try:
            B.__init_helper(check_nan=F, sort=F, check_increasing=D)
        except L as k:
            l = f(k.args[0])
            if h in l:
                raise M
            else:
                raise k
        return {S: A.array(Q, dtype=B._elevation.dtype), T: A.array(W, dtype=B._elevation.dtype), U: A.array(X, dtype=B._elevation.dtype), V: A.array(Y, dtype=B._elevation.dtype)}

    def modify_by_elevation(A, value, loc_modif=C):
        A.__ext_shrink_by_var(I, value=value, loc_modif=loc_modif)

    def modify_by_width(A, value, loc_modif=C):
        A.__ext_shrink_by_var(H, value=value, loc_modif=loc_modif)

    def extend_top_by_elevation(A, value):
        B = value
        if B > A.max_elevation:
            A.modify_by_elevation(B)
        if A.__v_close(B, A.max_elevation):
            return
        else:
            raise N

    def extend_top_by_width(A, value):
        B = value
        if B > A.max_width:
            A.modify_by_width(B)
        if A.__v_close(B, A.max_width):
            return
        else:
            raise N

    def extend_bottom_by_elevation(A, value):
        B = value
        if B < A.min_elevation:
            A.modify_by_elevation(B)
        if A.__v_close(B, A.min_elevation):
            return
        else:
            raise N

    def extend_bottom_by_width(A, value):
        B = value
        if B < A.min_width:
            raise N
        if A.__v_close(B, A.min_width):
            return
        raise L()

    def shrink_top_by_elevation(A, value):
        B = value
        if A.min_elevation < B < A.max_elevation:
            A.modify_by_elevation(B, loc_modif=O)
            return
        if A.__v_close(B, A.max_elevation):
            return
        if A.__v_close(B, A.min_elevation):
            B = A.min_elevation
            A.modify_by_elevation(B, loc_modif=O)
            return
        raise N

    def shrink_top_by_width(A, value):
        B = value
        if A.min_width < B < A.max_width:
            A.modify_by_width(B, loc_modif=O)
            return
        if A.__v_close(B, A.max_width):
            return
        if A.__v_close(B, A.min_width):
            B = A.min_width
            A.modify_by_width(B, loc_modif=O)
            return
        else:
            raise N

    def shrink_bottom_by_elevation(A, value):
        B = value
        if A.min_elevation < B < A.max_elevation:
            A.modify_by_elevation(B, loc_modif=Q)
            return
        if A.__v_close(B, A.min_elevation):
            return
        if A.__v_close(B, A.max_elevation):
            B = A.max_elevation
            A.modify_by_elevation(B, loc_modif=Q)
            return
        raise N

    def shrink_bottom_by_width(A, value):
        B = value
        if A.min_width < B < A.max_width:
            A.modify_by_width(B, loc_modif=Q)
            return
        if A.__v_close(B, A.min_width):
            return
        if A.__v_close(B, A.max_width):
            B = A.max_elevation
            A.modify_by_elevation(B, loc_modif=Q)
            return
        else:
            raise N

    def get_width_by_elevation(A, value):
        return A.__get_var1_by_w_z(var0_value=value, var0_ref=I, var1_ref=H)

    def get_elevation_by_width(A, value):
        return A.__get_var1_by_w_z(var0_value=value, var0_ref=H, var1_ref=I)

    def get_depth_by_width(A, value):
        B = A.get_elevation_by_width(value)
        return B - A._min_elevation

    def get_depth_by_elevation(B, value):
        C = value
        D = B.get_width_by_elevation(C)
        E = C - B._min_elevation
        if D > B.min_width:
            return E
        return A.nan

    def get_area_by_elevation(A, value):
        return A.__get_var1_by_w_z(var0_value=value, var0_ref=I, var1_ref=Y)

    def get_area_by_width(A, value):
        return A.__get_var1_by_w_z(var0_value=value, var0_ref=H, var1_ref=Y)

    def get_perimeter_by_elevation(A, value):
        return A.__get_var1_by_w_z(var0_value=value, var0_ref=I, var1_ref=g)

    def get_perimeter_by_width(A, value):
        return A.__get_var1_by_w_z(var0_value=value, var0_ref=H, var1_ref=g)

    def get_hydraulic_radius_by_elevation(A, value):
        return A.__get_var1_by_w_z(var0_value=value, var0_ref=I, var1_ref=o)

    def get_hydraulic_radius_by_width(A, value):
        return A.__get_var1_by_w_z(var0_value=value, var0_ref=H, var1_ref=o)

    def get_debitance_by_elevation(A, value):
        return A.__get_var1_by_w_z(var0_value=value, var0_ref=I, var1_ref=p)

    def get_debitance_by_width(A, value):
        return A.__get_var1_by_w_z(var0_value=value, var0_ref=H, var1_ref=p)

    def __eq__(B, other):
        C = other
        if C.elevation.size != B.elevation.size:
            return F
        if A.allclose(C.elevation, B.elevation, rtol=K, atol=B.__float_atol):
            if A.allclose(C.width, B.width, rtol=K, atol=B.__float_atol):
                return D
        return F

    def __gt__(A, other):
        if A.max_wetted_area > other.max_wetted_area:
            return D
        return F

    def __ge__(A, other):
        if A.max_wetted_area >= other.max_wetted_area:
            return D
        return F

    def __ne__(A, other):
        return not A.__eq__(other)

    def __lt__(A, other):
        return not A.__ge__(other)

    def __le__(A, other):
        return not A.__gt__(other)

    @G
    def width(self):
        return E.deepcopy(self._width)

    @G
    def elevation(self):
        return E.deepcopy(self._elevation)

    @G
    def wetted_perimeter(self):
        A = self
        if A._wet_per is C:
            A.compute()
        return E.deepcopy(A._wet_per_array)

    @G
    def wetted_area(self):
        A = self
        if A._wet_a is C:
            A.compute()
        return E.deepcopy(A._wet_a_array)

    @G
    def hydraulic_radius(self):
        A = self
        if A._hydro_rad is C:
            A.compute()
        return E.deepcopy(A._hydro_rad_array)

    @G
    def debitance(self):
        A = self
        if A._debitance is C:
            A.compute()
        return E.deepcopy(A._debitance_array)

    @G
    def depth(self):
        return E.deepcopy(self._depth)

    @G
    def min_width(self):
        B = self
        if B._min_width is C:
            B._min_width = A.min(B._width)
        return B._min_width

    @G
    def max_width(self):
        B = self
        if B._max_width is C:
            B._max_width = A.max(B._width)
        return B._max_width

    @G
    def min_elevation(self):
        B = self
        if B._min_elevation is C:
            B._min_elevation = A.min(B.elevation)
        return B._min_elevation

    @G
    def max_elevation(self):
        B = self
        if B._max_elevation is C:
            B._max_elevation = A.max(B.elevation)
        return B._max_elevation

    @G
    def max_depth(self):
        return A.max(self.depth)

    @G
    def max_wetted_perimeter(self):
        A = self
        if A._wet_per is C:
            A.compute()
        return A._wet_per

    @G
    def max_wetted_area(self):
        A = self
        if A._wet_a is C:
            A.compute()
        return A._wet_a

    @G
    def max_hydraulic_radius(self):
        A = self
        if A._hydro_rad is C:
            A.compute()
        return A._hydro_rad
if __name__ == '__main__':
    B = b(width=[20.0, 50.0, 150.0, 300.0, 400], elevation=[5.0, 7.0, 10, 15, 20], sort=F, check_increasing=F, check_nan=F)
    c = b(width=[1.0, 10.0, 20.0, 30.0, 40], elevation=[2.0, 2.0, 4, 12, 20])
    B.compute()
    W = E.deepcopy(B)
    P = dict(y_lim=(0, 25), x_max_lim=450.0)
    B.plot('0 big', **P)
    c.plot(plot_title='0 small', **P)
    B.add_wet_bathy_from_other(c)
    B.plot(plot_title='added wet bathy from other 3', **P)
    B.add_dry_bathy_from_other(W)
    B.plot(plot_title='added dry bathy from other 4', **P)
    B = E.deepcopy(W)
    B.combine_with_other(c, preference=n)
    B.plot(plot_title='combined with other wide 5', **P)
    B = E.deepcopy(W)
    B.combine_with_other(c, preference='narrow')
    B.plot(plot_title='combined with other narrow 6', **P)
    B.shrink_bottom_by_elevation(7.5)
    B.plot(plot_title='shrink bottom by elevation 7', **P)
    B.extend_bottom_by_elevation(4)
    B.plot(plot_title='extend bottom by elevation 8', **P)
    W.shrink_bottom_by_width(100)
    W.plot(plot_title='shrink bottom by width 8', **P)
    B = b([992.46, 1145.06], [20.15, 20.91])
    B.plot()
    B.extend_bottom_by_elevation(13.23)
    B.plot()
    w = A.array([K, 4.0, 5, 6, 7, 8, 9, 10, 11])
    with A.printoptions(precision=2, suppress=D):
        0
    B = b(width=[1.82, 106.07, 106.79, 214.37, 215.3, 216.0], elevation=[4.85, 4.9, 5.0, 5.07, 5.53, 6.8], sort=F, check_increasing=F, check_nan=F, float_atol=0.01)
    B.plot(y_lim=[2.0, 8.0], plot_title='before extension')
    B.extend_bottom_by_elevation(2.15)
    B.plot(y_lim=[2.0, 8.0], plot_title='after extension')