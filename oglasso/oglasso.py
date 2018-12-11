import numpy as np
import numpy.linalg.linalg as la


def ladmm(x0, A, b, lam=1, r=1, niter=10, tol=1e-3):
    (m, n) = A.shape
    z = x0
    x = x0
    u = np.zeros((n, 1))
    Q = la.inv(A.T.dot(A) + r * np.eye(n))
    sthreshv = np.vectorize(sthresh)
    k = 0
    while k < niter:
        x = Q.dot(A.T.dot(b) + r * (z - u))
        z = sthreshv(x + u, lam / r)
        u = u + x - z
        k += 1
    return (z, x, u)


def admm(x0, A, b, grps, lam=1, r=1, niter=10, tol=1e-3):
    (m, n) = A.shape
    z = x0
    xs = [x0[gi] for gi in grps]
    us = [np.zeros(len(gi)) for gi in grps]
    Q = la.inv(A.T.dot(A) + r * np.eye(n))
    k = 0
    while k < niter:
        xs = update_xs(z, us, lam, r, grps)
        z = update_z(xs, us, grps, r, A, b, Q)
        us = update_us(us, xs, z, grps)
        k += 1
    return (z, xs, us)


def update_xs(z, us, l, r, grps):
    N = len(us)
    xs = [Sthresh(z[grps[i]] + us[i], l / r) for i in range(N)]
    return (xs)


def update_z(xs, us, grps, r, A, b, Q):
    n = A.shape[1]
    N = len(grps)

    votes = np.zeros((n, 1))
    xsum = np.zeros((n, 1))
    usum = np.zeros((n, 1))
    for i in range(N):
        votes[grps[i]] += 1
        xsum[grps[i]] += xs[i]
        usum[grps[i]] += us[i]
    xbar = np.divide(xsum, votes)
    ubar = np.divide(usum, votes)

    z = Q.dot(A.T.dot(b) + r * (xbar - ubar))
    return (z)


def update_us(us, xs, z, grps):
    N = len(grps)
    usnew = [us[i] + xs[i] - z[grps[i]] for i in range(N)]
    return (usnew)


def sthresh(a, thresh):
    '''
    Apply soft threshold to the scalar `a`
    '''
    norm = abs(a)
    if norm == 0:
        scal = 0
    else:
        scal = (1 - thresh / norm)
    if scal < 0:
        scal = 0

    return (scal * a)


def Sthresh(vec, thresh):
    '''
    Perform vector form of soft thresholding of vector `vec`
    '''
    norm = la.norm(vec, 2)
    if norm == 0:
        scal = 0
    else:
        scal = (1 - thresh / norm)
    if scal < 0:
        scal = 0

    return (scal * vec)


def sim(m, n, grps, grate=0.5, comp=True, sig2=1):
    A = np.random.randn(m, n)
    N = len(grps)

    if comp:  # complement of union
        live = np.random.binomial(1, grate, N)
        mask = np.ones(n)  # mask starts on, turn stuff off
        for i in range(N):
            mask[grps[i]] *= live[i]
        xstar = np.random.randn(n) * mask

    else:
        live = np.random.binomial(1, grate, N)
        mask = np.zeros(n)  # mask starts off, turn stuff on
        for i in range(N):
            mask[grps[i]] = 1 * live[i]
        xstar = np.random.randn(n) * mask

    b = A.dot(xstar) + sig2 * np.random.randn(m)

    return (xstar, A, b)
