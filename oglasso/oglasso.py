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
    zs = [x0[gi] for gi in grps]
    x = x0
    us = [np.zeros(len(gi)) for gi in grps]
    Q = la.inv(A.T.dot(A) + r * np.eye(n))
    k = 0
    while k < niter:
        x = update_x(zs, us, grps, r, A, b, Q)
        zs = update_zs(x, us, lam, r, grps)
        us = update_us(us, x, zs, grps)
        k += 1
    return (zs, x, us)



def update_x(zs, us, grps, r, A, b, Q):
    n = A.shape[1]
    N = len(grps)

    votes = np.zeros((n, 1))
    zsum = np.zeros((n, 1))
    usum = np.zeros((n, 1))
    for i in range(N):
        votes[grps[i]] += 1
        zsum[grps[i]] += zs[i]
        usum[grps[i]] += us[i]
    zbar = np.divide(zsum, votes)
    ubar = np.divide(usum, votes)

    x = Q.dot(A.T.dot(b) + r * (zbar - ubar))
    return (x)


def update_zs(x, us, lam, r, grps):
    N = len(us)
    zs = [Sthresh(x[grps[i]] + us[i], lam / r) for i in range(N)]
    return (zs)



def update_us(us, x, zs, grps):
    N = len(grps)
    usnew = [us[i] + x[grps[i]] - zs[i] for i in range(N)]
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
