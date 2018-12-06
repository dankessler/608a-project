import numpy as np
import numpy.linalg.linalg as la

def admm(x0,A,b,grps,l=1,r=1,niter=10,tol=1e-3):
    (m,n) = A.shape
    N = len(grps)
    z = x0
    x = x0
    Q = la.inv(A.T.dot(A) + r * np.eye(n))
    xs = [x[gi] for gi in grps]
    ys = [np.zeros(len(gi)) for gi in grps]
    iter = 0
    while iter < niter:
        xs = update_xs(z, ys, l, r, grps)
        z = update_z(xs,ys,grps,r,A,b,Q)
        ys = update_ys(ys, xs, z, grps)
        iter += 1
    return(z,xs,ys)

def update_xs(z, ys, l, r, grps):
    N = len(ys)
    xs = [sthresh(z[grps[i]] + ys[i],l/r) for i in range(N)]
    return(xs)

def update_z(xs,ys,grps,r,A,b,Q):
    n = A.shape[1]
    N = len(grps)

    votes = np.zeros(n)
    xsum = np.zeros(n)
    ysum = np.zeros(n)
    for i in range(N):
        votes[grps[i]] += 1
        xsum[grps[i]] += xs[i]
        ysum[grps[i]] += ys[i]
    xbar = np.divide(xsum,votes)
    ybar = np.divide(ysum,votes)

    z = Q.dot(A.T.dot(b) + r*(xbar - ybar))
    return(z)

def update_ys(ys, xs, z, grps):
    N = len(grps)
    ys = [ys[i] + xs[i] - z[grps[i]] for i in range(N)]
    return(ys)

def sthresh(vec,thresh):
    '''
    Perform soft thresholding of vector `vec`
    '''
    norm = la.norm(vec,2)
    if norm==0:
        scal = 0
    else:
        scal = (1 - thresh/norm)
    if scal<0:
        scal=0

    return(scal * vec)

def sim(m,n,grps,grate=0.5,comp = True, sig2=1):
    A = np.random.randn(m,n)
    N = len(grps)

    if comp:                    # complement of union
        live = np.random.binomial(1,grate,N)
        mask = np.ones(n)       # mask starts on, turn stuff off
        for i in range(N):
            mask[grps[i]] *= live[i]
        xstar = np.random.randn(n) * mask

    else:
        live = np.random.binomial(1,grate,N)
        mask = np.zeros(n)      # mask starts off, turn stuff on
        for i in range(N):
            mask[grps[i]] = 1*live[i]
        xstar = np.random.randn(n) * mask

    b = A.dot(xstar) + sig2*np.random.randn(m)

    return(xstar,A,b)