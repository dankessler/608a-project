import numpy as np
import oglasso as og

## ladmm
m = 50
n = 100
grps = [[i] for i in range(n)]
grate = 0.1
comp = True
sig2 = 0

xstar,A,b = og.sim(m,n,grps,grate,comp,sig2)

out = og.ladmm(xstar,A,b,l=0.5,niter=1000)
print(xstar)
print(out[0])
print(out[1])

## low dimesional setting

## lasso setting

# m = 100
# n = 300
# grps = [[i] for i in range(n)]
# grate = 0.1
# comp = True
# sig2 = 1

# xstar,A,b = og.sim(m,n,grps,grate,comp,sig2)



# m = 100
# n = 300

# A = np.random.randn(m,n)
# xstar = np.array([1, 0, 0])
# grps = [[i] for i in range(n)]

# xstar = np.zeros(n)
# xstar[[1,3,5]] = 1

# b = A.dot(xstar) + 0*np.random.randn(m)

# x0 = xstar

# out = og.admm(x0,A,b,grps,niter=100,l=1000,r=1e1)

# out[0][0:10]


# ## non-overlapping group lasso


# ## overlapping group lasso
