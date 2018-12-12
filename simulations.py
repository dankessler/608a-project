import numpy as np
import oglasso.oglasso as og
import pandas as pd
import matplotlib.pyplot as plt

# common settings
np.random.seed(seed=1986)
m = 50
n = 100
r = 1
lams = np.logspace(-3, 3, num=10)
nsims = 1
eps = 1e-2
niter = 1000


def makeplots(df, name, sname):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    ax = df.plot(x='lam', y='pred', logx=True, ax=axes[0, 0], legend=False)
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('Prediction Error')
    ax.set_title(r'Prediction Error vs $\lambda$ For ' + name)
    # fig = ax.get_figure()
    # fig.savefig(sname + '_pred.png')

    ax = df.plot(x='lam', y='accu', logx=True, ax=axes[0, 1], legend=False)
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'Estimation Error: $\hat{x} - x^\star$')
    ax.set_title(r'Estimation Error vs $\lambda$ For ' + name)
    # fig = ax.get_figure()
    # fig.savefig(sname + '_accu.png')

    ax = df.plot(x='lam', y='prec', logx=True, ax=axes[1, 0], legend=False)
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('Precision in Support Recovery')
    ax.set_title(r'Recovery of Support: Precision vs $\lambda$ For ' + name)
    # fig = ax.get_figure()
    # fig.savefig(sname + '_prec.png')

    ax = df.plot(x='lam', y='reca', logx=True, ax=axes[1, 1], legend=False)
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('Recall in Support Recovery')
    ax.set_title(r'Recovery of Support: Recall vs $\lambda$ For ' + name)
    # fig = ax.get_figure()
    # fig.savefig(sname + '_reca.png')
    fig.savefig(sname + '.png')


# lasso
G = [[i] for i in range(n)]

xstar = np.zeros((n, 1))
xstar[0:25] = 10

x0 = np.zeros((n, 1))

res_lasso = pd.DataFrame(
    [og.sim2test(m, n, G, xstar, x0, r, lam, eps, niter) for lam in lams])
res_lasso['lam'] = lams

makeplots(res_lasso, 'Lasso', 'lasso')

# group-lasso
G = [
    [range(0, 25)],
    [range(25, n)],
]

xstar = np.zeros((n, 1))
xstar[0:25] = 10

x0 = np.zeros((n, 1))

res_glasso = pd.DataFrame(
    [og.sim2test(m, n, G, xstar, x0, r, lam, eps, niter) for lam in lams])
res_glasso['lam'] = lams

makeplots(res_glasso, 'Group Lasso', 'glasso')

# og-lasso
G = [range(i, i + 10) for i in range(0, 45, 5)]
xstar = np.zeros((n, 1))
xstar[0:5] = 10
xstar[15:20] = 10

x0 = np.zeros((n, 1))

res_oglasso = pd.DataFrame(
    [og.sim2test(m, n, G, xstar, x0, r, lam, eps, niter) for lam in lams])
res_oglasso['lam'] = lams

makeplots(res_oglasso, 'Overlapping Group Lasso', 'oglasso')
