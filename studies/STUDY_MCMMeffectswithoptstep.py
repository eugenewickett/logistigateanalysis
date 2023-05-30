from logistigate.logistigate import utilities as util  # Pull from the submodule "develop" branch
from logistigate.logistigate import methods, lg
from logistigate.logistigate import lossfunctions as lf
from logistigate.logistigate import samplingplanfunctions as sampf
from logistigate.logistigate.priors import prior_normal_assort
import os
import numpy as np
from numpy.random import choice
import scipy.special as sps
import scipy.stats as spstat
import scipy.optimize as spo
import matplotlib.pyplot as plt
import random
import time
from math import comb
import matplotlib.cm as cm

# 30-MAY-23
# How big should MCMC set be, and how sensitive are we to different MCMC sets?

Nfam = np.array([[1., 1., 10., 1., 3., 0., 1., 6., 7., 5., 0., 0., 4.],
                     [1., 1., 4., 2., 0., 1., 1., 2., 0., 4., 0., 0., 1.],
                     [3., 17., 31., 4., 2., 0., 1., 6., 0., 23., 1., 2., 5.],
                     [1., 1., 15., 2., 0., 0., 0., 1., 0., 6., 0., 0., 0.]])
Yfam = np.array([[0., 0., 7., 0., 3., 0., 1., 0., 1., 0., 0., 0., 4.],
                 [0., 0., 2., 2., 0., 1., 1., 0., 0., 1., 0., 0., 1.],
                 [0., 0., 15., 3., 2., 0., 0., 2., 0., 1., 1., 2., 5.],
                 [0., 0., 5., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
(numTN, numSN) = Nfam.shape  # For later use
csdict_fam = util.initDataDict(Nfam, Yfam)  # Initialize necessary logistigate keys

csdict_fam['TNnames'] = ['MOD_39', 'MOD_17', 'MODHIGH_95', 'MODHIGH_26']
csdict_fam['SNnames'] = ['MNFR ' + str(i + 1) for i in range(numSN)]

SNpriorMean = np.repeat(sps.logit(0.1), numSN)
TNpriorMean = sps.logit(np.array([0.1, 0.1, 0.15, 0.15]))
TNvar, SNvar = 2., 4.  # Variances for use with prior
csdict_fam['prior'] = prior_normal_assort(np.concatenate((SNpriorMean, TNpriorMean)),
                                np.diag(np.concatenate((np.repeat(SNvar, numSN), np.repeat(TNvar, numTN)))))

csdict_fam['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
numdraws = 5000
csdict_fam['numPostSamples'] = numdraws
csdict_fam = methods.GeneratePostSamples(csdict_fam)





numcanddraws, numtruthdraws, numdatadraws  = 5000, 5000, 3000

paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN))

canddraws, truthdraws, datadraws = util.distribute_draws(csdict_fam['postSamples'], numcanddraws,
                                                                     numtruthdraws, numdatadraws)
paramdict.update({'canddraws': canddraws, 'truthdraws': truthdraws, 'datadraws': datadraws})

# What is loss at different budgets?
# Now with a budget
sampbudget = 100
des = np.array([0.,1.,0.,0.]) # Node 2
allocarr = des * sampbudget

def cand_obj_val(x, truthdraws, Wvec, paramdict, riskmat):
    '''function for optimization step'''
    numnodes = x.shape[0]
    scoremat = lf.score_diff_matrix(truthdraws, x.reshape(1, numnodes), paramdict['scoredict'])[0]
    return np.sum(np.sum(scoremat*riskmat, axis=1)*Wvec)
'''
# RETURNS SAME VALUES AS IN (LW) MATRIX IF CANDDRAW IS USED FOR x; example:
i, j = 1, 1
x = canddraws[i]
print(cand_obj_val(x,truthdraws,W[:,j],paramdict))
print(LW[i, j])
'''

# define a gradient function for any candidate vector x
def cand_obj_val_jac(x, truthdraws, Wvec, paramdict, riskmat):
    """function gradient for optimization step"""
    jacmat = np.where(x < truthdraws, -paramdict['scoredict']['underestweight'], 1) * riskmat \
                * Wvec.reshape(truthdraws.shape[0],1)
    return np.sum(jacmat, axis=0)
'''
# Check gradient
x0 = truthdraws[50]
diff = 1e-5
for g in range(len(x0)):
    x1 = x0.copy()
    x1[g] += diff
    obj0 = cand_obj_val(x0, truthdraws, Wvec, paramdict)
    dobj0 = cand_obj_val_jac(x0, truthdraws, Wvec, paramdict)
    obj1 = cand_obj_val(x1, truthdraws, Wvec, paramdict)
    print(obj1-obj0)
    print(dobj0[g]*diff)
'''
def cand_obj_val_hess(x, truthdraws, Wvec, paramdict, riskmat):
    return np.zeros((x.shape[0],x.shape[0]))

# define an optimization function for a set of parameters, truthdraws, and weights matrix
def get_bayes_min_cand(truthdraws, Wvec, paramdict, xinit='na', optmethod='L-BFGS-B'):
    # Initialize with random truthdraw if not provided
    if isinstance(xinit, str):
        xinit = truthdraws[choice(np.arange(truthdraws.shape[0]))]
    # Get risk matrix
    riskmat = lf.risk_check_array(truthdraws, paramdict['riskdict'])
    # Minimize expected candidate loss
    spoOutput = spo.minimize(cand_obj_val, xinit, jac=cand_obj_val_jac,
                             hess=cand_obj_val_hess, method=optmethod,
                             args=(truthdraws, Wvec, paramdict, riskmat))
    return spoOutput


minslist = sampf.sampling_plan_loss_list(design, numtests, priordatadict, paramdict)





