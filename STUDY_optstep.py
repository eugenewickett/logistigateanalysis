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
import matplotlib.pyplot as plt
import random
import time
from math import comb
import matplotlib.cm as cm

# 23-MAY-23
# Debug why utility evaluations are not changing with different weights matrices

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
numdraws = 20000
csdict_fam['numPostSamples'] = numdraws
np.random.seed(999)  # To replicate draws later
csdict_fam = methods.GeneratePostSamples(csdict_fam)

numcanddraws, numtruthdraws, numdatadraws, numcandneigh = 5000, 5000, 3000, 1000

paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN), candneighnum=numcandneigh)

canddraws, truthdraws, datadraws = util.distribute_draws(csdict_fam['postSamples'], numcanddraws,
                                                                     numtruthdraws, numdatadraws)
paramdict.update({'canddraws': canddraws, 'truthdraws': truthdraws, 'datadraws': datadraws})
paramdict.update({'lossmatrix': lf.build_loss_matrix(truthdraws, canddraws, paramdict)})



import scipy.optimize as spo

# What is loss at different budgets?

# First our standard approach
base = sampf.baseloss(paramdict['lossmatrix'])

# Now with a budget
sampbudget = 100
des = np.array([0.,1.,0.,0.]) # Node 2
allocarr = des * sampbudget
W = sampf.build_weights_matrix(truthdraws,datadraws,allocarr,csdict_fam)

LW = np.matmul(paramdict['lossmatrix'], W)
LWmins = LW.min(axis=0)
loss100 = np.average(LWmins)
util100 = base - loss100
print('Utility at 100 tests, under standard approach: '+str(round(util100,4)))

# Using optimization
def cand_obj_val(x, truthdraws, Wvec, paramdict):
    '''function for optimization step'''
    numnodes = x.shape[0]
    scoremat = lf.score_diff_matrix(truthdraws, x.reshape(1, numnodes), paramdict['scoredict'])[0]
    riskvec = lf.risk_check_array(truthdraws,paramdict['riskdict'])
    #Wvalvec = np.sum(W, axis=1) / W.shape[1]
    return np.sum(np.sum(scoremat*riskvec,axis=1)*Wvec)
# RETURNS SAME VALUES AS IN (LW) MATRIX IF CANDDRAW IS USED FOR x; example:
i, j = 1, 1
x = canddraws[i]
print(cand_obj_val(x,truthdraws,W[:,j],paramdict))
print(LW[i, j])

# define an optimization function for a set of parameters, truthdraws, and weights matrix
def get_bayes_min_cand(truthdraws, Wvec, paramdict, xinit='na'):
    # Initialize with random truthdraw if not provided
    if isinstance(xinit, str):
        xinit = truthdraws[choice(np.arange(truthdraws.shape[0]))]
    # Minimize expected candidate loss
    # NEED BOUNDS?
    #bds = spo.Bounds(np.repeat(0., xinit.shape[0]), np.repeat(1., xinit.shape[0]))
    spoOutput = spo.minimize(cand_obj_val, xinit, args=(truthdraws, Wvec, paramdict), #bounds=bds,
                             tol= 1e-8)  # Reduce tolerance?
    return spoOutput

# First the baseline loss
opt_output = get_bayes_min_cand(truthdraws, np.ones(numtruthdraws)/numtruthdraws, paramdict)
base_opt = opt_output.fun
xinit_base = opt_output.x # use this as our xinit from now on

# Now do iterations over data at the budget level
# Use same W as already generated; stop using running average rule
j = -1 # initialize data index
eps = 1e-2 # stopping rule range
rangelist = [0,1]
minvalslist = []
cumavglist = []
while np.max(rangelist) - np.min(rangelist) > eps and j < 20: # numdatadraws-1: # our stopping rule
    # increment data index and get new data weights vector
    j += 1
    optout = get_bayes_min_cand(truthdraws, W[:,j], paramdict, xinit=xinit_base)
    minvalslist.append(optout.fun)
    cumavglist = np.cumsum(minvalslist)/np.arange(1,j+2)
    if np.mod(j,10) == 3:
        plt.plot(cumavglist)
        plt.show()








