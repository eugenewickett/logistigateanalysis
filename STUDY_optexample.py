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

############
# optimization functions
############
def cand_obj_val(x, truthdraws, Wvec, paramdict, riskmat):
    '''function for optimization step'''
    numnodes = x.shape[0]
    scoremat = lf.score_diff_matrix(truthdraws, x.reshape(1, numnodes), paramdict['scoredict'])[0]
    return np.sum(np.sum(scoremat*riskmat,axis=1)*Wvec)

# define a gradient function
def cand_obj_val_jac(x, truthdraws, Wvec, paramdict, riskmat):
    """function gradient for optimization step"""
    jacmat = np.where(x < truthdraws, -paramdict['scoredict']['underestweight'], 1) * riskmat \
                * Wvec.reshape(truthdraws.shape[0],1)
    return np.sum(jacmat, axis=0)

def cand_obj_val_hess(x,truthdraws,Wvec,paramdict,riskmat):
    return np.zeros((x.shape[0],x.shape[0]))

# define an optimization function for a set of parameters, truthdraws, and weights matrix
def get_bayes_min_cand(truthdraws, Wvec, paramdict, xinit='na', optmethod='BFGS'):
    # Initialize with random truthdraw if not provided
    if isinstance(xinit, str):
        xinit = truthdraws[choice(np.arange(truthdraws.shape[0]))]
    # Get risk matrix
    riskmat = lf.risk_check_array(truthdraws, paramdict['riskdict'])
    # Minimize expected candidate loss
    #bds = spo.Bounds(np.repeat(0., xinit.shape[0]), np.repeat(1., xinit.shape[0]))
    spoOutput = spo.minimize(cand_obj_val, xinit, jac=cand_obj_val_jac,
                             hess=cand_obj_val_hess,
                             method=optmethod, #bounds=bds, tol= 1e-5
                             args=(truthdraws, Wvec, paramdict, riskmat))
    return spoOutput

# critical ratio function
def getbayescritratioest(truthdraws, Wvec, q):
    # Establish the weight-sum target
    wtTarg = q * np.sum(Wvec)
    # Initialize return vector
    est = np.zeros(shape=(len(truthdraws[0])))
    # Iterate through each node's distribution of SFP rates, sorting the weights accordingly
    for gind in range(len(truthdraws[0])):
        currRates = truthdraws[:, gind]
        sortWts = [x for _, x in sorted(zip(currRates, Wvec))]
        sortWtsSum = [np.sum(sortWts[:i]) for i in range(len(sortWts))]
        critInd = np.argmax(sortWtsSum >= wtTarg)
        est[gind] = sorted(currRates)[critInd]
    return est
############
############

############
# Build case study data
############
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
csdict_fam = methods.GeneratePostSamples(csdict_fam)

numcanddraws, numtruthdraws, numdatadraws  = 5000, 5000, 500

paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN))
sampbudget = 100
allocarr = np.array([0., 1., 0., 0.]) * sampbudget

epsPerc, stoprange = 0.005, 10 # Desired accuracy of loss estimate, expressed as a percentage; also number of last observations to consider
numtruthdraws, numdatadraws = 5000, 500
method = 'L-BFGS-B'

# OPTIMIZATION STEP ALGORITHM
csdict_fam = methods.GeneratePostSamples(csdict_fam)
# Distribute draws
truthdraws = csdict_fam['postSamples'][choice(np.arange(numdraws), size=numtruthdraws, replace=False)]
datadraws = truthdraws[choice(np.arange(numtruthdraws), size=numdatadraws, replace=False)]
# Build W
W = sampf.build_weights_matrix(truthdraws, datadraws, allocarr, csdict_fam)
# Generate data until we converge OR run out of data
rangeList, j, minvalslist = [1e-3,1], -1, []
time0 = time.time()
while (np.max(rangeList)-np.min(rangeList)) / np.min(rangeList) > epsPerc and j < numdatadraws-1:
    j +=1 # iterate data draw index
    print('On data draw '+str(j))
    Wvec = W[:, j] # Get W vector for current data draw
    opt_output = get_bayes_min_cand(truthdraws, Wvec, paramdict, xinit=datadraws[j],  optmethod=method)
    minvalslist.append(opt_output.fun)
    cumavglist = np.cumsum(minvalslist) / np.arange(1, j + 2)
    if j > stoprange:
        rangeList = cumavglist[-stoprange:]
        print((np.max(rangeList)-np.min(rangeList)) / np.min(rangeList))
print('Loss estimate: ' + str(cumavglist[-1]))
print('Calculation time: ' + str(time.time() - time0))






