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

"""
How much faster is adding the nearest 1000 neighbors by Euclidean distance, rather than adding the 1000
candidates with the lowest baseline loss?
"""
# Setup usual Familiar case study context
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

paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN), candneighnum=1000)

numcanddraws, numtruthdraws, numdatadraws = 5000, 5000, 3000
canddraws, truthdraws, datadraws = util.distribute_draws(csdict_fam['postSamples'], numcanddraws,
                                                                     numtruthdraws, numdatadraws)
paramdict.update({'canddraws': canddraws, 'truthdraws': truthdraws, 'datadraws': datadraws})
paramdict.update({'lossmatrix': lf.build_loss_matrix(truthdraws, canddraws, paramdict)})


### BEST CURRENT METHOD OF FINDING NEAREST NEIGHBORS (pulled from wrapper function in logistigate)
# Get best current candidate
bestcand = paramdict['canddraws'][np.argmin(np.average(paramdict['lossmatrix'], axis=1))]

# Add neighbors of best candidate to set of Bayes draws
from scipy.spatial.distance import cdist
drawDists = cdist(bestcand.reshape(1, len(truthdraws[0])), drawspool)
neighborinds = np.argpartition(drawDists[0], paramdict['candneighnum'])[:paramdict['candneighnum']]
neighborArr = drawspool[neighborinds]

currcanddraws_neigh, lossmatrix_neigh = lf.add_cand_neighbors(paramdict, csdict_fam['postSamples'],
                                                              currtruthdraws)
### WHAT IF WE RETRIEVED THE LOSS FOR EVERY CANDIDATE INSTEAD OF CALCULATED DISTANCE


### WHAT IF WE USED THE CRITICAL RATIO
def get_crit_ratio_est(truthdraws, paramdict):
    """Retrieve Bayes estimate candidate that is the critical ratio for the SFP rate at each node"""
    return np.quantile(truthdraws,
                       paramdict['scoredict']['underestweight'] / (1 + paramdict['scoredict']['underestweight']),
                       axis=0)
est2 = get_crit_ratio_est(truthdraws,paramdict).reshape(1,17)
sampf.baseloss(lf.build_loss_matrix(truthdraws, est, paramdict))
# How to use the critical ratio with a weights matrix?
alloc = np.array([0.,50,0.,0.])
W = sampf.build_weights_matrix(truthdraws,datadraws,alloc,csdict_fam)
Wvec = W[:,0]
q = paramdict['scoredict']['underestweight'] / (1 + paramdict['scoredict']['underestweight'])
def getbayesest(truthdraws, Wvec, q):
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
losslist = []
for j in range(numdatadraws):
    currWvec = W[:, j]
    newest = getbayesest(truthdraws, currWvec, q)
    losslist.append(np.matmul(lf.build_loss_matrix(truthdraws, est.reshape(1,17), paramdict),
              currWvec))

###################
### WHAT IF WE OPTIMIZED AFTER GENERATING EACH WEIGHTS MATRIX
###################
sampbudget = 100
des = np.array([0.,1.,0.,0.])
allocarr = des * sampbudget
W = sampf.build_weights_matrix(truthdraws,datadraws,allocarr,csdict_fam)

import scipy.optimize as spo

x = canddraws[2]

cand_obj_val(x,truthdraws,W,paramdict)

lossmatrix = lf.build_loss_matrix(truthdraws, canddraws[:5], paramdict)
LWmat = np.matmul(lossmatrix,W)
LWmatavg = np.average(LWmat,axis=1)

def cand_obj_val(x, truthdraws, W, paramdict):
    '''function for optimization step'''
    numnodes = x.shape[0]
    scoremat = lf.score_diff_matrix(truthdraws, x.reshape(1, numnodes), paramdict['scoredict'])[0]
    riskvec = lf.risk_check_array(truthdraws,paramdict['riskdict'])
    Wvalvec = np.sum(W, axis=1) / W.shape[1]
    return np.sum(np.sum(scoremat*riskvec,axis=1)*Wvalvec)

plt.hist(Wvalvec)
plt.show()

def get_bayes_min_cand(truthdraws, W, paramdict):
    # Initialize with random truthdraw
    xinit = truthdraws[choice(np.arange(truthdraws.shape[0]))]
    # Minimize expected candidate loss
    # NEED BOUNDS?
    bds = spo.Bounds(np.repeat(0., xinit.shape[0]), np.repeat(1., xinit.shape[0]))
    spoOutput = spo.minimize(cand_obj_val, xinit, args=(truthdraws, W, paramdict), #bounds=bds,
                             tol= 1e-8)  # Reduce tolerance if not getting integer solutions
    return spoOutput

optout = get_bayes_min_cand(truthdraws, W, paramdict)
print(optout.x)
'''[0.10430342 0.04735841 0.50486985 0.78957791 0.92278796 0.62849234
0.60753786 0.15754878 0.1818898  0.04873693 0.66709724 0.81981531
0.9555529  0.11784781 0.25618332 0.08974038 0.09700689]'''
print(cand_obj_val(optout.x,truthdraws,W,paramdict))
'''2.389713248087856'''

Wvalvec = np.sum(W, axis=1) / W.shape[1]
q = paramdict['scoredict']['underestweight'] / (1 + paramdict['scoredict']['underestweight'])
newx = getbayesest(truthdraws,Wvalvec,q)
print(newx)
'''[0.1033372  0.04655314 0.50836986 0.80887622 0.93306503 0.72997065
0.66639821 0.15801045 0.18680485 0.04805534 0.7523503  0.86533188
0.95966307 0.11677952 0.26316762 0.08878236 0.09558447]'''
cand_obj_val(newx, truthdraws, W, paramdict)
'''2.427297718533237'''
# with 10k truthdraws
print(cand_obj_val(optout.x, truthdraws, W, paramdict))

#########################
#########################
# Test optimization vs analytical approach
budgetarr = np.arange(0,401,50)
numReps = 20

# Critical ratio
q = paramdict['scoredict']['underestweight'] / (1 + paramdict['scoredict']['underestweight'])

# Storage matrices
optresmat = np.zeros((numReps, budgetarr.shape[0]))
opttimemat = np.zeros((numReps, budgetarr.shape[0]))
analyzeresmat = np.zeros((numReps, budgetarr.shape[0]))
analyzetimemat = np.zeros((numReps, budgetarr.shape[0]))
for rep in range(numReps):
    print('Replication ' + str(rep) + '...')
    csdict_fam = methods.GeneratePostSamples(csdict_fam)
    truthdraws = csdict_fam['postSamples'][choice(np.arange(numdraws), size=numtruthdraws, replace=False)]
    datadraws = truthdraws[choice(np.arange(numtruthdraws), size=numdatadraws, replace=False)]
    for budgetind in range(budgetarr.shape[0]):
        print('Budget: ' + str(budgetarr[budgetind]))
        # update plan
        currbudget = budgetarr[budgetind]
        des = np.array([0., 1., 0., 0.])
        allocarr = des * currbudget
        if budgetind == 0:
            W = np.ones((numtruthdraws,numdatadraws)) / numtruthdraws
        else:
            W = sampf.build_weights_matrix(truthdraws, datadraws, allocarr, csdict_fam)
        # Do optimization
        time1 = time.time()
        optobject = get_bayes_min_cand(truthdraws, W, paramdict)
        optresmat[rep, budgetind] = optobject.fun
        opttimemat[rep, budgetind] = time.time() - time1
        # Do analytical solution
        time2 = time.time()
        analyze_x = getbayesest(truthdraws, np.sum(W, axis=1) / W.shape[1], q)
        analyzeresmat[rep, budgetind] =  cand_obj_val(analyze_x, truthdraws, W, paramdict)
        analyzetimemat[rep, budgetind] = time.time() - time2
# Plot
xind = 0
shift = 0.1
for budgetind in range(budgetarr.shape[0]):
    xind += 1
    # Opt vals first
    for rep in range(numReps):
        plt.plot(xind,)