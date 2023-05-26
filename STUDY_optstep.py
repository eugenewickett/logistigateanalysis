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

# 21-MAY-23
# Debug why utility evaluations are not changing with different weights matrices

# First do data setup
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

numcanddraws, numtruthdraws, numdatadraws  = 5000, 5000, 3000

paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN))

canddraws, truthdraws, datadraws = util.distribute_draws(csdict_fam['postSamples'], numcanddraws,
                                                                     numtruthdraws, numdatadraws)
paramdict.update({'canddraws': canddraws, 'truthdraws': truthdraws, 'datadraws': datadraws})
paramdict.update({'lossmatrix': lf.build_loss_matrix(truthdraws, canddraws, paramdict)})

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

# Objective function
def cand_obj_val(x, truthdraws, Wvec, paramdict, riskmat):
    '''function for optimization step'''
    numnodes = x.shape[0]
    scoremat = lf.score_diff_matrix(truthdraws, x.reshape(1, numnodes), paramdict['scoredict'])[0]
    return np.sum(np.sum(scoremat*riskmat,axis=1)*Wvec)
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

# Do any methods work faster than any others?
# Omitted 'Nelder-Mead', 'dogleg', 'trust-krylov'
methodlist = ['Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP',
              'trust-constr', 'trust-ncg', 'trust-exact']
numReps = 100
obj_delta_mat = np.zeros((len(methodlist),numReps))
time_delta_mat = np.zeros((len(methodlist),numReps))
for rep in range(numReps):
    print('Rep: '+str(rep))
    xinit = truthdraws[choice(np.arange(len(truthdraws)))]
    Wvec = W[:,rep]
    for methodind, method in enumerate(methodlist):
        time0=time.time()
        curroptobj = get_bayes_min_cand(truthdraws, Wvec, paramdict, xinit, optmethod=method)
        time1 = round(time.time() - time0,3)
        time_delta_mat[methodind][rep] = time1
        if methodind == 0:
            obj_std = curroptobj.fun
            obj_delta_mat[methodind][rep] = 0.
        else:
            obj_delta_mat[methodind][rep] = curroptobj.fun - obj_std
import warnings
warnings.filterwarnings("always") # "always"
'''25-MAY
np.save(os.path.join('studies', 'whichoptimizemethod_25MAY23', 'obj_delta_mat'), np.array(obj_delta_mat))
np.save(os.path.join('studies', 'whichoptimizemethod_25MAY23', 'time_delta_mat'), np.array(time_delta_mat))
'''
# Plot of objective difference
xlabs = []
for methodind, method in enumerate(methodlist):
    plt.boxplot(np.array(obj_delta_mat).T)
    xlabs.append(str(method))
plt.xticks(np.arange(1,11),xlabs,size=6)
plt.title('Objective gap from "Powell" under different scipy methods\n$N=100$, 100 replications, $|\Gamma_{truth}|=5k$')
plt.ylabel('Objective gap')
plt.xlabel('Scipy optimization method')
plt.show()
plt.close()
# Zoom in
xlabs = []
for methodind, method in enumerate(methodlist):
    xlabs.append(str(method))
plt.boxplot(np.array(obj_delta_mat).T)
plt.xticks(np.arange(1,11),xlabs,size=6)
plt.title('Objective gap from "Powell" under different scipy methods\n$N=100$, 100 replications, $|\Gamma_{truth}|=5k$')
plt.ylabel('Objective gap')
plt.xlabel('Scipy optimization method')
plt.ylim([-0.005,0.02])
plt.show()
plt.close()

# Plot of time needed
xlabs = []
for method in methodlist:
    xlabs.append(str(method))
plt.boxplot(np.array(time_delta_mat).T)
plt.xticks(np.arange(1,11),xlabs,size=6)
plt.title('Time (in sec.) needed for 1 data draw under different scipy methods\n$N=100$, 100 replications, $|\Gamma_{truth}|=5k$')
plt.ylabel('Time (s)')
plt.xlabel('Scipy optimization method')
plt.show()
plt.close()

###############
# What about using the critical ratio?
###############
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


###################
# CONDUCT SOME EXPERIMENTS CHANGING KEY PARAMETERS
###################
# Our base case is N=100, random xinit, BFGS method, eps=0.01, numtruthdraws=5k
#   We will change these initial parameters to gauge the effect on computation time and estimate variance
paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN))
sampbudget = 100
allocarr = np.array([0., 1., 0., 0.]) * sampbudget

epsPerc, stoprange = 0.01, 10 # Desired accuracy of loss estimate, expressed as a percentage; also number of last observations to consider
numtruthdraws, numdatadraws = 5000, 500
method = 'BFGS'

# conduct replications under different MCMC draws
numReps = 25
base_loss_est_mat, base_time_mat = np.zeros(numReps), np.zeros(numReps)
for rep in range(numReps):
    print('Replication '+str(rep))
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
        opt_output = get_bayes_min_cand(truthdraws, Wvec, paramdict, optmethod=method)
        minvalslist.append(opt_output.fun)
        cumavglist = np.cumsum(minvalslist) / np.arange(1, j + 2)
        if j > stopamount:
            rangeList = cumavglist[-stopamount:]
            print((np.max(rangeList)-np.min(rangeList)) / np.min(rangeList))
    base_loss_est_mat[rep] = cumavglist[-1]
    base_time_mat[rep] = time.time() - time0 # Time to get our estimate, given intialized values
'''
np.save(os.path.join('studies', 'diroptexperiments_25MAY23', 'base_loss_est_mat'), 
            np.array(base_loss_est_mat))
np.save(os.path.join('studies', 'diroptexperiments_25MAY23', 'base_time_mat'), 
            np.array(base_time_mat))
'''

##########
# NEXT EXPERIMENT: What is the effect of different optimization methods?
##########
sampbudget = 100
allocarr = np.array([0., 1., 0., 0.]) * sampbudget

epsPerc, stoprange = 0.01, 10 # Desired accuracy of loss estimate, expressed as a percentage; also number of last observations to consider
numtruthdraws, numdatadraws = 5000, 500

methodlist = ['L-BFGS-B','SLSQP', 'Powell', 'critratio']

# conduct replications under different MCMC draws
numReps = 25
loss_est_mat_optmethod, time_mat_optmethod = np.zeros((len(methodlist),numReps)), np.zeros((len(methodlist),numReps))
for rep in range(numReps):
    print('Replication '+str(rep))
    csdict_fam = methods.GeneratePostSamples(csdict_fam)
    # Distribute draws
    truthdraws = csdict_fam['postSamples'][choice(np.arange(numdraws), size=numtruthdraws, replace=False)]
    datadraws = truthdraws[choice(np.arange(numtruthdraws), size=numdatadraws, replace=False)]
    # Build W
    W = sampf.build_weights_matrix(truthdraws, datadraws, allocarr, csdict_fam)
    # EXPERIMENT: Loop through each method, using same draws and W
    for methodind, method in enumerate(methodlist):
        # Generate data until we converge OR run out of data
        rangeList, j, minvalslist = [1e-3,1], -1, []
        time0 = time.time()
        while (np.max(rangeList)-np.min(rangeList)) / np.min(rangeList) > epsPerc and j < numdatadraws-1:
            j +=1 # iterate data draw index
            print('On data draw '+str(j))
            Wvec = W[:, j] # Get W vector for current data draw
            if method == 'critratio':
                q = paramdict['scoredict']['underestweight']/(1+ paramdict['scoredict']['underestweight'])
                est = getbayescritratioest(truthdraws, Wvec, q)
                riskmat = lf.risk_check_array(truthdraws, paramdict['riskdict'])
                minvalslist.append(cand_obj_val(est, truthdraws, Wvec, paramdict, riskmat))
            else:
                opt_output = get_bayes_min_cand(truthdraws, Wvec, paramdict, optmethod=method)
                minvalslist.append(opt_output.fun)
            cumavglist = np.cumsum(minvalslist) / np.arange(1, j + 2)
            if j > stopamount:
                rangeList = cumavglist[-stopamount:]
                print((np.max(rangeList)-np.min(rangeList)) / np.min(rangeList))
        loss_est_mat_optmethod[methodind][rep] = cumavglist[-1]
        time_mat_optmethod[methodind][rep] = time.time() - time0
'''
np.save(os.path.join('studies', 'diroptexperiments_25MAY23', 'loss_est_mat_optmethod'), 
            np.array(loss_est_mat_optmethod))
np.save(os.path.join('studies', 'diroptexperiments_25MAY23', 'time_mat_optmethod'), 
            np.array(time_mat_optmethod))
'''
# Plot of time needed
xlabs = []
for method in methodlist:
    xlabs.append(str(method))
plt.boxplot(np.array(time_mat_optmethod).T)
plt.xticks(np.arange(1,5),xlabs,size=8)
plt.title('Time (in sec.) needed to converge under different scipy methods\n25 replications, $N=100$, $|\Gamma_{truth}|=5k$, $\epsilon=1%$')
plt.ylabel('Time (s)')
plt.xlabel('Scipy optimization method')
plt.show()
plt.close()

# Plot of time needed - zoom in, include BFGS
time_mat_2 = np.vstack((time_mat_optmethod[:2], base_time_mat))
xlabs = methodlist[:2] + ['BFGS']
plt.boxplot(np.array(time_mat_2).T)
plt.xticks(np.arange(1,4),xlabs,size=8)
plt.title('Time (in sec.) needed to converge under different scipy methods\n25 replications, $N=100$, $|\Gamma_{truth}|=5k$, $\epsilon=1%$')
plt.ylabel('Time (s)')
plt.xlabel('Scipy optimization method')
plt.ylim([0,12])
plt.show()
plt.close()

# Plot of distribution of converged loss estimates
xlabs = []
for method in methodlist:
    xlabs.append(str(method))
plt.boxplot(np.array(loss_est_mat_optmethod).T)
plt.xticks(np.arange(1,5),xlabs,size=8)
plt.title('Loss estimates after convergence under different scipy methods\n25 replications, $N=100$, $|\Gamma_{truth}|=5k$, $\epsilon=1%$')
plt.ylabel('Loss estimate')
plt.xlabel('Scipy optimization method')
plt.show()
plt.close()

##########
# NEXT EXPERIMENT: What is the effect of different sampling budgets?
##########
sampbudgetlist = [0, 100, 200, 300, 400]

epsPerc, stoprange = 0.01, 10 # Desired accuracy of loss estimate, expressed as a percentage; also number of last observations to consider
numtruthdraws, numdatadraws = 5000, 500

method = 'L-BFGS-B'

# conduct replications under different MCMC draws
numReps = 25
loss_est_mat_optmethod, time_mat_optmethod = np.zeros((len(methodlist),numReps)), np.zeros((len(methodlist),numReps))
datadrawsneeded_mat = np.zeros((len(methodlist), numReps))
for rep in range(numReps):
    print('Replication '+str(rep))
    csdict_fam = methods.GeneratePostSamples(csdict_fam)
    # Distribute draws
    truthdraws = csdict_fam['postSamples'][choice(np.arange(numdraws), size=numtruthdraws, replace=False)]
    datadraws = truthdraws[choice(np.arange(numtruthdraws), size=numdatadraws, replace=False)]
    # EXPERIMENT: Use different sampling budgets
    for sampbudgetind, sampbudget in sampbudgetlist:
        # Build W
        if sampbudget == 0:
            Wvec = np.ones((numtruthdraws)) / numtruthdraws
            time0 = time.time()
            opt_output = get_bayes_min_cand(truthdraws, Wvec, paramdict, optmethod=method)
            datadrawsneeded_mat[sampbudgetind][rep] = 1
            loss_est_mat_budget[sampbudgetind][rep] = opt_output.fun
        else:
            allocarr = np.array([0., 1., 0., 0.]) * sampbudget
            W = sampf.build_weights_matrix(truthdraws, datadraws, allocarr, csdict_fam)
            # Generate data until we converge OR run out of data
            rangeList, j, minvalslist = [1e-3,1], -1, []
            time0 = time.time()
            while (np.max(rangeList)-np.min(rangeList)) / np.min(rangeList) > epsPerc and j < numdatadraws-1:
                j +=1 # iterate data draw index
                print('On data draw '+str(j))
                Wvec = W[:, j] # Get W vector for current data draw
                opt_output = get_bayes_min_cand(truthdraws, Wvec, paramdict, optmethod=method)
                minvalslist.append(opt_output.fun)
                cumavglist = np.cumsum(minvalslist) / np.arange(1, j + 2)
                if j > stopamount:
                    rangeList = cumavglist[-stopamount:]
                    print((np.max(rangeList)-np.min(rangeList)) / np.min(rangeList))
            datadrawsneeded_mat[sampbudgetind][rep] = j + 1
            loss_est_mat_budget[sampbudgetind][rep] = cumavglist[-1]
        time_mat_budget[sampbudgetind][rep] = time.time() - time0












