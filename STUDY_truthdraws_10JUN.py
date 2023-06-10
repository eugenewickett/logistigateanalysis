from logistigate.logistigate import utilities as util # Pull from the submodule "develop" branch
from logistigate.logistigate import methods
from logistigate.logistigate.priors import prior_normal_assort
from logistigate.logistigate import lossfunctions as lf
from logistigate.logistigate import samplingplanfunctions as sampf

import os
import numpy as np
import matplotlib.cm as cm
import matplotlib
import matplotlib.pyplot as plt

from numpy.random import choice
import scipy.special as sps
import time
# Set up initial data
Nfam = np.array([[1., 1., 10., 1., 3., 0., 1., 6., 7., 5., 0., 0., 4.],
                      [1., 1., 4., 2., 0., 1., 1., 2., 0., 4., 0., 0., 1.],
                      [3., 17., 31., 4., 2., 0., 1., 6., 0., 23., 1., 2., 5.],
                      [1., 1., 15., 2., 0., 0., 0., 1., 0., 6., 0., 0., 0.]])
Yfam = np.array([[0., 0., 7., 0., 3., 0., 1., 0., 1., 0., 0., 0., 4.],
                      [0., 0., 2., 2., 0., 1., 1., 0., 0., 1., 0., 0., 1.],
                      [0., 0., 15., 3., 2., 0., 0., 2., 0., 1., 1., 2., 5.],
                      [0., 0., 5., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
(numTN, numSN) = Nfam.shape # For later use
csdict_fam = util.initDataDict(Nfam, Yfam) # Initialize necessary logistigate keys
csdict_fam['TNnames'] = ['MOD_39', 'MOD_17', 'MODHIGH_95', 'MODHIGH_26']
csdict_fam['SNnames'] = ['MNFR ' + str(i+1) for i in range(numSN)]

# Build prior
SNpriorMean = np.repeat(sps.logit(0.1), numSN)
# Establish test node priors according to assessment by regulators
TNpriorMean = sps.logit(np.array([0.1, 0.1, 0.15, 0.15]))
priorMean = np.concatenate((SNpriorMean, TNpriorMean))
TNvar, SNvar = 2., 4.  # Variances for use with prior; supply nodes are wide due to large
priorCovar = np.diag(np.concatenate((np.repeat(SNvar, numSN), np.repeat(TNvar, numTN))))
priorObj = prior_normal_assort(priorMean, priorCovar)
csdict_fam['prior'] = priorObj

# Set up MCMC
csdict_fam['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
# Generate posterior draws
numdraws = 70000
csdict_fam['numPostSamples'] = numdraws

paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN))

# Set limits of data collection and intervals for calculation
testmax, testint = 400, 10
testarr = np.arange(testint, testmax + testint, testint)

numdatadraws = 2000
des = np.array([1., 0., 0., 0.])

numreps=10
truth15arr = np.zeros((numreps, testarr.shape[0]+1))
truth15arr_lo, truth15arr_hi = np.zeros((numreps, testarr.shape[0]+1)), np.zeros((numreps, testarr.shape[0]+1))
truth50arr = np.zeros((numreps, testarr.shape[0]+1))
truth50arr_lo, truth50arr_hi = np.zeros((numreps, testarr.shape[0]+1)), np.zeros((numreps, testarr.shape[0]+1))
truth15times, truth50times = [], []
for rep in range(numreps):
    np.random.seed(1050 + rep)  # To replicate draws later
    csdict_fam = methods.GeneratePostSamples(csdict_fam)
    # 50k truthdraws
    np.random.seed(200 + rep)
    numtruthdraws = 50000
    truthdraws, datadraws = util.distribute_truthdata_draws(csdict_fam['postSamples'], numtruthdraws, numdatadraws)
    paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
    # Get base loss
    paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)
    util.print_param_checks(paramdict) # Check of used parameters
    for testind in range(testarr.shape[0]):
        time0 = time.time()
        currlosslist = sampf.sampling_plan_loss_list(des, testarr[testind], csdict_fam, paramdict)
        truth50times.append(time.time() - time0)
        avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
        truth50arr[rep][testind+1] = paramdict['baseloss'] - avg_loss
        truth50arr_lo[rep][testind+1] = paramdict['baseloss'] - avg_loss_CI[1]
        truth50arr_hi[rep][testind+1] = paramdict['baseloss'] - avg_loss_CI[0]
    # 15k truthdraws
    np.random.seed(200 + rep)
    numtruthdraws = 15000
    truthdraws, datadraws = util.distribute_truthdata_draws(csdict_fam['postSamples'], numtruthdraws, numdatadraws)
    paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
    # Get base loss
    paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)
    util.print_param_checks(paramdict)  # Check of used parameters
    for testind in range(testarr.shape[0]):
        time0 = time.time()
        currlosslist = sampf.sampling_plan_loss_list(des, testarr[testind], csdict_fam, paramdict)
        truth15times.append(time.time() - time0)
        avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
        truth15arr[rep][testind + 1] = paramdict['baseloss'] - avg_loss
        truth15arr_lo[rep][testind + 1] = paramdict['baseloss'] - avg_loss_CI[1]
        truth15arr_hi[rep][testind + 1] = paramdict['baseloss'] - avg_loss_CI[0]
    # Plot
    util.plot_marg_util(np.vstack((truth15arr,truth50arr)),testmax,testint,al=0.2,type='delta',
                        labels=['15k' for i in range(numreps)]+['50k' for i in range(numreps)],
                        colors=['red' for i in range(numreps)]+['blue' for i in range(numreps)],
                        dashes=[[1,0] for i in range(numreps)]+[[2,1] for i in range(numreps)])