"""
This script compares the performance of the knapsack heuristic and a new heuristic that greedily moves to the next
test node
"""

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
numdraws = 50000
csdict_fam['numPostSamples'] = numdraws
np.random.seed(1000) # To replicate draws later
csdict_fam = methods.GeneratePostSamples(csdict_fam)

# Loss specification
paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN))

# Set limits of data collection and intervals for calculation
testmax, testint = 400, 10
testarr = np.arange(testint, testmax + testint, testint)

# Set MCMC draws to use in fast algorithm
numtruthdraws, numdatadraws = 15000, 50
# Get random subsets for truth and data draws
truthdraws, datadraws = util.distribute_truthdata_draws(csdict_fam['postSamples'], numtruthdraws, numdatadraws)
paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
# Get base loss
paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)

util.print_param_checks(paramdict) # Check of used parameters

# Load matrices for KS heuristic
util_avg = np.load(os.path.join('casestudyoutputs', '31MAY', 'util_avg_fam.npy'))
util_hi = np.load(os.path.join('casestudyoutputs', '31MAY', 'util_hi_fam.npy'))
util_lo = np.load(os.path.join('casestudyoutputs', '31MAY', 'util_lo_fam.npy'))

# Plot
util.plot_marg_util_CI(util_avg, margutilarr_hi=util_hi, margutilarr_lo=util_lo, testmax=testmax, testint=testint,
                               titlestr='Familiar Setting')
# Delta plot
util.plot_marg_util_CI(util_avg, margutilarr_hi=util_hi, margutilarr_lo=util_lo, testmax=testmax, testint=testint,
                               titlestr='Familiar Setting',type='delta')

# Form allocation
allocarr_old, objValArr = sampf.smooth_alloc_forward(util_avg)
util.plot_plan(allocarr_old, paramlist=[str(i) for i in np.arange(testint, testmax + 1, testint)], testint=testint,
          labels=csdict_fam['TNnames'], titlestr='Familiar Setting', allocmax=250,
          colors=cm.rainbow(np.linspace(0, 0.5, numTN)), dashes=[[1, 0] for tn in range(numTN)])

# Form NEW allocation, using choice from last budget for next one
allocarr_new = np.array([[0., 0., 0., 0., 0., 0., 1., 1., 2., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 2., 3., 4., 5., 6., 6., 7., 7., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0.]])
bestalloc = allocarr_new[:,8]
for testind in range(9, testarr.shape[0]):
    print('Budget: '+str(testarr[testind]))
    # Evaluate increment of each test node
    intsaredistinct = False
    losslist_list = [[] for i in range(numTN)]
    nondominatedinds = [i for i in range(numTN)]
    dataiters = 0
    while not intsaredistinct: # while the CIs overlap...
        datadraws = truthdraws[choice(np.arange(numtruthdraws), size=numdatadraws, replace=False)]
        paramdict.update({'datadraws': datadraws})
        dataiters += numdatadraws
        for tnind in range(numTN):
            if tnind in nondominatedinds:
                curralloc = bestalloc.copy() * testint
                curralloc[tnind] += testint
                currdes = curralloc / np.sum(curralloc)
                currlosslist = sampf.sampling_plan_loss_list(currdes, testarr[testind], csdict_fam, paramdict)
                losslist_list[tnind] = losslist_list[tnind] + currlosslist
        print('Num loss evals: '+str(dataiters))
        # Check 99% CIs
        currCI_list = []
        for tnind in range(numTN):
            if tnind in nondominatedinds:
                _, currloss_CI = sampf.process_loss_list(losslist_list[tnind], zlevel=0.95)
                currCI_list.append(currloss_CI)
                print('TN '+str(tnind)+': '+str(currloss_CI))
        minCImax = min([x[1] for x in currCI_list])
        minCImaxind = nondominatedinds[np.argmin([x[1] for x in currCI_list])]
        if (minCImax < [x[0] for x in currCI_list]).sum()==len(currCI_list)-1: # our minmax is lower than all the others
            intsaredistinct = True
        else:
            for ind, item in enumerate(nondominatedinds):
                if currCI_list[ind][0] > minCImax: # Remove ind from nondominated list
                    _ = nondominatedinds.pop(nondominatedinds.index(item))
        if dataiters >= 2000: # Our max number of data draws
            intsaredistinct = True
    allocarr_new[:,testind] = bestalloc.copy()
    allocarr_new[minCImaxind,testind] += 1
    bestalloc = allocarr_new[:,testind]
    print('Best design:')
    print(bestalloc*testint)

'''
allocarr_new = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  2.,  2.,  2.,  2.,  3.,
         4.,  4.,  4.,  5.,  6.,  7.,  7.,  7.,  7.,  7.,  7.,  8.,  9.,
         9., 10., 10., 10., 10., 11., 11., 11., 12., 12., 13., 13., 14.,
        14.],
       [ 1.,  2.,  3.,  4.,  5.,  6.,  6.,  7.,  7.,  8.,  8.,  9.,  9.,
         9.,  9., 10., 10., 10., 10., 10., 10., 10., 10., 11., 11., 11.,
        12., 12., 13., 14., 14., 14., 14., 15., 15., 15., 15., 15., 15.,
        16.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,
         1.,  2.,  2.,  2.,  2.,  2.,  3.,  4.,  5.,  6.,  6.,  6.,  6.,
         6.,  6.,  6.,  6.,  7.,  7.,  8.,  8.,  8.,  9.,  9., 10., 10.,
        10.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.]])
'''

# How do allocations differ?
allocarr_old - allocarr_new

# What is difference in comprehensive utility?
'''
100: TN 1: (1.9494362155464533, 1.9602821342783578)
110: TN 2: (1.9226607184932332, 1.9343677677297897)
120: TN 1: (1.8948269333483694, 1.9067913935518819)
130: TN 0: (1.8695331527893724, 1.881862169459127)
140: TN 0: (1.842496400789059, 1.8556617140169536)
150: TN 2: (1.8087395567252296, 1.8241998745149552)
160: TN 1: (1.7794419427237969, 1.7945076874971295)
170: TN 0: (1.7524854614682261, 1.768563723600556)
180: TN 0: (1.7350681079304922, 1.7530561256222983)
190: TN 0: (1.7084418900112406, 1.726258160383779)
200: TN 2: (1.6792442582397926, 1.69896549716003)
210: TN 2: (1.6570923782569034, 1.6780365361383176)
220: TN 2: (1.6360796094297507, 1.655527399158422)
230: TN 2: (1.6069867105268545, 1.627936020106348)
240: TN 1: (1.5679589369687124, 1.5916295915249976)
250:
260:
270:
280:
290:
300:
310:
320:
330:
340:
350:
360:
370:
380:
390:
400:
'''

# Evaluate utility for heuristic, uniform, and rudimentary
util_avg_heur, util_hi_heur, util_lo_heur = np.zeros((int(testmax / testint) + 1)), \
                                            np.zeros((int(testmax / testint) + 1)), \
                                            np.zeros((int(testmax / testint) + 1))
util_avg_unif, util_hi_unif, util_lo_unif = np.zeros((int(testmax / testint) + 1)), \
                                            np.zeros((int(testmax / testint) + 1)), \
                                            np.zeros((int(testmax / testint) + 1))
util_avg_rudi, util_hi_rudi, util_lo_rudi = np.zeros((int(testmax / testint) + 1)), \
                                            np.zeros((int(testmax / testint) + 1)), \
                                            np.zeros((int(testmax / testint) + 1))
plotupdate = True
for testind in range(testarr.shape[0]):
    # Heuristic utility
    des_heur = allocArr[:, testind] / np.sum(allocArr[:, testind])
    currlosslist = sampf.sampling_plan_loss_list(des_heur, testarr[testind], csdict_fam, paramdict)
    avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
    util_avg_heur[testind+1] = paramdict['baseloss'] - avg_loss
    util_lo_heur[testind+1] = paramdict['baseloss'] - avg_loss_CI[1]
    util_hi_heur[testind+1] = paramdict['baseloss'] - avg_loss_CI[0]
    print(des_heur)
    print('Utility at ' + str(testarr[testind]) + ' tests, Heuristic: ' + str(util_avg_heur[testind+1]))
    # Uniform utility
    des_unif = util.round_design_low(np.ones(numTN) / numTN, testarr[testind]) / testarr[testind]
    currlosslist = sampf.sampling_plan_loss_list(des_unif, testarr[testind], csdict_fam, paramdict)
    avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
    util_avg_unif[testind+1] = paramdict['baseloss'] - avg_loss
    util_lo_unif[testind+1] = paramdict['baseloss'] - avg_loss_CI[1]
    util_hi_unif[testind+1] = paramdict['baseloss'] - avg_loss_CI[0]
    print(des_unif)
    print('Utility at ' + str(testarr[testind]) + ' tests, Uniform: ' + str(util_avg_unif[testind+1]))
    # Rudimentary utility
    des_rudi = util.round_design_low(np.divide(np.sum(Nfam, axis=1), np.sum(Nfam)), testarr[testind]) / testarr[testind]
    currlosslist = sampf.sampling_plan_loss_list(des_rudi, testarr[testind], csdict_fam, paramdict)
    avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
    util_avg_rudi[testind+1] = paramdict['baseloss'] - avg_loss
    util_lo_rudi[testind+1] = paramdict['baseloss'] - avg_loss_CI[1]
    util_hi_rudi[testind+1] = paramdict['baseloss'] - avg_loss_CI[0]
    print(des_rudi)
    print('Utility at ' + str(testarr[testind]) + ' tests, Rudimentary: ' + str(util_avg_rudi[testind+1]))
    if plotupdate:
        util_avg_arr = np.vstack((util_avg_heur, util_avg_unif, util_avg_rudi))
        util_hi_arr = np.vstack((util_hi_heur, util_hi_unif, util_hi_rudi))
        util_lo_arr = np.vstack((util_lo_heur, util_lo_unif, util_lo_rudi))
        # Plot
        util.plot_marg_util_CI(util_avg_arr, util_hi_arr, util_lo_arr, testmax=testmax, testint=testint,
                               titlestr='Familiar Setting, comparison with other approaches')

# Store matrices
np.save(os.path.join('casestudyoutputs', '31MAY', 'util_avg_arr_fam'), util_avg_arr)
np.save(os.path.join('casestudyoutputs', '31MAY', 'util_hi_arr_fam'), util_hi_arr)
np.save(os.path.join('casestudyoutputs', '31MAY', 'util_lo_arr_fam'), util_lo_arr)

