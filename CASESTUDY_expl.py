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
Nexpl = np.array([[1., 1., 10., 1., 3., 0., 1., 6., 7., 5., 0., 0., 4.],
                      [1., 1., 4., 2., 0., 1., 1., 2., 0., 4., 0., 0., 1.],
                      [3., 17., 31., 4., 2., 0., 1., 6., 0., 23., 1., 2., 5.],
                      [1., 1., 15., 2., 0., 0., 0., 1., 0., 6., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
Yexpl = np.array([[0., 0., 7., 0., 3., 0., 1., 0., 1., 0., 0., 0., 4.],
                      [0., 0., 2., 2., 0., 1., 1., 0., 0., 1., 0., 0., 1.],
                      [0., 0., 15., 3., 2., 0., 0., 2., 0., 1., 1., 2., 5.],
                      [0., 0., 5., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
(numTN, numSN) = Nexpl.shape # For later use
csdict_expl = util.initDataDict(Nexpl, Yexpl) # Initialize necessary logistigate keys
csdict_expl['TNnames'] = ['MOD_39', 'MOD_17', 'MODHIGH_95', 'MODHIGH_26',
                              'MODHIGH_EXPL_1', 'MOD_EXPL_1', 'MODHIGH_EXPL_2', 'MOD_EXPL_2']
csdict_expl['SNnames'] = ['MNFR ' + str(i + 1) for i in range(numSN)]

# Use observed data to form Q for tested nodes; use bootstrap data for untested nodes
numBoot = 44  # Average across each TN in original data set
SNprobs = np.sum(csdict_expl['N'], axis=0) / np.sum(csdict_expl['N'])
np.random.seed(33)
Qvecs = np.random.multinomial(numBoot, SNprobs, size=4) / numBoot
csdict_expl['Q'] = np.vstack((csdict_expl['Q'][:4], Qvecs))

# Build prior
SNpriorMean = np.repeat(sps.logit(0.1), numSN)
# Establish test node priors according to assessment by regulators
TNpriorMean = sps.logit(np.array([0.1, 0.1, 0.15, 0.15, 0.15, 0.1, 0.15, 0.1]))
TNvar, SNvar = 2., 4.  # Variances for use with prior; supply nodes are wide due to uncertainty
csdict_expl['prior'] = prior_normal_assort(np.concatenate((SNpriorMean, TNpriorMean)),
                               np.diag(np.concatenate((np.repeat(SNvar, numSN), np.repeat(TNvar, numTN)))))

# Set up MCMC
csdict_expl['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
# Generate posterior draws
numdraws = 10000
csdict_expl['numPostSamples'] = numdraws
np.random.seed(1000) # To replicate draws later
csdict_expl = methods.GeneratePostSamples(csdict_expl)
# Print inference from initial data
#util.plotPostSamples(csdict_expl, 'int90')

# Loss specification
paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN))

# Set limits of data collection and intervals for calculation
testmax, testint = 400, 10
testarr = np.arange(testint, testmax + testint, testint)

# Set MCMC draws to use in fast algorithm
numtruthdraws, numdatadraws = 4000, 1500
# Get random subsets for truth and data draws
truthdraws, datadraws = util.distribute_truthdata_draws(csdict_expl['postSamples'], numtruthdraws, numdatadraws)
paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
# Get base loss
paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)

util.print_param_checks(paramdict) # Check of used parameters
util_avg, util_hi, util_lo = sampf.get_opt_marg_util_nodes(csdict_expl, testmax, testint, paramdict, zlevel=0.95,
                                                           printupdate=True, plotupdate=True) # Wrapper function for utility at all test nodes
# Plot
util.plot_marg_util_CI(util_avg, margutilarr_hi=util_hi, margutilarr_lo=util_lo, testmax=testmax, testint=testint,
                               titlestr='Exploratory Setting')
# Store matrices
np.save(os.path.join('casestudyoutputs', '31MAY', 'util_avg_expl'), util_avg)
np.save(os.path.join('casestudyoutputs', '31MAY', 'util_hi_expl'), util_hi)
np.save(os.path.join('casestudyoutputs', '31MAY', 'util_lo_expl'), util_lo)

#util_avg = np.load(os.path.join('casestudyoutputs', '31MAY', 'margutil_avg_arr_fam.npy'))
#util_hi = np.load(os.path.join('casestudyoutputs', '31MAY', 'margutil_hi_arr_fam.npy'))
#util_lo = np.load(os.path.join('casestudyoutputs', '31MAY', 'margutil_lo_arr_fam.npy'))

# Form allocation
allocArr, objValArr = sampf.smooth_alloc_forward(util_avg)
util.plot_plan(allocArr, paramlist=[str(i) for i in np.arange(testint, testmax + 1, testint)], testint=testint,
               labels=csdict_expl['TNnames'], titlestr='Exploratory Setting', allocmax=250,
               colors=cm.rainbow(np.linspace(0, 0.5, numTN)), dashes=[[1, 0] for tn in range(numTN)])

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
    currlosslist = sampf.sampling_plan_loss_list(des_heur, testarr[testind], csdict_expl, paramdict)
    avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
    util_avg_heur[testind] = paramdict['baseloss'] - avg_loss
    util_lo_heur[testind] = paramdict['baseloss'] - avg_loss_CI[1]
    util_hi_heur[testind] = paramdict['baseloss'] - avg_loss_CI[0]
    print(des_heur)
    print('Utility at ' + str(testarr[testind]) + ' tests, Heuristic: ' + str(util_avg_heur[testind]))
    # Uniform utility
    des_unif = util.round_design_low(np.ones(numTN) / numTN, testarr[testind]) / testarr[testind]
    currlosslist = sampf.sampling_plan_loss_list(des_unif, testarr[testind], csdict_expl, paramdict)
    avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
    util_avg_unif[testind] = paramdict['baseloss'] - avg_loss
    util_lo_unif[testind] = paramdict['baseloss'] - avg_loss_CI[1]
    util_hi_unif[testind] = paramdict['baseloss'] - avg_loss_CI[0]
    print(des_unif)
    print('Utility at ' + str(testarr[testind]) + ' tests, Uniform: ' + str(util_avg_unif[testind]))
    # Rudimentary utility
    des_rudi = util.round_design_low(np.divide(np.sum(Nfam, axis=1), np.sum(Nfam)), testarr[testind]) / testarr[testind]
    currlosslist = sampf.sampling_plan_loss_list(des_rudi, testarr[testind], csdict_expl, paramdict)
    avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
    util_avg_rudi[testind] = paramdict['baseloss'] - avg_loss
    util_lo_rudi[testind] = paramdict['baseloss'] - avg_loss_CI[1]
    util_hi_rudi[testind] = paramdict['baseloss'] - avg_loss_CI[0]
    print(des_rudi)
    print('Utility at ' + str(testarr[testind]) + ' tests, Rudimentary: ' + str(util_avg_rudi[testind]))
    if plotupdate:
        util_avg_arr = np.vstack((util_avg_heur, util_avg_unif, util_avg_rudi))
        util_hi_arr = np.vstack((util_hi_heur, util_hi_unif, util_hi_rudi))
        util_lo_arr = np.vstack((util_lo_heur, util_lo_unif, util_lo_rudi))
        # Plot
        util.plot_marg_util_CI(util_avg_arr, util_hi_arr, util_lo_arr, testmax=testmax, testint=testint,
                               titlestr='Exploratory Setting, comparison with other approaches')

# Store matrices
np.save(os.path.join('casestudyoutputs', '31MAY', 'util_avg_arr_expl'), util_avg_arr)
np.save(os.path.join('casestudyoutputs', '31MAY', 'util_hi_arr_expl'), util_hi_arr)
np.save(os.path.join('casestudyoutputs', '31MAY', 'util_lo_arr_expl'), util_lo_arr)

