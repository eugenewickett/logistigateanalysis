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

# Region catchment proportions, for market terms
TNcach = np.array([0.17646, 0.05752, 0.09275, 0.09488])
TNcach = TNcach[:4] / np.sum(TNcach[:4])
SNcach = np.matmul(TNcach, csdict_fam['Q'])

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
numdraws = 10000
csdict_fam['numPostSamples'] = numdraws
np.random.seed(1000) # To replicate draws later
csdict_fam = methods.GeneratePostSamples(csdict_fam)

# Loss specification
paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.concatenate((SNcach, TNcach)))

# Set limits of data collection and intervals for calculation
testmax, testint = 400, 10
testarr = np.arange(testint, testmax + testint, testint)

# Set MCMC draws to use in fast algorithm
numtruthdraws, numdatadraws = 4000, 1500
# Get random subsets for truth and data draws
truthdraws, datadraws = util.distribute_truthdata_draws(csdict_fam['postSamples'], numtruthdraws, numdatadraws)
paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
# Get base loss
paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)

util.print_param_checks(paramdict) # Check of used parameters
util_avg, util_hi, util_lo = sampf.get_opt_marg_util_nodes(csdict_fam, testmax, testint, paramdict, zlevel=0.95,
                            printupdate=True, plotupdate=True) # Wrapper function for utility at all test nodes
# Plot
util.plot_marg_util_CI(util_avg, margutilarr_hi=util_hi, margutilarr_lo=util_lo, testmax=testmax, testint=testint,
                               titlestr='Familiar Setting')
# Store matrices
np.save(os.path.join('casestudyoutputs', '31MAY', 'util_avg_fam_market'), util_avg)
np.save(os.path.join('casestudyoutputs', '31MAY', 'util_hi_fam_market'), util_hi)
np.save(os.path.join('casestudyoutputs', '31MAY', 'util_lo_fam_market'), util_lo)

#util_avg = np.load(os.path.join('casestudyoutputs', '31MAY', 'margutil_avg_arr_fam.npy'))
#util_hi = np.load(os.path.join('casestudyoutputs', '31MAY', 'margutil_hi_arr_fam.npy'))
#util_lo = np.load(os.path.join('casestudyoutputs', '31MAY', 'margutil_lo_arr_fam.npy'))

# Form allocation
allocArr, objValArr = sampf.smooth_alloc_forward(util_avg)
util.plot_plan(allocArr, paramlist=[str(i) for i in np.arange(testint, testmax + 1, testint)], testint=testint,
          labels=csdict_fam['TNnames'], titlestr='Familiar Setting', allocmax=250,
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
for testind in range(testarr.shape[0]):
    # Heuristic utility
    des_heur = allocArr[:, testind] / np.sum(allocArr[:, testind])
    print(des_heur)
    currlosslist = sampf.sampling_plan_loss_list(des_heur, testarr[testind], csdict_fam, paramdict)
    avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
    util_avg_heur[testind+1] = paramdict['baseloss'] - avg_loss
    util_lo_heur[testind+1] = paramdict['baseloss'] - avg_loss_CI[1]
    util_hi_heur[testind+1] = paramdict['baseloss'] - avg_loss_CI[0]
    print('Utility at ' + str(testarr[testind]) + ' tests, Heuristic: ' + str(util_avg_heur[testind]))
    # Uniform utility
    des_unif = util.round_design_low(np.ones(numTN) / numTN, testarr[testind]) / testarr[testind]
    print(des_unif)
    currlosslist = sampf.sampling_plan_loss_list(des_unif, testarr[testind], csdict_fam, paramdict)
    avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
    util_avg_unif[testind+1] = paramdict['baseloss'] - avg_loss
    util_lo_unif[testind+1] = paramdict['baseloss'] - avg_loss_CI[1]
    util_hi_unif[testind+1] = paramdict['baseloss'] - avg_loss_CI[0]
    print('Utility at ' + str(testarr[testind]) + ' tests, Uniform: ' + str(util_avg_unif[testind]))
    # Rudimentary utility
    des_rudi = util.round_design_low(np.divide(np.sum(Nfam, axis=1), np.sum(Nfam)), testarr[testind]) / testarr[testind]
    print(des_rudi)
    currlosslist = sampf.sampling_plan_loss_list(des_rudi, testarr[testind], csdict_fam, paramdict)
    avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
    util_avg_rudi[testind+1] = paramdict['baseloss'] - avg_loss
    util_lo_rudi[testind+1] = paramdict['baseloss'] - avg_loss_CI[1]
    util_hi_rudi[testind+1] = paramdict['baseloss'] - avg_loss_CI[0]
    print('Utility at ' + str(testarr[testind]) + ' tests, Rudimentary: ' + str(util_avg_unif[testind]))

    util_avg_arr = np.vstack((util_avg_heur, util_avg_unif, util_avg_rudi))
    util_hi_arr = np.vstack((util_hi_heur, util_hi_unif, util_hi_rudi))
    util_lo_arr = np.vstack((util_lo_heur, util_lo_unif, util_lo_rudi))
    # Plot
    util.plot_marg_util_CI(util_avg_arr, util_hi_arr, util_lo_arr, testmax=testmax, testint=testint,
                               titlestr='Familiar Setting with Market Term, comparison with other approaches')

# Store matrices
np.save(os.path.join('casestudyoutputs', '31MAY', 'util_avg_arr_fam_market'), util_avg_arr)
np.save(os.path.join('casestudyoutputs', '31MAY', 'util_hi_arr_fam_market'), util_hi_arr)
np.save(os.path.join('casestudyoutputs', '31MAY', 'util_lo_arr_fam_market'), util_lo_arr)
