from logistigate.logistigate import utilities as util # Pull from the submodule "develop" branch
from logistigate.logistigate import methods
from logistigate.logistigate.priors import prior_normal_assort
from logistigate.logistigate import lossfunctions as lf
from logistigate.logistigate import samplingplanfunctions as sampf

import os
import numpy as np
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

# Region catchment proportions, for market terms
TNcach = np.array([0.17646, 0.05752, 0.09275, 0.09488, 0.17695, 0.22799, 0.07805, 0.0954])
tempQ = csdict_expl['N'][:4] / np.sum(csdict_expl['N'][:4], axis=1).reshape(4, 1)
tempTNcach = TNcach[:4] / np.sum(TNcach[:4])
SNcach = np.matmul(tempTNcach, tempQ)

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
numdraws = 75000
csdict_expl['numPostSamples'] = numdraws
np.random.seed(1000) # To replicate draws later
csdict_expl = methods.GeneratePostSamples(csdict_expl)

# Loss specification; use market term
paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.concatenate((SNcach, TNcach)))

# Set limits of data collection and intervals for calculation
testmax, testint = 400, 10
testarr = np.arange(testint, testmax + testint, testint)

# Set MCMC draws to use in fast algorithm
numtruthdraws, numdatadraws = 75000, 2000
# Get random subsets for truth and data draws
np.random.seed(444)
truthdraws, datadraws = util.distribute_truthdata_draws(csdict_expl['postSamples'], numtruthdraws, numdatadraws)
paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
# Get base loss
paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)

util.print_param_checks(paramdict)  # Check of used parameters

alloc, util_avg, util_hi, util_lo = sampf.get_greedy_allocation(csdict_expl, testmax, testint, paramdict,
                                                                printupdate=True, plotupdate=False)
'''
[5,4,6,7,5,4,5,4,7,
 6,5,1,4,7,0,5,6,1
 0,5,7,1,6,7,
 ]
[(0.02224712932231554, 0.02313869789825007),(0.04469460727097013, 0.04574405032222573),(0.05470162662897646, 0.05582343369082718),
(0.06419605712606549, 0.06536766610098341),(0.07272337525383174, 0.07384903087491865),(0.08048076792275097, 0.08156953628668942),
(0.08551633367747924, 0.08655897081285502),(0.08987600134260038, 0.09090151153185538),(0.09459705213823186, 0.09563788610071214),
(0.09834317463494222, 0.09944099768137668),(0.10181810982353162, 0.10290004115132934),(0.1055390615223297, 0.10669517681941101),
(0.10890262529415096, 0.11007436831322759),(0.11159103421990726, 0.11288005037182083),(0.11477042317716801, 0.11611459184258027),
(0.11802242743602909, 0.11941152549039802),(0.12163986801837101, 0.12316919576285895),(0.12426883699103858, 0.12594006797632123),
(0.12725651971359211, 0.12894394197678774),(0.130244556143302, 0.13209617159942566),(0.132793734566584, 0.13474564722959004),
(0.1361857713456384, 0.13831094549960496),(0.13901684868503528, 0.14116619287151638),(0.1416482725994258, 0.1440063907983088),

]
'''

np.save(os.path.join('casestudyoutputs', 'exploratory', 'expl_market_alloc'), alloc)
np.save(os.path.join('casestudyoutputs', 'exploratory', 'expl_market_util_avg'), util_avg)
np.save(os.path.join('casestudyoutputs', 'exploratory', 'expl_market_util_hi'), util_hi)
np.save(os.path.join('casestudyoutputs', 'exploratory', 'expl_market_util_lo'), util_lo)

# Evaluate utility for uniform and rudimentary
util_avg_unif, util_hi_unif, util_lo_unif = np.zeros((int(testmax / testint) + 1)), \
                                            np.zeros((int(testmax / testint) + 1)), \
                                            np.zeros((int(testmax / testint) + 1))
util_avg_rudi, util_hi_rudi, util_lo_rudi = np.zeros((int(testmax / testint) + 1)), \
                                            np.zeros((int(testmax / testint) + 1)), \
                                            np.zeros((int(testmax / testint) + 1))
plotupdate = True
for testind in range(testarr.shape[0]):
    # Uniform utility
    des_unif = util.round_design_low(np.ones(numTN) / numTN, testarr[testind]) / testarr[testind]
    currlosslist = sampf.sampling_plan_loss_list(des_unif, testarr[testind], csdict_expl, paramdict)
    avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
    util_avg_unif[testind+1] = paramdict['baseloss'] - avg_loss
    util_lo_unif[testind+1] = paramdict['baseloss'] - avg_loss_CI[1]
    util_hi_unif[testind+1] = paramdict['baseloss'] - avg_loss_CI[0]
    print(des_unif)
    print('Utility at ' + str(testarr[testind]) + ' tests, Uniform: ' + str(util_avg_unif[testind+1]))
    # Rudimentary utility
    des_rudi = util.round_design_low(np.divide(np.sum(Nexpl, axis=1), np.sum(Nexpl)), testarr[testind]) / testarr[testind]
    currlosslist = sampf.sampling_plan_loss_list(des_rudi, testarr[testind], csdict_expl, paramdict)
    avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
    util_avg_rudi[testind+1] = paramdict['baseloss'] - avg_loss
    util_lo_rudi[testind+1] = paramdict['baseloss'] - avg_loss_CI[1]
    util_hi_rudi[testind+1] = paramdict['baseloss'] - avg_loss_CI[0]
    print(des_rudi)
    print('Utility at ' + str(testarr[testind]) + ' tests, Rudimentary: ' + str(util_avg_rudi[testind+1]))
    if plotupdate:
        util_avg_arr = np.vstack((util_avg_unif, util_avg_rudi))
        util_hi_arr = np.vstack((util_hi_unif, util_hi_rudi))
        util_lo_arr = np.vstack((util_lo_unif, util_lo_rudi))
        # Plot
        util.plot_marg_util_CI(util_avg_arr, util_hi_arr, util_lo_arr, testmax=testmax, testint=testint,
                               titlestr='Exploratory Setting with Market Term, comparison with other approaches')

# Store matrices
np.save(os.path.join('casestudyoutputs', 'exploratory', 'util_avg_arr_expl_market'), util_avg_arr)
np.save(os.path.join('casestudyoutputs', 'exploratory', 'util_hi_arr_expl_market'), util_hi_arr)
np.save(os.path.join('casestudyoutputs', 'exploratory', 'util_lo_arr_expl_market'), util_lo_arr)

targind = 5 # where do we want to gauge budget savings?
targval = util_avg_arr[0][targind]

# Uniform
kInd = next(x for x, val in enumerate(util_avg_arr[1].tolist()) if val > targval)
unif_saved = round((targval - util_avg_arr[1][kInd - 1]) / (util_avg_arr[1][kInd] - util_avg_arr[1][kInd - 1]) *\
                      testint) + (kInd - 1) * testint - targind*testint
print(unif_saved)  #
# Rudimentary
kInd = next(x for x, val in enumerate(util_avg_arr[2].tolist()) if val > targval)
rudi_saved = round((targval - util_avg_arr[2][kInd - 1]) / (util_avg_arr[2][kInd] - util_avg_arr[2][kInd - 1]) *\
                      testint) + (kInd - 1) * testint - targind*testint
print(rudi_saved)  #