# -*- coding: utf-8 -*-
'''
Script that generates data and utility outputs for the example provided in Section 4.3 of the article "Measuring
sampling plan utility in post-marketing surveillance of supply chains." The data from the case study of this article
are not eligible to be published; however, the computations provided here mirror those used with the case study data.
Computations require use of the logistigate package, available at https://github.com/eugenewickett/logistigate/.
There are comments throughout the script that provide explanation of these computations. This script produces the
plots shown in Section 4.3.
'''

from logistigate.logistigate import utilities as util # Pull from the submodule "develop" branch
from logistigate.logistigate import methods, lg
from logistigate.logistigate import lossfunctions as lf
from logistigate.logistigate import samplingplanfunctions as sampf
import numpy as np
import os
from numpy.random import choice
import matplotlib.pyplot as plt

# Prior tests and results
N = np.array([[7, 5], [0, 3], [3, 4], [8, 3]], dtype=float)
Y = np.array([[3, 1], [0, 0], [0, 0], [2, 1]], dtype=float)
# Number of test nodes and supply nodes
(numTN, numSN) = N.shape
# Diagnostic sensitivity and specificity
s, r = 0.9, 0.95
# Create a logistigate data dictionary object for accessing prior data
exdict = util.initDataDict(N, Y, diagSens=s, diagSpec=r)
# Specify a prior and parameters for MCMC generation; see Wickett, et al. (2023) as referenced in the manuscript
exdict['prior'] = methods.prior_normal()
exdict['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
numdraws = 20000
exdict['numPostSamples'] = numdraws
np.random.seed(10) # To replicate draws later
# Generate and plot MCMC draws
exdict = methods.GeneratePostSamples(exdict)
util.plotPostSamples(exdict, 'int90')

# Use a sourcing matrix capturing supply-chain patterns; see Wickett, et al. (2023)
Q = np.array([[0.5, 0.5], [0.2, 0.8], [0.35, 0.65], [0.6, 0.4]])
exdict.update({'Q': Q})

# Specify the three sampling plans
des1 = np.array([0., 1., 0., 0.])   # Least Tested
des2 = np.ones(numTN) / numTN       # Uniform
des3 = np.array([0.5, 0., 0., 0.5]) # Highest SFPs
des_list = [des1, des2, des3]
# Maximum budget and considered test interval
testmax, testint = 40, 4
testarr = np.arange(testint, testmax + testint, testint)

# Build a dictionary storing loss specifications
# This loss uses an assessment score and risk
paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=1., riskthreshold=0.2, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN))

# Specify parameters for the efficient utility estimation algorithm
#   'truth' draws correspond to h_0, 'data' draws correspond to h_1
numtruthdraws, numdatadraws = 7500, 7000
# Randomly select MCMC draws for truth and data draws
np.random.seed(15) # Ensure replicability
truthdraws, datadraws = util.distribute_truthdata_draws(exdict['postSamples'], numtruthdraws, numdatadraws)
paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})

# Get base loss, i.e., the loss when only the prior data are used
paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)
util.print_param_checks(paramdict) # Check of used parameters
# Initialize utility estimate vectors for each sampling plan
util_avg_1, util_hi_1, util_lo_1 = np.zeros((int(testmax / testint) + 1)), \
                                    np.zeros((int(testmax / testint) + 1)), \
                                    np.zeros((int(testmax / testint) + 1))
util_avg_2, util_hi_2, util_lo_2 = np.zeros((int(testmax / testint) + 1)), \
                                    np.zeros((int(testmax / testint) + 1)), \
                                    np.zeros((int(testmax / testint) + 1))
util_avg_3, util_hi_3, util_lo_3 = np.zeros((int(testmax / testint) + 1)), \
                                    np.zeros((int(testmax / testint) + 1)), \
                                    np.zeros((int(testmax / testint) + 1))
# Loop through each test value by the test interval, and estimate the utility under each plan
# NOTE: MAY TAKE UPWARDS OF THIRTY MINUTES
for testind in range(testarr.shape[0]):
    # Least Tested
    # Compile a list of h_1 losses, integrated over the h_0 truth draws
    currlosslist = sampf.sampling_plan_loss_list(des1, testarr[testind], exdict, paramdict)
    # Form confidence interval based on the loss list
    avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
    # Store utility estimate
    util_avg_1[testind+1] = paramdict['baseloss'] - avg_loss
    util_lo_1[testind+1] = paramdict['baseloss'] - avg_loss_CI[1]
    util_hi_1[testind+1] = paramdict['baseloss'] - avg_loss_CI[0]
    print('Utility at ' + str(testarr[testind]) + ' tests, Least Tested: ' + str(util_avg_1[testind+1]))
    # Uniform
    currlosslist = sampf.sampling_plan_loss_list(des2, testarr[testind], exdict, paramdict)
    avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
    util_avg_2[testind + 1] = paramdict['baseloss'] - avg_loss
    util_lo_2[testind + 1] = paramdict['baseloss'] - avg_loss_CI[1]
    util_hi_2[testind + 1] = paramdict['baseloss'] - avg_loss_CI[0]
    print('Utility at ' + str(testarr[testind]) + ' tests, Uniform: ' + str(util_avg_2[testind + 1]))
    # Highest SFPs
    currlosslist = sampf.sampling_plan_loss_list(des3, testarr[testind], exdict, paramdict)
    avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
    util_avg_3[testind + 1] = paramdict['baseloss'] - avg_loss
    util_lo_3[testind + 1] = paramdict['baseloss'] - avg_loss_CI[1]
    util_hi_3[testind + 1] = paramdict['baseloss'] - avg_loss_CI[0]
    print('Utility at ' + str(testarr[testind]) + ' tests, Highest SFPs: ' + str(util_avg_3[testind + 1]))

    # Plot
    util_avg_arr = np.vstack((util_avg_1, util_avg_2, util_avg_3))
    util_hi_arr = np.vstack((util_hi_1, util_hi_2, util_hi_3))
    util_lo_arr = np.vstack((util_lo_1, util_lo_2, util_lo_3))
    # Plot
    util.plot_marg_util_CI(util_avg_arr, util_hi_arr, util_lo_arr, testmax=testmax, testint=testint,
                           colors=['blue', 'red', 'green'], titlestr='Example supply chain',
                           labels=['Focused', 'Uniform', 'Adapted'])

util.plot_marg_util(util_avg_arr, testmax=testmax, testint=testint,
                           colors=['blue', 'red', 'green'], titlestr='Example supply chain',
                           labels=['Focused', 'Uniform', 'Adapted'])

# If needed, store utility estimates
np.save(os.path.join('casestudyoutputs', 'util_avg_arr_example_base'), util_avg_arr)
np.save(os.path.join('casestudyoutputs', 'util_hi_arr_example_base'), util_hi_arr)
np.save(os.path.join('casestudyoutputs', 'util_lo_arr_example_base'), util_lo_arr)

######################
# CHANGE THE LOSS SPECIFICATION AND REDO
######################
# Increase the underestimation penalty by a factor of 10, and repeat the calculation of
#   utility estimate vectors
paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=10, riskthreshold=0.2, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN))
numtruthdraws, numdatadraws = 7500, 2000
np.random.seed(15)
truthdraws, datadraws = util.distribute_truthdata_draws(exdict['postSamples'], numtruthdraws, numdatadraws)
paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)

util.print_param_checks(paramdict)

util_avg_1, util_hi_1, util_lo_1 = np.zeros((int(testmax / testint) + 1)), \
                                    np.zeros((int(testmax / testint) + 1)), \
                                    np.zeros((int(testmax / testint) + 1))
util_avg_2, util_hi_2, util_lo_2 = np.zeros((int(testmax / testint) + 1)), \
                                    np.zeros((int(testmax / testint) + 1)), \
                                    np.zeros((int(testmax / testint) + 1))
util_avg_3, util_hi_3, util_lo_3 = np.zeros((int(testmax / testint) + 1)), \
                                    np.zeros((int(testmax / testint) + 1)), \
                                    np.zeros((int(testmax / testint) + 1))
for testind in range(testarr.shape[0]):
    # Least Tested
    currlosslist = sampf.sampling_plan_loss_list(des1, testarr[testind], exdict, paramdict)
    avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
    util_avg_1[testind+1] = paramdict['baseloss'] - avg_loss
    util_lo_1[testind+1] = paramdict['baseloss'] - avg_loss_CI[1]
    util_hi_1[testind+1] = paramdict['baseloss'] - avg_loss_CI[0]
    print('Utility at ' + str(testarr[testind]) + ' tests, Focused: ' + str(util_avg_1[testind+1]))
    # Uniform
    currlosslist = sampf.sampling_plan_loss_list(des2, testarr[testind], exdict, paramdict)
    avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
    util_avg_2[testind + 1] = paramdict['baseloss'] - avg_loss
    util_lo_2[testind + 1] = paramdict['baseloss'] - avg_loss_CI[1]
    util_hi_2[testind + 1] = paramdict['baseloss'] - avg_loss_CI[0]
    print('Utility at ' + str(testarr[testind]) + ' tests, Uniform: ' + str(util_avg_2[testind + 1]))
    # Highest SFPs
    currlosslist = sampf.sampling_plan_loss_list(des3, testarr[testind], exdict, paramdict)
    avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
    util_avg_3[testind + 1] = paramdict['baseloss'] - avg_loss
    util_lo_3[testind + 1] = paramdict['baseloss'] - avg_loss_CI[1]
    util_hi_3[testind + 1] = paramdict['baseloss'] - avg_loss_CI[0]
    print('Utility at ' + str(testarr[testind]) + ' tests, Adapted: ' + str(util_avg_3[testind + 1]))

    # Plot
    util_avg_arr = np.vstack((util_avg_1, util_avg_2, util_avg_3))
    util_hi_arr = np.vstack((util_hi_1, util_hi_2, util_hi_3))
    util_lo_arr = np.vstack((util_lo_1, util_lo_2, util_lo_3))
    # Plot
    util.plot_marg_util_CI(util_avg_arr, util_hi_arr, util_lo_arr, testmax=testmax, testint=testint,
                           colors=['blue', 'red', 'green'], titlestr='Example supply chain, loss change',
                           labels=['Focused', 'Uniform', 'Adapted'])

util.plot_marg_util(util_avg_arr, testmax=testmax, testint=testint,
                           colors=['blue', 'red', 'green'], titlestr='Example supply chain, loss change',
                           labels=['Focused', 'Uniform', 'Adapted'])

np.save(os.path.join('casestudyoutputs', 'util_avg_arr_example_adj'), util_avg_arr)
np.save(os.path.join('casestudyoutputs', 'util_hi_arr_example_adj'), util_hi_arr)
np.save(os.path.join('casestudyoutputs', 'util_lo_arr_example_adj'), util_lo_arr)
