"""
Code for the example supply chain of the introduction/Section 3.2 of the paper.
"""

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

N = np.array([[7, 5], [0, 3], [3, 4], [8, 3]], dtype=float)
Y = np.array([[3, 1], [0, 0], [0, 0], [2, 1]], dtype=float)
(numTN, numSN) = N.shape
s, r = 0.9, 0.95
exdict = util.initDataDict(N, Y, diagSens=s, diagSpec=r)
exdict['prior'] = methods.prior_normal()
exdict['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
numdraws = 20000
exdict['numPostSamples'] = numdraws
np.random.seed(10) # To replicate draws later
exdict = methods.GeneratePostSamples(exdict)
util.plotPostSamples(exdict, 'int90')

# Sourcing matrix; EVALUATE
Q = np.array([[0.5, 0.5], [0.2, 0.8], [0.35, 0.65], [0.6, 0.4]])
exdict.update({'Q': Q})

# Designs
des1 = np.array([0., 1., 0., 0.])   # Focused
des2 = np.ones(numTN) / numTN       # Balanced
des3 = np.array([0.5, 0., 0., 0.5]) # Adapted
des_list = [des1, des2, des3]
testmax, testint = 60, 4
testarr = np.arange(testint, testmax + testint, testint)

paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=1., riskthreshold=0.2, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN))

numtruthdraws, numdatadraws = 7500, 7000
np.random.seed(15)
truthdraws, datadraws = util.distribute_truthdata_draws(exdict['postSamples'], numtruthdraws, numdatadraws)
paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})

# Get base loss
paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)

util.print_param_checks(paramdict) # Check of used parameters

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
    # Focused
    currlosslist = sampf.sampling_plan_loss_list(des1, testarr[testind], exdict, paramdict)
    avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.9)
    util_avg_1[testind+1] = paramdict['baseloss'] - avg_loss
    util_lo_1[testind+1] = paramdict['baseloss'] - avg_loss_CI[1]
    util_hi_1[testind+1] = paramdict['baseloss'] - avg_loss_CI[0]
    print('Utility at ' + str(testarr[testind]) + ' tests, Focused: ' + str(util_avg_1[testind+1]))
    # Uniform
    currlosslist = sampf.sampling_plan_loss_list(des2, testarr[testind], exdict, paramdict)
    avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.9)
    util_avg_2[testind + 1] = paramdict['baseloss'] - avg_loss
    util_lo_2[testind + 1] = paramdict['baseloss'] - avg_loss_CI[1]
    util_hi_2[testind + 1] = paramdict['baseloss'] - avg_loss_CI[0]
    print('Utility at ' + str(testarr[testind]) + ' tests, Uniform: ' + str(util_avg_2[testind + 1]))
    # Adapted
    currlosslist = sampf.sampling_plan_loss_list(des3, testarr[testind], exdict, paramdict)
    avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.9)
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
                           colors=['blue','red','green'], titlestr='Example supply chain',
                           labels=['Focused', 'Uniform', 'Adapted'])

util.plot_marg_util(util_avg_arr, testmax=testmax, testint=testint,
                           colors=['blue','red','green'], titlestr='Example supply chain',
                           labels=['Focused', 'Uniform', 'Adapted'])

np.save(os.path.join('casestudyoutputs', '31MAY', 'util_avg_arr_example_base'), util_avg_arr)
np.save(os.path.join('casestudyoutputs', '31MAY', 'util_hi_arr_example_base'), util_hi_arr)
np.save(os.path.join('casestudyoutputs', '31MAY', 'util_lo_arr_example_base'), util_lo_arr)

######################
# CHANGE THE LOSS SPECIFICATION
######################
paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=15, riskthreshold=0.2, riskslope=0.6,
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
    # Focused
    currlosslist = sampf.sampling_plan_loss_list(des1, testarr[testind], exdict, paramdict)
    avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.9)
    util_avg_1[testind+1] = paramdict['baseloss'] - avg_loss
    util_lo_1[testind+1] = paramdict['baseloss'] - avg_loss_CI[1]
    util_hi_1[testind+1] = paramdict['baseloss'] - avg_loss_CI[0]
    print('Utility at ' + str(testarr[testind]) + ' tests, Focused: ' + str(util_avg_1[testind+1]))
    # Uniform
    currlosslist = sampf.sampling_plan_loss_list(des2, testarr[testind], exdict, paramdict)
    avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.9)
    util_avg_2[testind + 1] = paramdict['baseloss'] - avg_loss
    util_lo_2[testind + 1] = paramdict['baseloss'] - avg_loss_CI[1]
    util_hi_2[testind + 1] = paramdict['baseloss'] - avg_loss_CI[0]
    print('Utility at ' + str(testarr[testind]) + ' tests, Uniform: ' + str(util_avg_2[testind + 1]))
    # Adapted
    currlosslist = sampf.sampling_plan_loss_list(des3, testarr[testind], exdict, paramdict)
    avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.9)
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
                           colors=['blue','red','green'], titlestr='Example supply chain, loss change',
                           labels=['Focused', 'Uniform', 'Adapted'])



util.plot_marg_util(util_avg_arr, testmax=testmax, testint=testint,
                           colors=['blue','red','green'], titlestr='Example supply chain, loss change',
                           labels=['Focused', 'Uniform', 'Adapted'])

np.save(os.path.join('casestudyoutputs', '31MAY', 'util_avg_arr_example_adj'), util_avg_arr)
np.save(os.path.join('casestudyoutputs', '31MAY', 'util_hi_arr_example_adj'), util_hi_arr)
np.save(os.path.join('casestudyoutputs', '31MAY', 'util_lo_arr_example_adj'), util_lo_arr)