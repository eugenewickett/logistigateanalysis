from logistigate.logistigate import utilities as util # Pull from the submodule "develop" branch
from logistigate.logistigate import methods
from logistigate.logistigate.priors import prior_normal_assort
from logistigate.logistigate import lossfunctions as lf
from logistigate.logistigate import samplingplanfunctions as sampf

import os
import pickle
import time

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import pandas as pd
import numpy as np
from numpy.random import choice
import random
import scipy.stats as sps
import scipy.special as spsp
import scipy.optimize as spo

numTN, numSN = 2, 2
N = np.array([[1.,0.], [1.,0.]])
Y = np.array([[0., 0.], [0., 0.]])
TNnames = ['TN1', 'TN2']
SNnames = ['SN1', 'SN2']
dataTbl = [['TN1', 'SN1', 0],['TN2', 'SN1', 0]]
testdatadict = {'dataTbl':dataTbl, 'type':'Tracked', 'TNnames':TNnames, 'SNnames':SNnames}
# Set up logistigate dictionary
lgdict = util.initDataDict(N, Y)
lgdict.update({'TNnames':TNnames, 'SNnames':SNnames})

SNpriorMean = np.repeat(spsp.logit(0.1), numSN)
TNpriorMean = np.repeat(spsp.logit(0.1), numTN)
# Concatenate prior means
priorMean = np.concatenate((SNpriorMean, TNpriorMean))
TNvar, SNvar = 2., 2.  # Variances for use with prior; supply nodes are wider due to unknown risk assessments
priorCovar = np.diag(np.concatenate((np.repeat(SNvar, numSN), np.repeat(TNvar, numTN))))
priorObj = prior_normal_assort(priorMean, priorCovar)
lgdict['prior'] = priorObj

# MCMC
numdraws = 10000
lgdict['numPostSamples'] = numdraws
lgdict['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
lgdict = methods.GeneratePostSamples(lgdict, maxTime=5000)

util.plotPostSamples(lgdict, 'int90')

lgdict.update({'Q':np.array([[1.,0.],[1.,0.]])})

# Utility
paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN))

# Set MCMC draws to use in fast algorithm
numtruthdraws, numdatadraws = 10000, 1000
# Get random subsets for truth and data draws
np.random.seed(56)
truthdraws, datadraws = util.distribute_truthdata_draws(lgdict['postSamples'], numtruthdraws, numdatadraws)
paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
# Get base loss
paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)

util.print_param_checks(paramdict)


def getUtilityEstimate(n, lgdict, paramdict, zlevel=0.95):
    """
    Return a utility estimate average and confidence interval for allocation array n
    """
    testnum = int(np.sum(n))
    des = n/testnum
    currlosslist = sampf.sampling_plan_loss_list(des, testnum, lgdict, paramdict)
    currloss_avg, currloss_CI = sampf.process_loss_list(currlosslist, zlevel=zlevel)
    return paramdict['baseloss'] - currloss_avg, (paramdict['baseloss']-currloss_CI[1], paramdict['baseloss']-currloss_CI[0])

n = np.array([1,0])
util, util_CI = getUtilityEstimate(n, lgdict, paramdict)

# Iterate through pairs of samples and plot
utiltotallist, utiltotalCIlist = [0], [(0,0)]
utilsumlist, utilsumCIlist = [0], [(0,0)]
for i in range(1, 100):
    n = np.array([i,0])
    util1, util1_CI = getUtilityEstimate(n, lgdict, paramdict)
    n = np.array([0, i])
    util2, util2_CI = getUtilityEstimate(n, lgdict, paramdict)
    utilsumlist.append(util1+util2)
    utilsumCIlist.append((util1_CI[0]+util2_CI[0], util1_CI[1]+util2_CI[1]))
    # Holistic utility
    n = np.array([i, i])
    utiltotal, utiltotal_CI = getUtilityEstimate(n, lgdict, paramdict)
    utiltotallist.append(utiltotal)
    utiltotalCIlist.append(utiltotal_CI)
    # Update plotting lists
    utilsumCIlistlower = [x[0] for x in utilsumCIlist]
    utilsumCIlistupper = [x[1] for x in utilsumCIlist]
    utiltotalCIlistlower = [x[0] for x in utiltotalCIlist]
    utiltotalCIlistupper = [x[1] for x in utiltotalCIlist]
    # Plot
    plt.plot(range(0,i + 1), utilsumlist, color='blue',linewidth=3)
    plt.plot(range(0,i + 1), utiltotallist, color='black',linewidth=3)
    plt.plot(range(0,i + 1), utilsumCIlistlower, color='lightblue',linestyle='dashed')
    plt.plot(range(0,i + 1), utilsumCIlistupper, color='lightblue',linestyle='dashed')
    plt.plot(range(0,i + 1), utiltotalCIlistlower, color='gray',linestyle='dashed')
    plt.plot(range(0,i + 1), utiltotalCIlistupper, color='gray',linestyle='dashed')
    plt.legend(['$U(n_1)+U(n_2)$','$U(n_1+n_2)$'])
    plt.title('2 TNs, 1 SN')
    plt.show()

##################
##################
##################
# Now do with 5 supply nodes
numTN, numSN = 2, 7
N = np.vstack((np.ones(numSN),np.ones(numSN)))
Y = np.vstack((np.zeros(numSN),np.zeros(numSN)))
TNnames = ['TN1', 'TN2']
SNnames = ['SN'+str(i) for i in range(1,numSN+1)]
dataTbl = [ [tn, sn, 0] for tn in TNnames for sn in SNnames]
testdatadict = {'dataTbl':dataTbl, 'type':'Tracked', 'TNnames':TNnames, 'SNnames':SNnames}
# Set up logistigate dictionary
lgdict = util.initDataDict(N, Y)
lgdict.update({'TNnames':TNnames, 'SNnames':SNnames})

SNpriorMean = np.repeat(spsp.logit(0.1), numSN)
TNpriorMean = np.repeat(spsp.logit(0.1), numTN)
# Concatenate prior means
priorMean = np.concatenate((SNpriorMean, TNpriorMean))
TNvar, SNvar = 2., 2.  # Variances for use with prior; supply nodes are wider due to unknown risk assessments
priorCovar = np.diag(np.concatenate((np.repeat(SNvar, numSN), np.repeat(TNvar, numTN))))
priorObj = prior_normal_assort(priorMean, priorCovar)
lgdict['prior'] = priorObj

# MCMC
numdraws = 10000
lgdict['numPostSamples'] = numdraws
lgdict['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
lgdict = methods.GeneratePostSamples(lgdict, maxTime=5000)

util.plotPostSamples(lgdict, 'int90')

Qvec = np.ones(numSN)/numSN
lgdict.update({'Q':np.vstack((Qvec,Qvec)) })

# Utility
paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN))

# Set MCMC draws to use in fast algorithm
numtruthdraws, numdatadraws = 10000, 1000
# Get random subsets for truth and data draws
np.random.seed(56)
truthdraws, datadraws = util.distribute_truthdata_draws(lgdict['postSamples'], numtruthdraws, numdatadraws)
paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
# Get base loss
paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)

util.print_param_checks(paramdict)


n = np.array([1,0])
util, util_CI = getUtilityEstimate(n, lgdict, paramdict)

# Iterate through pairs of samples and plot
utiltotallist, utiltotalCIlist = [0], [(0,0)]
utilsumlist, utilsumCIlist = [0], [(0,0)]
for i in range(5, 101, 5):
    print('On '+str(i) +' tests...')
    n = np.array([i,0])
    util1, util1_CI = getUtilityEstimate(n, lgdict, paramdict)
    n = np.array([0, i])
    util2, util2_CI = getUtilityEstimate(n, lgdict, paramdict)
    utilsumlist.append(util1+util2)
    utilsumCIlist.append((util1_CI[0]+util2_CI[0], util1_CI[1]+util2_CI[1]))
    # Holistic utility
    n = np.array([i, i])
    utiltotal, utiltotal_CI = getUtilityEstimate(n, lgdict, paramdict)
    utiltotallist.append(utiltotal)
    utiltotalCIlist.append(utiltotal_CI)
    # Update plotting lists
    utilsumCIlistlower = [x[0] for x in utilsumCIlist]
    utilsumCIlistupper = [x[1] for x in utilsumCIlist]
    utiltotalCIlistlower = [x[0] for x in utiltotalCIlist]
    utiltotalCIlistupper = [x[1] for x in utiltotalCIlist]
    # Plot
    plt.plot(range(0,i + 1,5), utilsumlist, color='blue',linewidth=3)
    plt.plot(range(0,i + 1,5), utiltotallist, color='black',linewidth=3)
    plt.plot(range(0,i + 1,5), utilsumCIlistlower, color='lightblue',linestyle='dashed')
    plt.plot(range(0,i + 1,5), utilsumCIlistupper, color='lightblue',linestyle='dashed')
    plt.plot(range(0,i + 1,5), utiltotalCIlistlower, color='gray',linestyle='dashed')
    plt.plot(range(0,i + 1,5), utiltotalCIlistupper, color='gray',linestyle='dashed')
    plt.legend(['$U(n_1)+U(n_2)$','$U(n_1+n_2)$'])
    plt.title('2 TNs, 7 SNs')
    plt.show()