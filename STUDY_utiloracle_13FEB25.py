"""
This study aims to inspect differences in oracle performance for the efficient and  importance sampling methods of
utility estimation, for different numbers of truth/data draws, using the data of the case study for the
orienteering paper.

PARTS
1: FOR FIXED ALLOCATION (IP-RP SOLUTION), RUN 10 ITERATIONS EACH OF EFFICIENT METHOD AND IMP SAMP METHOD
    AND STORE RESULTING CI FOR UTILITY ESTIMATE, USING DIFFERENT INITIAL MCMC DRAWS EACH TIME;
    USE {1, 2, 5, 10, 20} BATCHES
2: REPEAT FOR DIFFERENT FIXED ALLOCATION (B=1400)
3: INCLUDE INCORPORATION OF A DELTA FOR % OF EXTREME IMPORTANCE WEIGHTS TO DROP
4: EVALUATE CHOICES FOR DELTA
"""

from logistigate.logistigate import utilities as util  # Pull from the submodule "develop" branch
from logistigate.logistigate import methods
from logistigate.logistigate.priors import prior_normal_assort
from logistigate.logistigate import lossfunctions as lf
from logistigate.logistigate import samplingplanfunctions as sampf
from logistigate.logistigate import orienteering as opf

from operationalizedsamplingplans.senegalsetup import *

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
import itertools
import scipy.stats as sps
import scipy.special as spsp

import scipy.optimize as spo
from scipy.optimize import LinearConstraint
from scipy.optimize import milp

# Initialize some standard plot parameters
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.family"] = "serif"

# Pull data from newly constructed CSV files
dept_df, regcost_mat, regNames, deptNames, manufNames, numReg, testdatadict = GetSenegalCSVData()
(numTN, numSN) = testdatadict['N'].shape  # For later use

PrintDataSummary(testdatadict)  # Cursory data check

# Set up logistigate dictionary
lgdict = util.initDataDict(testdatadict['N'], testdatadict['Y'])
lgdict.update({'TNnames': deptNames, 'SNnames': manufNames})

# Set up priors for SFP rates at nodes
SetupSenegalPriors(lgdict)

# Set up MCMC
lgdict['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 1000, 'delta': 0.4}

# Add boostrap-sampled sourcing vectors for non-tested test nodes; 20 is the avg number of tests per visited dept
AddBootstrapQ(lgdict, numboot=int(np.sum(lgdict['N'])/np.count_nonzero(np.sum(lgdict['Q'], axis=1))), randseed=44)

# Loss specification
paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN))

##################
# PART 1: B=700
##################

numreps = 10  # Replications of different sets of initial MCMC draws
numbatcharr = [1, 2, 5, 10, 15]
store_baseloss = np.zeros((len(numbatcharr), numreps))
store_utilhi = np.zeros((2, len(numbatcharr), numreps))  # Efficient, then imp samp
store_utillo = np.zeros((2, len(numbatcharr), numreps))  # Efficient, then imp samp

deptList_IPRP = ['Dakar', 'Keur Massar', 'Pikine', 'Diourbel', 'Bambey', 'Mbacke', 'Fatick', 'Foundiougne', 'Gossas']
allocList_IPRP = [42, 21, 7, 9, 10, 7, 11, 10, 9]
n_IPRP = GetAllocVecFromLists(deptNames, deptList_IPRP, allocList_IPRP)

for currbatchind in range(len(numbatcharr)):
    numbatch = numbatcharr[currbatchind]
    for rep in range(numreps):
        # Retrieve previously generated MCMC draws, which are in batches of 5000; each batch takes up about 3MB
        RetrieveMCMCBatches(lgdict, numbatch,
                            os.path.join('operationalizedsamplingplans', 'numpy_objects', 'draws'),
                            maxbatchnum=50, rand=True, randseed=rep+currbatchind+22)
        # Set up utility estimation parameter dictionary with desired truth and data draws
        SetupUtilEstParamDict(lgdict, paramdict, numbatch*5000, 500, randseed=rep+currbatchind+106)
        util.print_param_checks(paramdict)
        store_baseloss[currbatchind, rep] = paramdict['baseloss']

        # EFFICIENT ESTIMATE
        _, util_IPRP_CI = getUtilityEstimate(n_IPRP, lgdict, paramdict, zlevel=0.95)
        store_utillo[0, currbatchind, rep] = util_IPRP_CI[0]
        store_utilhi[0, currbatchind, rep] = util_IPRP_CI[1]

        # IMP SAMP ESTIMATE
        _, util_IPRP_CI = sampf.getImportanceUtilityEstimate(n_IPRP, lgdict, paramdict,
                                                             numimportdraws=numbatch * 5000, impweightoutlierprop=0.00,
                                                             zlevel=0.95)
        store_utillo[1, currbatchind, rep] = util_IPRP_CI[0]
        store_utilhi[1, currbatchind, rep] = util_IPRP_CI[1]

        # Plot
        fig, ax = plt.subplots()
        fig.set_figheight(7)
        fig.set_figwidth(12)

        x = np.arange(numreps*len(numbatcharr)*2)
        flat_utillo = store_utillo.flatten()
        flat_utilhi = store_utilhi.flatten()
        CIavg = (flat_utillo + flat_utilhi) / 2
        #ax.plot(x, CIavg, marker='o', linestyle='None')
        ax.errorbar(x, CIavg, yerr=[CIavg-flat_utillo, flat_utilhi-CIavg],
                    fmt='o', ecolor='g', capthick=4)
        ax.set_title('95% CI for IP-RP solution under different parameters and estimation methods')
        #ax.grid('on')

        xticklist = ['' for j in range(numreps*len(numbatcharr)*2)]
        for currbatchnameind, currbatchname in enumerate(numbatcharr):
            xticklist[currbatchnameind * 10] = str(currbatchname) + ' batch\nEffic'
            xticklist[currbatchnameind * 10 + 50] = str(currbatchname) + ' batch\nImpSamp'
        plt.xticks(x, xticklist)  # little trick to get textual X labels instead of numerical
        plt.xlabel('Method and parameterization')
        plt.ylabel('Utility estimate')

        plt.ylim([0, 7])
        ax.tick_params(axis='x', labelsize=8)
        label_X = ax.xaxis.get_label()
        label_Y = ax.yaxis.get_label()
        label_X.set_style('italic')
        label_X.set_size(12)
        label_Y.set_style('italic')
        label_Y.set_size(12)

        plt.show()

        # Plot baseloss also
        fig, ax = plt.subplots()
        fig.set_figheight(7)
        fig.set_figwidth(12)

        x = np.arange(numreps * len(numbatcharr))
        flat_baseloss = store_baseloss.flatten()
        ax.plot(x, flat_baseloss, 'o', color='orange')
        ax.set_title('Base loss for IP-RP solution under different parameters and estimation methods\nB=700')

        xticklist = ['' for j in range(numreps * len(numbatcharr))]
        for currbatchnameind, currbatchname in enumerate(numbatcharr):
            xticklist[currbatchnameind * 10] = str(currbatchname) + ' batch\nEffic'
        plt.xticks(x, xticklist)  # little trick to get textual X labels instead of numerical
        plt.xlabel('Method and parameterization')
        plt.ylabel('Loss estimate')

        plt.ylim([0,13])
        ax.tick_params(axis='x', labelsize=8)
        label_X = ax.xaxis.get_label()
        label_Y = ax.yaxis.get_label()
        label_X.set_style('italic')
        label_X.set_size(12)
        label_Y.set_style('italic')
        label_Y.set_size(12)

        plt.show()

        # Save numpy objects
        np.save(os.path.join('studies', 'utiloracle_13FEB25', 'store_baseloss'), store_baseloss)
        np.save(os.path.join('studies', 'utiloracle_13FEB25', 'store_utillo'), store_utillo)
        np.save(os.path.join('studies', 'utiloracle_13FEB25', 'store_utilhi'), store_utilhi)

##################
# END PART 1
##################

##################
# PART 2: B=700, REMOVING 0.5% OUTLIERS FROM IMP SAMP
##################
# There is a lot of variance/inflation with the imp samp method; does removing largely weighted draws improve things?
store_utilhi_outlier = np.zeros((len(numbatcharr), numreps))  # Efficient, then imp samp
store_utillo_outlier = np.zeros((len(numbatcharr), numreps))  # Efficient, then imp samp

for currbatchind in range(len(numbatcharr)):
    numbatch = numbatcharr[currbatchind]
    for rep in range(numreps):
        # Retrieve previously generated MCMC draws, which are in batches of 5000; each batch takes up about 3MB
        RetrieveMCMCBatches(lgdict, numbatch,
                            os.path.join('operationalizedsamplingplans', 'numpy_objects', 'draws'),
                            maxbatchnum=50, rand=True, randseed=rep+currbatchind+22)
        # Set up utility estimation parameter dictionary with desired truth and data draws
        SetupUtilEstParamDict(lgdict, paramdict, numbatch*5000, 500, randseed=rep+currbatchind+106)
        util.print_param_checks(paramdict)

        # IMP SAMP ESTIMATE, WITH 0.5% HIGHEST IMPORTANCE WEIGHTS REMOVED
        _, util_IPRP_CI = sampf.getImportanceUtilityEstimate(n_IPRP, lgdict, paramdict,
                                                             numimportdraws=numbatch * 5000,
                                                             impweightoutlierprop=0.005,
                                                             zlevel=0.95)
        store_utillo_outlier[currbatchind, rep] = util_IPRP_CI[0]
        store_utilhi_outlier[currbatchind, rep] = util_IPRP_CI[1]

        # Plot
        fig, ax = plt.subplots()
        fig.set_figheight(7)
        fig.set_figwidth(16)

        x = np.arange(numreps * len(numbatcharr) * 3)
        flat_utillo = store_utillo.flatten()
        flat_utilhi = store_utilhi.flatten()
        CIavg = (flat_utillo + flat_utilhi) / 2
        flat_utillo = np.concatenate((flat_utillo, store_utillo_outlier.flatten()))
        flat_utilhi = np.concatenate((flat_utilhi, store_utilhi_outlier.flatten()))
        CIavg = np.concatenate((CIavg, (store_utillo_outlier.flatten()+store_utilhi_outlier.flatten())/2))

        ax.errorbar(x, CIavg, yerr=[CIavg - flat_utillo, flat_utilhi - CIavg],
                    fmt='o', ecolor='g', capthick=4)
        ax.set_title('95% CI for IP-RP solution under different parameters and estimation methods\nB=700')
        # ax.grid('on')

        xticklist = ['' for j in range(numreps * len(numbatcharr) * 3)]
        for currbatchnameind, currbatchname in enumerate(numbatcharr):
            xticklist[currbatchnameind * 10] = str(currbatchname) + ' batch\nEffic'
            xticklist[currbatchnameind * 10 + 50] = str(currbatchname) + ' batch\nImpSamp'
            xticklist[currbatchnameind * 10 + 100] = str(currbatchname) + ' batch\nImpSamp_Out'
        plt.xticks(x, xticklist)  # little trick to get textual X labels instead of numerical
        plt.xlabel('Method and parameterization')
        plt.ylabel('Utility estimate')

        plt.ylim([0, 7])
        ax.tick_params(axis='x', labelsize=8)
        label_X = ax.xaxis.get_label()
        label_Y = ax.yaxis.get_label()
        label_X.set_style('italic')
        label_X.set_size(12)
        label_Y.set_style('italic')
        label_Y.set_size(12)

        plt.show()

        np.save(os.path.join('studies', 'utiloracle_13FEB25', 'store_utillo_outlier'), store_utillo_outlier)
        np.save(os.path.join('studies', 'utiloracle_13FEB25', 'store_utilhi_outlier'), store_utilhi_outlier)

##################
# END PART 2
##################

##################
# PART 3: B=1400
##################
deptList_IPRP = ['Dakar', 'Keur Massar', 'Pikine', 'Louga', 'Linguere', 'Kaolack', 'Guinguineo',
                 'Nioro du Rip', 'Kaffrine', 'Birkilane', 'Malem Hoddar', 'Bambey', 'Mbacke',
                 'Fatick', 'Foundiougne', 'Gossas']
allocList_IPRP = [19, 21, 7, 7, 11, 38, 9, 18, 8, 8, 8, 10, 7, 11, 10, 9]
n_IPRP = GetAllocVecFromLists(deptNames, deptList_IPRP, allocList_IPRP)

store_baseloss_1400 = np.zeros((len(numbatcharr), numreps))
store_utilhi_1400 = np.zeros((3, len(numbatcharr), numreps))  # Efficient, then imp samp (no out remove), then imp samp (0.5% out remove)
store_utillo_1400 = np.zeros((3, len(numbatcharr), numreps))  # Efficient, then imp samp (no out remove), then imp samp (0.5% out remove)

for currbatchind in range(len(numbatcharr)):
    numbatch = numbatcharr[currbatchind]
    for rep in range(numreps):
        # Retrieve previously generated MCMC draws, which are in batches of 5000; each batch takes up about 3MB
        RetrieveMCMCBatches(lgdict, numbatch,
                            os.path.join('operationalizedsamplingplans', 'numpy_objects', 'draws'),
                            maxbatchnum=50, rand=True, randseed=rep+currbatchind+24)
        # Set up utility estimation parameter dictionary with desired truth and data draws
        SetupUtilEstParamDict(lgdict, paramdict, numbatch*5000, 500, randseed=rep+currbatchind+106)
        util.print_param_checks(paramdict)
        store_baseloss_1400[currbatchind, rep] = paramdict['baseloss']

        # EFFICIENT ESTIMATE
        _, util_IPRP_CI = getUtilityEstimate(n_IPRP, lgdict, paramdict, zlevel=0.95)
        store_utillo_1400[0, currbatchind, rep] = util_IPRP_CI[0]
        store_utilhi_1400[0, currbatchind, rep] = util_IPRP_CI[1]

        # IMP SAMP ESTIMATE
        _, util_IPRP_CI = sampf.getImportanceUtilityEstimate(n_IPRP, lgdict, paramdict,
                                                             numimportdraws=numbatch * 5000, impweightoutlierprop=0.00,
                                                             zlevel=0.95)
        store_utillo_1400[1, currbatchind, rep] = util_IPRP_CI[0]
        store_utilhi_1400[1, currbatchind, rep] = util_IPRP_CI[1]

        # IMP SAMP ESTIMATE, WITH 0.5% HIGHEST IMPORTANCE WEIGHTS REMOVED
        _, util_IPRP_CI = sampf.getImportanceUtilityEstimate(n_IPRP, lgdict, paramdict,
                                                             numimportdraws=numbatch * 5000,
                                                             impweightoutlierprop=0.005,
                                                             zlevel=0.95)
        store_utillo_1400[2, currbatchind, rep] = util_IPRP_CI[0]
        store_utilhi_1400[2, currbatchind, rep] = util_IPRP_CI[1]

        # Plot
        fig, ax = plt.subplots()
        fig.set_figheight(7)
        fig.set_figwidth(16)

        x = np.arange(numreps * len(numbatcharr) * 3)
        flat_utillo = store_utillo_1400.flatten()
        flat_utilhi = store_utilhi_1400.flatten()
        CIavg = (flat_utillo + flat_utilhi) / 2

        ax.errorbar(x, CIavg, yerr=[CIavg - flat_utillo, flat_utilhi - CIavg],
                    fmt='o', ecolor='g', capthick=4)
        ax.set_title('95% CI for IP-RP solution under different parameters and estimation methods\nB=1400')
        # ax.grid('on')

        xticklist = ['' for j in range(numreps * len(numbatcharr) * 3)]
        for currbatchnameind, currbatchname in enumerate(numbatcharr):
            xticklist[currbatchnameind * 10] = str(currbatchname) + ' batch\nEffic'
            xticklist[currbatchnameind * 10 + 50] = str(currbatchname) + ' batch\nImpSamp'
            xticklist[currbatchnameind * 10 + 100] = str(currbatchname) + ' batch\nImpSamp_Out'
        plt.xticks(x, xticklist)
        plt.xlabel('Method and parameterization')
        plt.ylabel('Utility estimate')

        plt.ylim([0, 10])
        ax.tick_params(axis='x', labelsize=8)
        label_X = ax.xaxis.get_label()
        label_Y = ax.yaxis.get_label()
        label_X.set_style('italic')
        label_X.set_size(12)
        label_Y.set_style('italic')
        label_Y.set_size(12)

        plt.show()

        # Store arrays
        np.save(os.path.join('studies', 'utiloracle_13FEB25', 'store_baseloss_1400'), store_baseloss_1400)
        np.save(os.path.join('studies', 'utiloracle_13FEB25', 'store_utillo_1400'), store_utillo_1400)
        np.save(os.path.join('studies', 'utiloracle_13FEB25', 'store_utilhi_1400'), store_utilhi_1400)

##################
# END PART 3
##################

##################
# PART 4: Plot B=700,1400 together
##################
utilhi_700 = np.load(os.path.join('studies', 'utiloracle_13FEB25', 'store_utilhi.npy'))
utilhi_700_extm = np.load(os.path.join('studies', 'utiloracle_13FEB25', 'store_utilhi_outlier.npy'))
utilhi_1400 = np.load(os.path.join('studies', 'utiloracle_13FEB25', 'store_utilhi_1400.npy'))
utillo_700 = np.load(os.path.join('studies', 'utiloracle_13FEB25', 'store_utillo.npy'))
utillo_700_extm = np.load(os.path.join('studies', 'utiloracle_13FEB25', 'store_utillo_outlier.npy'))
utillo_1400 = np.load(os.path.join('studies', 'utiloracle_13FEB25', 'store_utillo_1400.npy'))


fig, ax = plt.subplots()
fig.set_figheight(7)
fig.set_figwidth(27)

x = np.arange(numreps * len(numbatcharr) * 6)
flat_utillo = np.concatenate((utillo_700.flatten(), utillo_700_extm.flatten(), utillo_1400.flatten()))
flat_utilhi = np.concatenate((utilhi_700.flatten(), utilhi_700_extm.flatten(), utilhi_1400.flatten()))
CIavg = (flat_utillo + flat_utilhi) / 2

ax.errorbar(x, CIavg, yerr=[CIavg - flat_utillo, flat_utilhi - CIavg],
            fmt='o', ecolor='g', capthick=4)
ax.set_title('95% CI for IP-RP solution under different parameters, estimation methods, and budgets')

xticklist = ['' for j in range(numreps * len(numbatcharr) * 6)]
for currbatchnameind, currbatchname in enumerate(numbatcharr):
    xticklist[currbatchnameind * 10] = str(currbatchname) + ' batch\nEffic,B=700'
    xticklist[currbatchnameind * 10 + 50] = str(currbatchname) + ' batch\nImpSamp,B=700'
    xticklist[currbatchnameind * 10 + 100] = str(currbatchname) + ' batch\nImpSamp_Extm,,B=700'
    xticklist[currbatchnameind * 10 + 150] = str(currbatchname) + ' batch\nEffic,B=1400'
    xticklist[currbatchnameind * 10 + 200] = str(currbatchname) + ' batch\nImpSamp,B=1400'
    xticklist[currbatchnameind * 10 + 250] = str(currbatchname) + ' batch\nImpSamp_Extm,,B=1400'
plt.xticks(x, xticklist)
plt.xlabel('Method and parameterization')
plt.ylabel('Utility estimate')

plt.ylim([0, 10])
ax.tick_params(axis='x', labelsize=6)
label_X = ax.xaxis.get_label()
label_Y = ax.yaxis.get_label()
label_X.set_style('italic')
label_X.set_size(12)
label_Y.set_style('italic')
label_Y.set_size(12)

plt.show()

##################
# END PART 4
##################

##################
# PART 5: Impact of extrema elimination; use 25k MCMC draws
##################
# B=700 first
deptList_IPRP = ['Dakar', 'Keur Massar', 'Pikine', 'Diourbel', 'Bambey', 'Mbacke', 'Fatick', 'Foundiougne', 'Gossas']
allocList_IPRP = [42, 21, 7, 9, 10, 7, 11, 10, 9]
n_IPRP = GetAllocVecFromLists(deptNames, deptList_IPRP, allocList_IPRP)
testnum = int(np.sum(n_IPRP))
des = n_IPRP / testnum

deltvec = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
numbatch, numreps = 5, 10  # Use 50k draws for this part of the study
numdatadrawsforimportance = 1000
numimportdraws = numbatch * 5000
zlevel = 0.95

store_util_extrm_700 = np.zeros((numreps, len(deltvec)))
store_utilhi_extrm_700 = np.zeros((numreps, len(deltvec)))
store_utillo_extrm_700 = np.zeros((numreps, len(deltvec)))

for rep in range(numreps):
    # Retrieve 25k previously generated MCMC draws, which are in batches of 5000
    RetrieveMCMCBatches(lgdict, numbatch,
                        os.path.join('operationalizedsamplingplans', 'numpy_objects', 'draws'),
                        maxbatchnum=50, rand=True, randseed=rep + 10 + 24)
    # Set up utility estimation parameter dictionary with desired truth and data draws
    SetupUtilEstParamDict(lgdict, paramdict, numbatch * 5000, 500, randseed=rep + numbatch + 106)
    util.print_param_checks(paramdict)
    # ADJUSTED LOSS-LIST GENERATION HERE
    # Get utility intervals under each choice of delta
    roundalg = 'lo'

    (numTN, numSN), Q, s, r = lgdict['N'].shape, lgdict['Q'], lgdict['diagSens'], lgdict['diagSpec']
    # Identify an 'average' data set that will help establish the important region for importance sampling
    importancedatadrawinds = np.random.choice(np.arange(paramdict['datadraws'].shape[0]),
                                              size=numdatadrawsforimportance,  # Oversample if needed
                                              replace=paramdict['datadraws'].shape[0] < numdatadrawsforimportance)
    importancedatadraws = paramdict['datadraws'][importancedatadrawinds]
    zMatData = util.zProbTrVec(numSN, importancedatadraws, sens=s, spec=r)  # Probs. using data draws
    NMat = np.moveaxis(np.array([np.random.multinomial(n_IPRP[tnInd], Q[tnInd], size=numdatadrawsforimportance)
                                 for tnInd in range(numTN)]), 1, 0).astype(int)
    YMat = np.random.binomial(NMat, zMatData)
    # Get average rounded data set from these few draws
    NMatAvg, YMatAvg = np.round(np.average(NMat, axis=0)).astype(int), np.round(np.average(YMat, axis=0)).astype(int)
    # Add these data to a new data dictionary and generate a new set of MCMC draws
    impdict = lgdict.copy()
    impdict['N'], impdict['Y'] = lgdict['N'] + NMatAvg, lgdict['Y'] + YMatAvg
    # Generate a new MCMC importance set
    impdict['numPostSamples'] = numimportdraws
    impdict = methods.GeneratePostSamples(impdict, maxTime=5000)
    # Get simulated data likelihoods - don't normalize
    numdatadraws = paramdict['datadraws'].shape[0]
    zMatTruth = util.zProbTrVec(numSN, impdict['postSamples'], sens=s,
                                spec=r)  # Matrix of SFP probabilities, as a function of SFP rate draws
    zMatData = util.zProbTrVec(numSN, paramdict['datadraws'], sens=s, spec=r)  # Probs. using data draws
    NMat = np.moveaxis(np.array([np.random.multinomial(n_IPRP[tnInd], Q[tnInd], size=numdatadraws)
                                 for tnInd in range(numTN)]), 1, 0).astype(int)
    YMat = np.random.binomial(NMat, zMatData)
    tempW = np.zeros(shape=(numimportdraws, numdatadraws))
    for snInd in range(
            numSN):  # Loop through each SN and TN combination; DON'T vectorize as resulting matrix can be too big
        for tnInd in range(numTN):
            if n_IPRP[tnInd] > 0 and Q[tnInd, snInd] > 0:  # Save processing by only looking at feasible traces
                # Get zProbs corresponding to current trace
                bigZtemp = np.transpose(
                    np.reshape(np.tile(zMatTruth[:, tnInd, snInd], numdatadraws), (numdatadraws, numimportdraws)))
                bigNtemp = np.reshape(np.tile(NMat[:, tnInd, snInd], numimportdraws), (numimportdraws, numdatadraws))
                bigYtemp = np.reshape(np.tile(YMat[:, tnInd, snInd], numimportdraws), (numimportdraws, numdatadraws))
                combNYtemp = np.reshape(np.tile(spsp.comb(NMat[:, tnInd, snInd], YMat[:, tnInd, snInd]), numimportdraws),
                                        (numimportdraws, numdatadraws))
                tempW += (bigYtemp * np.log(bigZtemp)) + ((bigNtemp - bigYtemp) * np.log(1 - bigZtemp)) + np.log(
                    combNYtemp)
    Wimport = np.exp(tempW)

    # Get risk matrix
    Rimport = lf.risk_check_array(impdict['postSamples'], paramdict['riskdict'])
    # Get critical ratio
    q = paramdict['scoredict']['underestweight'] / (1 + paramdict['scoredict']['underestweight'])

    # Get likelihood weights WRT original data set: p(gamma|d_0)
    zMatImport = util.zProbTrVec(numSN, impdict['postSamples'], sens=s,
                                 spec=r)  # Matrix of SFP probabilities along each trace
    NMatPrior, YMatPrior = lgdict['N'], lgdict['Y']
    Vimport = np.zeros(shape=numimportdraws)
    for snInd in range(
            numSN):  # Loop through each SN and TN combination; DON'T vectorize as resulting matrix can be too big
        for tnInd in range(numTN):
            if NMatPrior[tnInd, snInd] > 0:
                bigZtemp = np.transpose(
                    np.reshape(np.tile(zMatImport[:, tnInd, snInd], 1), (1, numimportdraws)))
                bigNtemp = np.reshape(np.tile(NMatPrior[tnInd, snInd], numimportdraws), (numimportdraws, 1))
                bigYtemp = np.reshape(np.tile(YMatPrior[tnInd, snInd], numimportdraws), (numimportdraws, 1))
                combNYtemp = np.reshape(np.tile(spsp.comb(NMatPrior[tnInd, snInd], YMatPrior[tnInd, snInd]),
                                                numimportdraws), (numimportdraws, 1))
                Vimport += np.squeeze(
                    (bigYtemp * np.log(bigZtemp)) + ((bigNtemp - bigYtemp) * np.log(1 - bigZtemp)) + np.log(
                        combNYtemp))
    Vimport = np.exp(Vimport)

    # Get likelihood weights WRT average data set: p(gamma|d_0, d_imp)
    NMatPrior, YMatPrior = impdict['N'].copy(), impdict['Y'].copy()
    Uimport = np.zeros(shape=numimportdraws)
    for snInd in range(
            numSN):  # Loop through each SN and TN combination; DON'T vectorize as resulting matrix can be too big
        for tnInd in range(numTN):
            if NMatPrior[tnInd, snInd] > 0:
                bigZtemp = np.transpose(
                    np.reshape(np.tile(zMatImport[:, tnInd, snInd], 1), (1, numimportdraws)))
                bigNtemp = np.reshape(np.tile(NMatPrior[tnInd, snInd], numimportdraws), (numimportdraws, 1))
                bigYtemp = np.reshape(np.tile(YMatPrior[tnInd, snInd], numimportdraws), (numimportdraws, 1))
                combNYtemp = np.reshape(np.tile(spsp.comb(NMatPrior[tnInd, snInd], YMatPrior[tnInd, snInd]),
                                                numimportdraws), (numimportdraws, 1))
                Uimport += np.squeeze(
                    (bigYtemp * np.log(bigZtemp)) + ((bigNtemp - bigYtemp) * np.log(1 - bigZtemp)) + np.log(
                        combNYtemp))
    Uimport = np.exp(Uimport)

    # Importance likelihood ratio for importance draws
    VoverU = (Vimport / Uimport)

    # Compile list of optima
    for deltind, delt in enumerate(deltvec):
        print('delta: '+str(delt))
        minslist = []
        for j in range(Wimport.shape[1]):
            tempwtarray = Wimport[:, j] * VoverU * numimportdraws / np.sum(Wimport[:, j] * VoverU)
            # Remove inds for top delt of weights
            tempremoveinds = np.where(tempwtarray > np.quantile(tempwtarray, 1 - delt)) # KEY CHANGE HERE
            tempwtarray = np.delete(tempwtarray, tempremoveinds)
            tempwtarray = tempwtarray / np.sum(tempwtarray)
            tempimportancedraws = np.delete(impdict['postSamples'], tempremoveinds, axis=0)
            tempRimport = np.delete(Rimport, tempremoveinds, axis=0)
            est = sampf.bayesest_critratio(tempimportancedraws, tempwtarray, q)
            minslist.append(sampf.cand_obj_val(est, tempimportancedraws, tempwtarray, paramdict, tempRimport))

        # RETURN minslist
        currloss_avg, currloss_CI = sampf.process_loss_list(minslist, zlevel=zlevel)
        store_util_extrm_700[rep, deltind] = paramdict['baseloss'] - currloss_avg
        store_utillo_extrm_700[rep, deltind] = paramdict['baseloss'] - currloss_CI[1]
        store_utilhi_extrm_700[rep, deltind] = paramdict['baseloss'] - currloss_CI[0]

        # Plot
        fig, ax = plt.subplots()
        fig.set_figheight(7)
        fig.set_figwidth(11)

        for k in range(numreps):
            ax.errorbar(deltvec, store_util_extrm_700[k],
                        yerr=[store_util_extrm_700[k] - store_utillo_extrm_700[k],
                              store_utilhi_extrm_700[k] - store_util_extrm_700[k]],
                        fmt='o-', ecolor='g', capthick=4)
        ax.set_title('95% CI for IP-RP solution, B=700, under different $\delta$\n10 replications')
        plt.xlabel('$\delta$')
        plt.ylabel('Utility estimate')

        plt.ylim([0, 6])
        ax.tick_params(axis='x', labelsize=8)
        label_X = ax.xaxis.get_label()
        label_Y = ax.yaxis.get_label()
        label_X.set_style('italic')
        label_X.set_size(12)
        label_Y.set_style('italic')
        label_Y.set_size(12)

        plt.show()
        # Store arrays
        np.save(os.path.join('studies', 'utiloracle_13FEB25', 'extrema_study', 'store_util_extrm_700'),
                store_util_extrm_700)
        np.save(os.path.join('studies', 'utiloracle_13FEB25', 'extrema_study', 'store_utillo_extrm_700'),
                store_utillo_extrm_700)
        np.save(os.path.join('studies', 'utiloracle_13FEB25', 'extrema_study', 'store_utilhi_extrm_700'),
                store_utilhi_extrm_700)

##################
# END PART 5
##################


##################
# PART 6: Impact of extrema elimination; use 25k MCMC draws
##################
# B=1400
deptList_IPRP = ['Dakar', 'Keur Massar', 'Pikine', 'Louga', 'Linguere', 'Kaolack', 'Guinguineo',
                 'Nioro du Rip', 'Kaffrine', 'Birkilane', 'Malem Hoddar', 'Bambey', 'Mbacke',
                 'Fatick', 'Foundiougne', 'Gossas']
allocList_IPRP = [19, 21, 7, 7, 11, 38, 9, 18, 8, 8, 8, 10, 7, 11, 10, 9]
n_IPRP = GetAllocVecFromLists(deptNames, deptList_IPRP, allocList_IPRP)
testnum = int(np.sum(n_IPRP))
des = n_IPRP / testnum

deltvec = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
numbatch, numreps = 5, 10  # Use 50k draws for this part of the study
numdatadrawsforimportance = 1000
numimportdraws = numbatch * 5000
zlevel = 0.95

store_util_extrm_1400 = np.zeros((numreps, len(deltvec)))
store_utilhi_extrm_1400 = np.zeros((numreps, len(deltvec)))
store_utillo_extrm_1400 = np.zeros((numreps, len(deltvec)))

for rep in range(numreps):
    # Retrieve 25k previously generated MCMC draws, which are in batches of 5000
    RetrieveMCMCBatches(lgdict, numbatch,
                        os.path.join('operationalizedsamplingplans', 'numpy_objects', 'draws'),
                        maxbatchnum=50, rand=True, randseed=rep + 10 + 24)
    # Set up utility estimation parameter dictionary with desired truth and data draws
    SetupUtilEstParamDict(lgdict, paramdict, numbatch * 5000, 500, randseed=rep + numbatch + 106)
    util.print_param_checks(paramdict)
    # ADJUSTED LOSS-LIST GENERATION HERE
    # Get utility intervals under each choice of delta
    roundalg = 'lo'

    (numTN, numSN), Q, s, r = lgdict['N'].shape, lgdict['Q'], lgdict['diagSens'], lgdict['diagSpec']
    # Identify an 'average' data set that will help establish the important region for importance sampling
    importancedatadrawinds = np.random.choice(np.arange(paramdict['datadraws'].shape[0]),
                                              size=numdatadrawsforimportance,  # Oversample if needed
                                              replace=paramdict['datadraws'].shape[0] < numdatadrawsforimportance)
    importancedatadraws = paramdict['datadraws'][importancedatadrawinds]
    zMatData = util.zProbTrVec(numSN, importancedatadraws, sens=s, spec=r)  # Probs. using data draws
    NMat = np.moveaxis(np.array([np.random.multinomial(n_IPRP[tnInd], Q[tnInd], size=numdatadrawsforimportance)
                                 for tnInd in range(numTN)]), 1, 0).astype(int)
    YMat = np.random.binomial(NMat, zMatData)
    # Get average rounded data set from these few draws
    NMatAvg, YMatAvg = np.round(np.average(NMat, axis=0)).astype(int), np.round(np.average(YMat, axis=0)).astype(int)
    # Add these data to a new data dictionary and generate a new set of MCMC draws
    impdict = lgdict.copy()
    impdict['N'], impdict['Y'] = lgdict['N'] + NMatAvg, lgdict['Y'] + YMatAvg
    # Generate a new MCMC importance set
    impdict['numPostSamples'] = numimportdraws
    impdict = methods.GeneratePostSamples(impdict, maxTime=5000)
    # Get simulated data likelihoods - don't normalize
    numdatadraws = paramdict['datadraws'].shape[0]
    zMatTruth = util.zProbTrVec(numSN, impdict['postSamples'], sens=s,
                                spec=r)  # Matrix of SFP probabilities, as a function of SFP rate draws
    zMatData = util.zProbTrVec(numSN, paramdict['datadraws'], sens=s, spec=r)  # Probs. using data draws
    NMat = np.moveaxis(np.array([np.random.multinomial(n_IPRP[tnInd], Q[tnInd], size=numdatadraws)
                                 for tnInd in range(numTN)]), 1, 0).astype(int)
    YMat = np.random.binomial(NMat, zMatData)
    tempW = np.zeros(shape=(numimportdraws, numdatadraws))
    for snInd in range(
            numSN):  # Loop through each SN and TN combination; DON'T vectorize as resulting matrix can be too big
        for tnInd in range(numTN):
            if n_IPRP[tnInd] > 0 and Q[tnInd, snInd] > 0:  # Save processing by only looking at feasible traces
                # Get zProbs corresponding to current trace
                bigZtemp = np.transpose(
                    np.reshape(np.tile(zMatTruth[:, tnInd, snInd], numdatadraws), (numdatadraws, numimportdraws)))
                bigNtemp = np.reshape(np.tile(NMat[:, tnInd, snInd], numimportdraws), (numimportdraws, numdatadraws))
                bigYtemp = np.reshape(np.tile(YMat[:, tnInd, snInd], numimportdraws), (numimportdraws, numdatadraws))
                combNYtemp = np.reshape(np.tile(spsp.comb(NMat[:, tnInd, snInd], YMat[:, tnInd, snInd]), numimportdraws),
                                        (numimportdraws, numdatadraws))
                tempW += (bigYtemp * np.log(bigZtemp)) + ((bigNtemp - bigYtemp) * np.log(1 - bigZtemp)) + np.log(
                    combNYtemp)
    Wimport = np.exp(tempW)

    # Get risk matrix
    Rimport = lf.risk_check_array(impdict['postSamples'], paramdict['riskdict'])
    # Get critical ratio
    q = paramdict['scoredict']['underestweight'] / (1 + paramdict['scoredict']['underestweight'])

    # Get likelihood weights WRT original data set: p(gamma|d_0)
    zMatImport = util.zProbTrVec(numSN, impdict['postSamples'], sens=s,
                                 spec=r)  # Matrix of SFP probabilities along each trace
    NMatPrior, YMatPrior = lgdict['N'], lgdict['Y']
    Vimport = np.zeros(shape=numimportdraws)
    for snInd in range(
            numSN):  # Loop through each SN and TN combination; DON'T vectorize as resulting matrix can be too big
        for tnInd in range(numTN):
            if NMatPrior[tnInd, snInd] > 0:
                bigZtemp = np.transpose(
                    np.reshape(np.tile(zMatImport[:, tnInd, snInd], 1), (1, numimportdraws)))
                bigNtemp = np.reshape(np.tile(NMatPrior[tnInd, snInd], numimportdraws), (numimportdraws, 1))
                bigYtemp = np.reshape(np.tile(YMatPrior[tnInd, snInd], numimportdraws), (numimportdraws, 1))
                combNYtemp = np.reshape(np.tile(spsp.comb(NMatPrior[tnInd, snInd], YMatPrior[tnInd, snInd]),
                                                numimportdraws), (numimportdraws, 1))
                Vimport += np.squeeze(
                    (bigYtemp * np.log(bigZtemp)) + ((bigNtemp - bigYtemp) * np.log(1 - bigZtemp)) + np.log(
                        combNYtemp))
    Vimport = np.exp(Vimport)

    # Get likelihood weights WRT average data set: p(gamma|d_0, d_imp)
    NMatPrior, YMatPrior = impdict['N'].copy(), impdict['Y'].copy()
    Uimport = np.zeros(shape=numimportdraws)
    for snInd in range(
            numSN):  # Loop through each SN and TN combination; DON'T vectorize as resulting matrix can be too big
        for tnInd in range(numTN):
            if NMatPrior[tnInd, snInd] > 0:
                bigZtemp = np.transpose(
                    np.reshape(np.tile(zMatImport[:, tnInd, snInd], 1), (1, numimportdraws)))
                bigNtemp = np.reshape(np.tile(NMatPrior[tnInd, snInd], numimportdraws), (numimportdraws, 1))
                bigYtemp = np.reshape(np.tile(YMatPrior[tnInd, snInd], numimportdraws), (numimportdraws, 1))
                combNYtemp = np.reshape(np.tile(spsp.comb(NMatPrior[tnInd, snInd], YMatPrior[tnInd, snInd]),
                                                numimportdraws), (numimportdraws, 1))
                Uimport += np.squeeze(
                    (bigYtemp * np.log(bigZtemp)) + ((bigNtemp - bigYtemp) * np.log(1 - bigZtemp)) + np.log(
                        combNYtemp))
    Uimport = np.exp(Uimport)

    # Importance likelihood ratio for importance draws
    VoverU = (Vimport / Uimport)

    # Compile list of optima
    for deltind, delt in enumerate(deltvec):
        print('delta: '+str(delt))
        minslist = []
        for j in range(Wimport.shape[1]):
            tempwtarray = Wimport[:, j] * VoverU * numimportdraws / np.sum(Wimport[:, j] * VoverU)
            # Remove inds for top delt of weights
            tempremoveinds = np.where(tempwtarray > np.quantile(tempwtarray, 1 - delt)) # KEY CHANGE HERE
            tempwtarray = np.delete(tempwtarray, tempremoveinds)
            tempwtarray = tempwtarray / np.sum(tempwtarray)
            tempimportancedraws = np.delete(impdict['postSamples'], tempremoveinds, axis=0)
            tempRimport = np.delete(Rimport, tempremoveinds, axis=0)
            est = sampf.bayesest_critratio(tempimportancedraws, tempwtarray, q)
            minslist.append(sampf.cand_obj_val(est, tempimportancedraws, tempwtarray, paramdict, tempRimport))

        # RETURN minslist
        currloss_avg, currloss_CI = sampf.process_loss_list(minslist, zlevel=zlevel)
        store_util_extrm_1400[rep, deltind] = paramdict['baseloss'] - currloss_avg
        store_utillo_extrm_1400[rep, deltind] = paramdict['baseloss'] - currloss_CI[1]
        store_utilhi_extrm_1400[rep, deltind] = paramdict['baseloss'] - currloss_CI[0]

        # Plot
        fig, ax = plt.subplots()
        fig.set_figheight(7)
        fig.set_figwidth(11)

        for k in range(numreps):
            ax.errorbar(deltvec, store_util_extrm_1400[k],
                        yerr=[store_util_extrm_1400[k] - store_utillo_extrm_1400[k],
                              store_utilhi_extrm_1400[k] - store_util_extrm_1400[k]],
                        fmt='o-', ecolor='g', capthick=4)
        ax.set_title('95% CI for IP-RP solution, B=1400, under different $\delta$\n10 replications')
        plt.xlabel('$\delta$')
        plt.ylabel('Utility estimate')

        plt.ylim([0, 8])
        ax.tick_params(axis='x', labelsize=8)
        label_X = ax.xaxis.get_label()
        label_Y = ax.yaxis.get_label()
        label_X.set_style('italic')
        label_X.set_size(12)
        label_Y.set_style('italic')
        label_Y.set_size(12)

        plt.show()
        # Store arrays
        np.save(os.path.join('studies', 'utiloracle_13FEB25', 'extrema_study', 'store_util_extrm_1400'),
                store_util_extrm_1400)
        np.save(os.path.join('studies', 'utiloracle_13FEB25', 'extrema_study', 'store_utillo_extrm_1400'),
                store_utillo_extrm_1400)
        np.save(os.path.join('studies', 'utiloracle_13FEB25', 'extrema_study', 'store_utilhi_extrm_1400'),
                store_utilhi_extrm_1400)

##################
# END PART 6
##################