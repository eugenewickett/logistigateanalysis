"""
This study aims to inspect differences in oracle performance for the efficient and  importance sampling methods of
utility estimation, for different numbers of truth/data draws, using the data of the case study for the
orienteering paper.

PARTS
1: FOR FIXED ALLOCATION (IP-RP SOLUTION), RUN 10 ITERATIONS EACH OF EFFICIENT METHOD AND IMP SAMP METHOD
    AND STORE RESULTING CI FOR UTILITY ESTIMATE, USING DIFFERENT INITIAL MCMC DRAWS EACH TIME;
    USE {1, 2, 5, 10, 20} BATCHES
2: REPEAT FOR DIFFERENT FIXED ALLOCATION (IP-RP SOLN DIVIDED BY 2)
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
    for rep in range(10):
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
# PART 2: B=700, REMOVING OUTLIERS FROM IMP SAMP
##################
# There is a lot of variance/inflation with the imp samp method; does removing largely weighted draws improve things?
store_utilhi_outlier = np.zeros((len(numbatcharr), numreps))  # Efficient, then imp samp
store_utillo_outlier = np.zeros((len(numbatcharr), numreps))  # Efficient, then imp samp

for currbatchind in range(len(numbatcharr)):
    numbatch = numbatcharr[currbatchind]
    for rep in range(10):
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
        # todo: FIX FROM HERE
        flat_utillo = store_utillo.flatten()
        flat_utilhi = store_utilhi.flatten()
        CIavg = (flat_utillo + flat_utilhi) / 2
        # ax.plot(x, CIavg, marker='o', linestyle='None')
        ax.errorbar(x, CIavg, yerr=[CIavg - flat_utillo, flat_utilhi - CIavg],
                    fmt='o', ecolor='g', capthick=4)
        ax.set_title('95% CI for IP-RP solution under different parameters and estimation methods\nB=700')
        # ax.grid('on')

        xticklist = ['' for j in range(numreps * len(numbatcharr) * 2)]
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

