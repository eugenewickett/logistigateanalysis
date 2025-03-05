"""
Attempting to understand odd behavior for utility estimates in the 'existing' setting
We will estimate utility under different efficient and imp sampling parameters
Part 1: Greedy allocation
Part 2: Uniform allocation
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

plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.family"] = "serif"

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

# Some summaries
TNtesttotals = np.sum(Nfam, axis=1)
TNsfptotals = np.sum(Yfam, axis=1)
TNrates = np.divide(TNsfptotals,TNtesttotals)
print('Tests at each test node:')
print(TNtesttotals)
print('Positives at each test node:')
print(TNsfptotals)
print('Positive rates at each test node:')
print(TNrates)

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
# Path for previously generated MCMC draws
mcmcfiledest = os.path.join(os.getcwd(), 'utilitypaper', 'casestudy_python_files', 'numpy_obj', 'mcmc_draws',
                            'existing')

# Set up utility estimation parameters
paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN))

numreps = 8  # Replications of different sets of initial MCMC draws
numbatcharr = [1, 2, 5, 10, 15]
testindarr = [1, 5, 10, 20, 30, 40]
store_baseloss = np.zeros((len(numbatcharr), numreps))
store_utilhi = np.zeros((2, len(numbatcharr), len(testindarr), numreps))  # Efficient, then imp samp
store_utillo = np.zeros((2, len(numbatcharr), len(testindarr), numreps))  # Efficient, then imp samp

leadfilestr = os.path.join(os.getcwd(), 'studies', 'utilpapercasestudy_5MAR25')

alloc = np.load(os.path.join(leadfilestr, 'exist_alloc.npy'))

def RetrieveMCMCBatches(lgdict, numbatches, filedest_leadstring, maxbatchnum=50, rand=False, randseed=1):
    """Adds previously generated MCMC draws to lgdict, using the file destination marked by filedest_leadstring"""
    if rand==False:
        tempobj = np.load(filedest_leadstring + '0.npy')  # Initialize
    elif rand==True:  # Generate a unique list of integers
        np.random.seed(randseed)
        groupindlist = np.random.choice(np.arange(0, maxbatchnum), size=numbatches, replace=False)
        tempobj = np.load(filedest_leadstring + str(groupindlist[0]) + '.npy')  # Initialize
    for drawgroupind in range(2, numbatches+1):
        if rand==False:
            newobj = np.load(filedest_leadstring + str(drawgroupind) + '.npy')
            tempobj = np.concatenate((tempobj, newobj))
        elif rand==True:
            newobj = np.load(filedest_leadstring + str(groupindlist[drawgroupind-1]) + '.npy')
            tempobj = np.concatenate((tempobj, newobj))
    lgdict.update({'postSamples': tempobj, 'numPostSamples': tempobj.shape[0]})
    return

def SetupUtilEstParamDict(lgdict, paramdict, numtruthdraws, numdatadraws, randseed):
    """Sets up parameter dictionary with desired truth and data draws"""
    np.random.seed(randseed)
    truthdraws, datadraws = util.distribute_truthdata_draws(lgdict['postSamples'], numtruthdraws, numdatadraws)
    paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
    paramdict.update({'baseloss': sampf.baseloss(paramdict['truthdraws'], paramdict)})
    return

def getUtilityEstimate(n, lgdict, paramdict, zlevel=0.95):
    """
    Return a utility estimate average and confidence interval for allocation array n,
    using efficient estimation (NOT importance sampling)
    """
    testnum = int(np.sum(n))
    des = n/testnum
    currlosslist = sampf.sampling_plan_loss_list(des, testnum, lgdict, paramdict)
    currloss_avg, currloss_CI = sampf.process_loss_list(currlosslist, zlevel=zlevel)
    return paramdict['baseloss'] - currloss_avg, (paramdict['baseloss']-currloss_CI[1], paramdict['baseloss']-currloss_CI[0])


for currbatchind in range(len(numbatcharr)):  # Number of MCMC batches of 5k to use
    numbatch = numbatcharr[currbatchind]
    for rep in range(numreps):
        # Retrieve previously generated MCMC draws, which are in batches of 5000; each batch takes up about 3MB
        RetrieveMCMCBatches(csdict_fam, numbatch, os.path.join(mcmcfiledest,'draws'),
                            maxbatchnum=40, rand=True, randseed=rep+currbatchind+232)
        # Set up utility estimation parameter dictionary with desired truth and data draws
        SetupUtilEstParamDict(csdict_fam, paramdict, numbatch*5000, 500, randseed=rep+currbatchind+132)
        util.print_param_checks(paramdict)
        store_baseloss[currbatchind, rep] = paramdict['baseloss']

        for testind in range(len(testindarr)):  # Iterate through each number of tests
            # EFFICIENT ESTIMATE
            _, util_CI = getUtilityEstimate(alloc[:, testindarr[testind]], csdict_fam, paramdict, zlevel=0.95)
            store_utillo[0, currbatchind, testind, rep] = util_CI[0]
            store_utilhi[0, currbatchind, testind, rep] = util_CI[1]

            # IMP SAMP ESTIMATE
            _, util_CI = sampf.getImportanceUtilityEstimate(alloc[:, testindarr[testind]], csdict_fam, paramdict,
                                                                 numimportdraws=numbatch * 5000, preservevar=False,
                                                                 extremadelta=0.01, zlevel=0.95)
            store_utillo[1, currbatchind, testind, rep] = util_CI[0]
            store_utilhi[1, currbatchind, testind, rep] = util_CI[1]

            # Plot
            fig, ax = plt.subplots()
            fig.set_figheight(7)
            fig.set_figwidth(15)

            x = np.arange(numreps*len(numbatcharr)*len(testindarr)*2)
            flat_utillo = store_utillo.flatten()
            flat_utilhi = store_utilhi.flatten()
            CIavg = (flat_utillo + flat_utilhi) / 2
            #ax.plot(x, CIavg, marker='o', linestyle='None')
            ax.errorbar(x, CIavg, yerr=[CIavg-flat_utillo, flat_utilhi-CIavg],
                        fmt='o', ecolor='g', capthick=4)
            ax.set_title('95% CI for greedy allocation under different parameters, estimation methods, and numbers of tests')
            #ax.grid('on')

            xticklist = ['' for j in range(numreps*len(numbatcharr)*len(testindarr)*2)]
            # for currbatchnameind, currbatchname in enumerate(numbatcharr):
            #     for currtestnum, testnum in enumerate(testindarr):
            #         xticklist[(currbatchnameind + currtestnum) * numreps] = str(currbatchname) + ' batch\n' +\
            #                                                               str(testnum) + ' test\nEffic'
            #         xticklist[(currbatchnameind + currtestnum) * numreps + numreps*len(numbatcharr)*len(testindarr)] =\
            #             str(currbatchname) + ' batch\n' + str(testnum) + ' test\nImpSamp'
            plt.xticks(x, xticklist)  # trick to get textual X labels instead of numerical
            plt.xlabel('Method and parameterization')
            plt.ylabel('Utility estimate')
            plt.ylim([0, np.max(flat_utilhi)*1.05])
            ax.tick_params(axis='x', labelsize=8)
            label_X = ax.xaxis.get_label()
            label_Y = ax.yaxis.get_label()
            label_X.set_style('italic')
            label_X.set_size(12)
            label_Y.set_style('italic')
            label_Y.set_size(12)
            plt.show()

            # Plot baseloss also
            # fig, ax = plt.subplots()
            # fig.set_figheight(7)
            # fig.set_figwidth(12)
            #
            # x = np.arange(numreps * len(numbatcharr))
            # flat_baseloss = store_baseloss.flatten()
            # ax.plot(x, flat_baseloss, 'o', color='orange')
            # ax.set_title('Base loss for IP-RP solution under different parameters and estimation methods\nB=700')
            #
            # xticklist = ['' for j in range(numreps * len(numbatcharr))]
            # for currbatchnameind, currbatchname in enumerate(numbatcharr):
            #     xticklist[currbatchnameind * 10] = str(currbatchname) + ' batch\nEffic'
            # plt.xticks(x, xticklist)  # little trick to get textual X labels instead of numerical
            # plt.xlabel('Method and parameterization')
            # plt.ylabel('Loss estimate')
            #
            # plt.ylim([0,13])
            # ax.tick_params(axis='x', labelsize=8)
            # label_X = ax.xaxis.get_label()
            # label_Y = ax.yaxis.get_label()
            # label_X.set_style('italic')
            # label_X.set_size(12)
            # label_Y.set_style('italic')
            # label_Y.set_size(12)
            #
            # plt.show()

            # Save numpy objects
            np.save(os.path.join('studies', 'utilpapercasestudy_5MAR25', 'store_baseloss'), store_baseloss)
            np.save(os.path.join('studies', 'utilpapercasestudy_5MAR25', 'store_utillo'), store_utillo)
            np.save(os.path.join('studies', 'utilpapercasestudy_5MAR25', 'store_utilhi'), store_utilhi)
