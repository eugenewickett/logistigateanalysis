"""
Attempting to understand odd behavior for utility estimates in the 'existing' setting
We will estimate utility under different efficient and imp sampling parameters
Part 1: Greedy allocation
Part 2: Chart differences between efficient and imp sampling
Part 3: Verify that adding more draws to efficient method solves bias issue
Part 4: Run example of callibration step
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

def RetrieveMCMCBatches(lgdict, numbatches, filedest_leadstring, maxbatchnum=80, rand=False, randseed=1):
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

stop = False
rep = 0
while not stop:
    store_baseloss = np.load(os.path.join(leadfilestr, 'store_baseloss.npy'))
    store_utilhi = np.load(os.path.join(leadfilestr, 'store_utilhi.npy'))
    store_utillo = np.load(os.path.join(leadfilestr, 'store_utillo.npy'))
    if store_utilhi[-1, -1, -1, -1] > 0:
        stop = True
    else:
        # Establish where we are at
        currbatchind = np.min(np.where(store_baseloss == 0)[0])
        rep = np.where(store_baseloss == 0)[1][0]
        print('Batch: '+str(currbatchind))
        print('Replication: ' + str(rep))

        numbatch = numbatcharr[currbatchind]
        # Retrieve previously generated MCMC draws, which are in batches of 5000; each batch takes up about 3MB
        RetrieveMCMCBatches(csdict_fam, numbatch, os.path.join(mcmcfiledest,'draws'),
                            maxbatchnum=40, rand=True, randseed=rep+currbatchind+232)
        # Set up utility estimation parameter dictionary with desired truth and data draws
        SetupUtilEstParamDict(csdict_fam, paramdict, numbatch*5000, 500, randseed=rep+currbatchind+132)
        util.print_param_checks(paramdict)
        store_baseloss[currbatchind, rep] = paramdict['baseloss']

        for testind in range(len(testindarr)):  # Iterate through each number of tests
            # EFFICIENT ESTIMATE
            _, util_CI = getUtilityEstimate(alloc[:, testindarr[testind]]*10, csdict_fam, paramdict, zlevel=0.95)
            store_utillo[0, currbatchind, testind, rep] = util_CI[0]
            store_utilhi[0, currbatchind, testind, rep] = util_CI[1]

            # IMP SAMP ESTIMATE
            _, util_CI = sampf.getImportanceUtilityEstimate(alloc[:, testindarr[testind]]*10, csdict_fam, paramdict,
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
            currcol = 'red'
            for xiter in range(int(len(x)/numreps)):
                ax.errorbar(x[(xiter*numreps):((xiter*numreps)+numreps)],
                            CIavg[(xiter*numreps):((xiter*numreps)+numreps)],
                            yerr=[(CIavg-flat_utillo)[(xiter*numreps):((xiter*numreps)+numreps)],
                                  (flat_utilhi-CIavg)[(xiter*numreps):((xiter*numreps)+numreps)]],
                            color=currcol, markersize=4, fmt='o', ecolor='black',
                            capthick=3)
                if currcol == 'red':
                    currcol = 'blue'
                else:
                    currcol = 'red'
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
            np.save(os.path.join(leadfilestr, 'store_baseloss'), store_baseloss)
            np.save(os.path.join(leadfilestr, 'store_utillo'), store_utillo)
            np.save(os.path.join(leadfilestr, 'store_utilhi'), store_utilhi)

########
# PART 2: Examine differences in efficient and imp sampling approaches
########
fig, ax = plt.subplots()
fig.set_figheight(7)
fig.set_figwidth(15)

x = np.arange(numreps*len(numbatcharr)*len(testindarr)*2)
# Rearrange storage matrices to group together by tests-->MCMC draws-->method
origaxislst = [0, 2]
newaxislst = [2, 0]
store_utilhi_rearr = np.moveaxis(store_utilhi, origaxislst, newaxislst)
store_utillo_rearr = np.moveaxis(store_utillo, origaxislst, newaxislst)
flat_utillo = store_utillo_rearr.flatten()
flat_utilhi = store_utilhi_rearr.flatten()
CIavg = (flat_utillo + flat_utilhi) / 2
currcol = 'red'
for xiter in range(int(len(x)/numreps)):
    ax.errorbar(x[(xiter*numreps):((xiter*numreps)+numreps)],
                CIavg[(xiter*numreps):((xiter*numreps)+numreps)],
                yerr=[(CIavg-flat_utillo)[(xiter*numreps):((xiter*numreps)+numreps)],
                      (flat_utilhi-CIavg)[(xiter*numreps):((xiter*numreps)+numreps)]],
                color=currcol, markersize=4, fmt='o', ecolor='black',
                capthick=3)
    if currcol == 'red':
        currcol = 'blue'
    else:
        currcol = 'red'
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


########
# PART 3: Verify that additional draws solves issue of estimate bias
########
# Run 200, 300, and 400 tests for this number of batches
numbatcharr_new = [20, 30, 40, 50, 60]
numbatcharr_tot = numbatcharr + numbatcharr_new

# Generate expanded storages of utilities
store_baseloss = np.load(os.path.join(leadfilestr, 'store_baseloss.npy'))
store_utilhi = np.load(os.path.join(leadfilestr, 'store_utilhi.npy'))
store_utillo = np.load(os.path.join(leadfilestr, 'store_utillo.npy'))
# # Just add zeros to include new number of batches
# store_baseloss_new = np.zeros(tuple(sum(x) for x in zip(store_baseloss.shape, (len(numbatcharr_new), 0))))
# store_utilhi_new = np.zeros(tuple(sum(x) for x in zip(store_utilhi.shape, (0, len(numbatcharr_new), 0, 0))))
# store_utillo_new = np.zeros(tuple(sum(x) for x in zip(store_utillo.shape, (0, len(numbatcharr_new), 0, 0))))
# store_baseloss_new[:len(numbatcharr_new), :] = store_baseloss.copy()
# store_utilhi_new[:, :len(numbatcharr_new), :, :] = store_utilhi.copy()
# store_utillo_new[:, :len(numbatcharr_new), :, :] = store_utillo.copy()

def sampling_plan_loss_list_parallel(design, numtests, priordatadict, paramdict):
    """
    Produces a list of sampling plan losses for a test budget under a given data set and specified parameters,
    using the efficient estimation algorithm, but processing weights one data simulation at a time; this
    parallelization enables processing of large numbers of truth draws.
    design: sampling probability vector along all test nodes/traces
    numtests: test budget
    priordatadict: logistigate data dictionary capturing known data
    paramdict: parameter dictionary containing a loss matrix, truth and data MCMC draws, and an optional method for
        rounding the design to an integer allocation
    """
    if 'roundalg' in paramdict:  # Set default rounding algorithm for plan
        roundalg = paramdict['roundalg'].copy()
    else:
        roundalg = 'lo'
    # Initialize samples to be drawn from traces, per the design, using a rounding algorithm
    sampMat = util.generate_sampling_array(design, numtests, roundalg)
    # Get risk matrix
    R = lf.risk_check_array(paramdict['truthdraws'], paramdict['riskdict'])
    # Get critical ratio
    q = paramdict['scoredict']['underestweight'] / (1 + paramdict['scoredict']['underestweight'])
    # Compile list of optima
    minslist = []
    for j in range(paramdict['datadraws'].shape[0]):
        if np.mod(j,10) == 0:
            print('On data sim: ' + str(j))
        # Get weights matrix
        W = sampf.build_weights_matrix(paramdict['truthdraws'],
                            np.reshape(paramdict['datadraws'][j], (1, paramdict['datadraws'][j].shape[0])),
                                       sampMat, priordatadict)
        est = sampf.bayesest_critratio(paramdict['truthdraws'], W[:, 0], q)
        minslist.append(sampf.cand_obj_val(est, paramdict['truthdraws'], W[:, 0], paramdict, R))
    return minslist


def getUtilityEstimate_parallel(n, lgdict, paramdict, zlevel=0.95):
    """
    Return a utility estimate average and confidence interval for allocation array n,
    using efficient estimation (NOT importance sampling)
    """
    testnum = int(np.sum(n))
    des = n/testnum
    currlosslist = sampling_plan_loss_list_parallel(des, testnum, lgdict, paramdict)
    currloss_avg, currloss_CI = sampf.process_loss_list(currlosslist, zlevel=zlevel)
    return paramdict['baseloss'] - currloss_avg, (paramdict['baseloss']-currloss_CI[1], paramdict['baseloss']-currloss_CI[0])



stop = False
rep = 0
while not stop:
    store_baseloss_new = np.load(os.path.join(leadfilestr, 'store_baseloss_new.npy'))
    store_utilhi_new = np.load(os.path.join(leadfilestr, 'store_utilhi_new.npy'))
    store_utillo_new = np.load(os.path.join(leadfilestr, 'store_utillo_new.npy'))
    if store_utilhi_new[0, -1, -1, -1] > 0:  # We're not running imp sampling for these
        stop = True
    else:
        # Establish where we're at
        currbatchind = np.min(np.where(store_baseloss_new == 0)[0])
        rep = np.where(store_baseloss_new == 0)[1][0]
        print('Batch: '+str(currbatchind))
        print('Replication: ' + str(rep))

        numbatch = numbatcharr_tot[currbatchind]
        # Retrieve previously generated MCMC draws, which are in batches of 5000; each batch takes up about 3MB
        RetrieveMCMCBatches(csdict_fam, numbatch, os.path.join(mcmcfiledest, 'draws'),
                            maxbatchnum=80, rand=True, randseed=rep+currbatchind+239)
        # Set up utility estimation parameter dictionary with desired truth and data draws
        SetupUtilEstParamDict(csdict_fam, paramdict, numbatch*5000, 500, randseed=rep+currbatchind+132)
        util.print_param_checks(paramdict)
        store_baseloss_new[currbatchind, rep] = paramdict['baseloss']

        for testind in range(3, len(testindarr)):  # Skip 10, 50, and 100 tests
            # EFFICIENT ESTIMATE FOR 100k+ MCMC DRAWS
            _, util_CI = getUtilityEstimate_parallel(alloc[:, testindarr[testind]]*10, csdict_fam, paramdict,
                                                     zlevel=0.95)
            store_utillo_new[0, currbatchind, testind, rep] = util_CI[0]
            store_utilhi_new[0, currbatchind, testind, rep] = util_CI[1]

            # Plot
            fig, ax = plt.subplots()
            fig.set_figheight(7)
            fig.set_figwidth(15)

            x = np.arange(numreps*len(numbatcharr_tot)*len(testindarr)*2)
            store_utilhi_rearr = np.moveaxis(store_utilhi_new, origaxislst, newaxislst)
            store_utillo_rearr = np.moveaxis(store_utillo_new, origaxislst, newaxislst)
            flat_utillo = store_utillo_rearr.flatten()
            flat_utilhi = store_utilhi_rearr.flatten()
            CIavg = (flat_utillo + flat_utilhi) / 2
            currcol = 'red'
            for xiter in range(int(len(x)/numreps)):
                ax.errorbar(x[(xiter*numreps):((xiter*numreps)+numreps)],
                            CIavg[(xiter*numreps):((xiter*numreps)+numreps)],
                            yerr=[(CIavg-flat_utillo)[(xiter*numreps):((xiter*numreps)+numreps)],
                                  (flat_utilhi-CIavg)[(xiter*numreps):((xiter*numreps)+numreps)]],
                            color=currcol, markersize=3, fmt='o', ecolor='black',
                            capthick=2)
                if currcol == 'red':
                    currcol = 'blue'
                else:
                    currcol = 'red'
            ax.set_title('95% CI for greedy allocation under different parameters, estimation methods, and numbers of tests')
            #ax.grid('on')

            xticklist = ['' for j in range(numreps*len(numbatcharr_tot)*len(testindarr)*2)]
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
            np.save(os.path.join(leadfilestr, 'store_baseloss_new'), store_baseloss_new)
            np.save(os.path.join(leadfilestr, 'store_utillo_new'), store_utillo_new)
            np.save(os.path.join(leadfilestr, 'store_utilhi_new'), store_utilhi_new)


########
# PART 4: Complete a run-through of the proposed callibration procedure
########
# PROPOSED CALLIBRATION OF SUFFICIENT MCMC DRAWS:
#   Choose Nmax for desired analysis; larger Nmax means more needed compute time
#   Generate enough (?) batches of 5k MCMC draws to be used later
#   Use 100 (?) data draws for what follows
#   Run 10 (?) imp samp evals for uniform allocation of Nmax tests, using 25k (?) draws of different batches
#       Identify U_imp, the average across these imp samp evals
#   Try different amounts of \Gamma_0 for uniform allocation of Nmax tests, using 10 (?) efficient evals
#       Stop when U_eff (avg across effic evals) is less than U_imp
