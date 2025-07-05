"""
Utility estimates for the 'existing' setting
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
csdict_existing = util.initDataDict(Nfam, Yfam) # Initialize necessary logistigate keys
csdict_existing['TNnames'] = ['MOD_39', 'MOD_17', 'MODHIGH_95', 'MODHIGH_26']
csdict_existing['SNnames'] = ['MNFR ' + str(i + 1) for i in range(numSN)]

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
csdict_existing['prior'] = priorObj

# Set up MCMC
csdict_existing['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
# Path for previously generated MCMC draws
mcmcfiledest = os.path.join(os.getcwd(), 'utilitypaper', 'casestudy_python_files', 'numpy_obj', 'mcmc_draws',
                            'existing')
'''
numdraws = 5000  # Blocks of 5k draws
csdict_existing['numPostSamples'] = numdraws
for rep in range(80, 100):
    np.random.seed(rep+1000)
    csdict_existing = methods.GeneratePostSamples(csdict_existing)
    np.save(os.path.join(mcmcfiledest, 'draws'+str(rep)+'.npy'), csdict_existing['postSamples'])
'''

# Print inference from initial data
# util.plotPostSamples(csdict_existing, 'int90')

# Loss specification
paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN))

# Set limits of data collection and intervals for calculation
testmax, testint = 400, 10
testarr = np.arange(testint, testmax + testint, testint)

# Generate calibration plots to decide how many truth draws to use
# sampf.makecalibrationplot(csdict_existing, paramdict, testmax, mcmcfiledest,
#                     batchlist=[2, 5, 10, 20, 30, 40, 50, 60, 70, 80], nrep=10, numdatadraws=300)

# Set MCMC draws to use in fast algorithm
numtruthdraws, numdatadraws = 300000, 1000
util.RetrieveMCMCBatches(csdict_existing, int(numtruthdraws / 5000), os.path.join(mcmcfiledest, 'draws'), maxbatchnum=100,
                         rand=True, randseed=122)

# Get random subsets for truth and data draws
np.random.seed(444)
truthdraws, datadraws = util.distribute_truthdata_draws(csdict_existing['postSamples'], numtruthdraws, numdatadraws)
paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
# Get base loss
paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)

util.print_param_checks(paramdict)  # Check of used parameters

###############
# GREEDY HEURISTIC
###############
alloc, util_avg, util_hi, util_lo = sampf.get_greedy_allocation(csdict_existing, testmax, testint, paramdict,
                                                                estmethod='parallel',
                                                                printupdate=True, plotupdate=True,
                                                                plottitlestr='Existing Setting')

storestr = os.path.join(os.getcwd(), 'utilitypaper', 'casestudy_python_files', 'existing')

np.save(os.path.join(storestr, 'exist_alloc'), alloc)
np.save(os.path.join(storestr, 'util_avg_greedy'), util_avg)
np.save(os.path.join(storestr, 'util_hi_greedy'), util_hi)
np.save(os.path.join(storestr, 'util_lo_greedy'), util_lo)

#####################
#####################
#####################

def unif_design_mat(numTN, testmax, testint=1):
    """
    Generates a design matrix that allocates tests uniformly across all test nodes, for a max number of tests (testmax),
    a testing interval (testint), and a number of test nodes (numTN)
    """
    numcols = int(testmax / testint)
    testarr = np.arange(testint, testmax + testint, testint)
    des = np.zeros((numTN, int(testmax / testint)))
    for j in range(numcols):
        des[:, j] = np.ones(numTN) * np.floor(testarr[j] / numTN)
        numtoadd = testarr[j] - np.sum(des[:, j])
        if numtoadd > 0:
            for k in range(int(numtoadd)):
                des[k, j] += 1

    return des / testarr

def rudi_design_mat(numTN, testmax, testint=1):
    """
    Generates a design matrix that allocates tests uniformly across all test nodes, for a max number of tests (testmax),
    a testing interval (testint), and a number of test nodes (numTN)
    """
    numcols = int(testmax / testint)
    testarr = np.arange(testint, testmax + testint, testint)
    des = np.zeros((numTN, int(testmax / testint)))
    for j in range(numcols):
        des[:, j] = np.floor(np.divide(np.sum(Nfam, axis=1), np.sum(Nfam)) * testarr[j])
        numtoadd = testarr[j] - np.sum(des[:, j])
        if numtoadd > 0:
            for k in range(int(numtoadd)):
                des[k, j] += 1

    return des / testarr

unif_mat = unif_design_mat(numTN, testmax, testint)
rudi_mat = rudi_design_mat(numTN, testmax, testint)

stop = False
while not stop:
    alloc = np.load(os.path.join(storestr, 'exist_alloc.npy'))
    util_avg_greedy, util_hi_greedy, util_lo_greedy = np.load(
        os.path.join(storestr, 'util_avg_greedy_eff.npy')), \
        np.load(os.path.join(storestr, 'util_hi_greedy_eff.npy')), \
        np.load(os.path.join(storestr, 'util_lo_greedy_eff.npy'))
    util_avg_unif, util_hi_unif, util_lo_unif = np.load(
        os.path.join(storestr, 'util_avg_unif_eff.npy')), \
        np.load(os.path.join(storestr, 'util_hi_unif_eff.npy')), \
        np.load(os.path.join(storestr, 'util_lo_unif_eff.npy'))
    util_avg_rudi, util_hi_rudi, util_lo_rudi = np.load(
        os.path.join(storestr, 'util_avg_rudi_eff.npy')), \
        np.load(os.path.join(storestr, 'util_hi_rudi_eff.npy')), \
        np.load(os.path.join(storestr, 'util_lo_rudi_eff.npy'))

    if util_avg_greedy[-1] > 0:
        stop = True
    else:  # Do a set of utility estimates at the next zero
        # Index skips first column, which should be zeros for all
        currind = np.where(util_avg_greedy[1:] == 0)[0][0]
        print("Current testnum: " + str((currind+1)*testint))
        currbudget = testarr[currind]

        curralloc = alloc[:, currind + 1]
        des_greedy = curralloc / np.sum(curralloc)
        des_unif = unif_mat[:, currind]
        des_rudi = rudi_mat[:, currind]

        # Greedy
        currlosslist = sampf.sampling_plan_loss_list_parallel(des_greedy, currbudget, csdict_existing, paramdict)

        avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
        util_avg_greedy[currind + 1] = paramdict['baseloss'] - avg_loss
        util_lo_greedy[currind + 1] = paramdict['baseloss'] - avg_loss_CI[1]
        util_hi_greedy[currind + 1] = paramdict['baseloss'] - avg_loss_CI[0]
        print(des_greedy)
        print('Utility at ' + str(currbudget) + ' tests, Greedy: ' + str(util_avg_greedy[currind + 1]))

        # Uniform
        currlosslist = sampf.sampling_plan_loss_list_parallel(des_unif, currbudget, csdict_existing, paramdict)

        avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
        util_avg_unif[currind + 1] = paramdict['baseloss'] - avg_loss
        util_lo_unif[currind + 1] = paramdict['baseloss'] - avg_loss_CI[1]
        util_hi_unif[currind + 1] = paramdict['baseloss'] - avg_loss_CI[0]
        print(des_unif)
        print('Utility at ' + str(currbudget) + ' tests, Uniform: ' + str(util_avg_unif[currind + 1]))

        # Rudimentary
        currlosslist = sampf.sampling_plan_loss_list_parallel(des_rudi, currbudget, csdict_existing, paramdict)

        avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
        util_avg_rudi[currind + 1] = paramdict['baseloss'] - avg_loss
        util_lo_rudi[currind + 1] = paramdict['baseloss'] - avg_loss_CI[1]
        util_hi_rudi[currind + 1] = paramdict['baseloss'] - avg_loss_CI[0]
        print(des_rudi)
        print('Utility at ' + str(currbudget) + ' tests, Rudimentary: ' + str(util_avg_rudi[currind + 1]))

        np.save(os.path.join(storestr, 'util_avg_greedy_eff'), util_avg_greedy)
        np.save(os.path.join(storestr, 'util_hi_greedy_eff'), util_hi_greedy)
        np.save(os.path.join(storestr, 'util_lo_greedy_eff'), util_lo_greedy)
        np.save(os.path.join(storestr, 'util_avg_unif_eff'), util_avg_unif)
        np.save(os.path.join(storestr, 'util_hi_unif_eff'), util_hi_unif)
        np.save(os.path.join(storestr, 'util_lo_unif_eff'), util_lo_unif)
        np.save(os.path.join(storestr, 'util_avg_rudi_eff'), util_avg_rudi)
        np.save(os.path.join(storestr, 'util_hi_rudi_eff'), util_hi_rudi)
        np.save(os.path.join(storestr, 'util_lo_rudi_eff'), util_lo_rudi)

    # Plot
    #numint = util_avg.shape[0]
    util.plot_marg_util_CI(np.vstack((util_avg_greedy, util_avg_unif, util_avg_rudi)),
                           np.vstack((util_hi_greedy, util_hi_unif, util_hi_rudi)),
                           np.vstack((util_lo_greedy, util_lo_unif, util_lo_rudi)),
                           testmax, testint, titlestr="Greedy, Uniform, and Rudimentary",
                           labels=['Greedy', 'Uniform', 'Rudimentary'])




targind = 10 # what do we want to gauge budget savings against (our "peg")?
targval = util_avg[targind]

# Uniform
kInd = next(x for x, val in enumerate(util_avg_arr[0].tolist()) if val > targval)
unif_saved = round((targval - util_avg_arr[0][kInd - 1]) / (util_avg_arr[0][kInd] - util_avg_arr[0][kInd - 1]) *\
                      testint) + (kInd - 1) * testint - targind*testint
print(unif_saved)  # 33
# Rudimentary
kInd = next(x for x, val in enumerate(util_avg_arr[1].tolist()) if val > targval)
rudi_saved = round((targval - util_avg_arr[1][kInd - 1]) / (util_avg_arr[1][kInd] - util_avg_arr[1][kInd - 1]) *\
                      testint) + (kInd - 1) * testint - targind*testint
print(rudi_saved)  # 57
