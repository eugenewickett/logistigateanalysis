"""
Utility estimates for the 'all-provinces' setting
5-JUL UPDATE: For the 300k/1k fast estimation, we use the previously identified allocation rather
                than generate the allocation all over again AND identify the utility estimates, for
                time considerations. This shouldn't drastically affect the allocation produced by
                the greedy heuristic.
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
csdict_allprov = util.initDataDict(Nexpl, Yexpl) # Initialize necessary logistigate keys
csdict_allprov['TNnames'] = ['MOD_39', 'MOD_17', 'MODHIGH_95', 'MODHIGH_26',
                              'MODHIGH_EXPL_1', 'MOD_EXPL_1', 'MODHIGH_EXPL_2', 'MOD_EXPL_2']
csdict_allprov['SNnames'] = ['MNFR ' + str(i + 1) for i in range(numSN)]

# Use observed data to form Q for tested nodes; use bootstrap data for untested nodes
numBoot = 44  # Average across each TN in original data set
SNprobs = np.sum(csdict_allprov['N'], axis=0) / np.sum(csdict_allprov['N'])
np.random.seed(33)
Qvecs = np.random.multinomial(numBoot, SNprobs, size=4) / numBoot
csdict_allprov['Q'] = np.vstack((csdict_allprov['Q'][:4], Qvecs))

# Build prior
SNpriorMean = np.repeat(sps.logit(0.1), numSN)
# Establish test node priors according to assessment by regulators
TNpriorMean = sps.logit(np.array([0.1, 0.1, 0.15, 0.15, 0.15, 0.1, 0.15, 0.1]))
TNvar, SNvar = 2., 4.  # Variances for use with prior; supply nodes are wide due to uncertainty
csdict_allprov['prior'] = prior_normal_assort(np.concatenate((SNpriorMean, TNpriorMean)),
                                              np.diag(np.concatenate((np.repeat(SNvar, numSN),
                                           np.repeat(TNvar, numTN)))))

# Set up MCMC
csdict_allprov['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
# Path for previously generated MCMC draws
mcmcfiledest = os.path.join(os.getcwd(), 'utilitypaper', 'casestudy_python_files',
                            'numpy_obj', 'mcmc_draws', 'allprovinces')

# Loss specification
paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN))

# Set limits of data collection and intervals for calculation
testmax, testint = 400, 10
testarr = np.arange(testint, testmax + testint, testint)

# sampf.makecalibrationplot(csdict_allprov, paramdict, testmax, mcmcfiledest,
#                           batchlist=[2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#                           nrep=8, numdatadraws=200)






# Set MCMC draws to use in fast algorithm, based on results of calibration plots
numtruthdraws, numdatadraws = 300000, 1000
util.RetrieveMCMCBatches(csdict_allprov, int(numtruthdraws/5000),
                         os.path.join(mcmcfiledest, 'draws'), maxbatchnum=100,
                         rand=True, randseed=1222)

# Get random subsets for truth and data draws
np.random.seed(444)
truthdraws, datadraws = util.distribute_truthdata_draws(csdict_allprov['postSamples'], numtruthdraws, numdatadraws)
paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
# Get base loss
paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)

util.print_param_checks(paramdict)  # Check of used parameters

storestr = os.path.join(os.getcwd(), 'utilitypaper', 'casestudy_python_files', 'allprovinces')


###############
# GREEDY HEURISTIC
###############
alloc, util_avg, util_hi, util_lo = sampf.get_greedy_allocation(csdict_allprov, testmax, testint, paramdict,
                                                                estmethod='parallel',
                                                                printupdate=True, plotupdate=True,
                                                                plottitlestr='All-provinces Setting')

# np.save(os.path.join(storestr, 'exist_alloc'), alloc)
# np.save(os.path.join(storestr, 'util_avg_greedy'), util_avg)
# np.save(os.path.join(storestr, 'util_hi_greedy'), util_hi)
# np.save(os.path.join(storestr, 'util_lo_greedy'), util_lo)

alloc = np.load(os.path.join(storestr, 'allprov_alloc.npy'))

# 5-JUL-25: TRANSCRIBED FROM OLD ALLOCATION PLOT
bestallocnodes = [4, 6, 7, 5, 1,
                  7, 7, 5, 0, 5,
                  7, 4, 4, 6, 6,
                  6, 3, 2, 2, 6,
                  1, 1, 1, 1, 3,
                  1, 4, 1, 4, 7,
                  7, 1, 1, 3, 6,
                  6, 6, 3, 1, 3]

'''
(0.13763228686300044, 0.14425541844816392)
(0.26890504514003544, 0.27762338324634106)
(0.3919619127557854, 0.4025577137367198)
(0.48955070330733985, 0.5019357825545061)
(0.5741617853005092, 0.5888570926643819)
(0.6363024343700694, 0.6510312563696932)
'''

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
    des = np.zeros((numTN, numcols))
    for j in range(numcols):
        des[:, j] = np.concatenate((np.floor(np.divide(np.sum(Nexpl[:4], axis=1), np.sum(Nexpl[:4])) * testarr[j]),
                                   np.zeros((4))))
        numtoadd = testarr[j] - np.sum(des[:, j])
        if numtoadd > 0:
            for k in range(int(numtoadd)):
                des[k, j] += 1

    return des / testarr

unif_mat = unif_design_mat(numTN, testmax, testint)
rudi_mat = rudi_design_mat(numTN, testmax, testint)

# Initialize our utility estimate storage arrays
# util_avg_greedy, util_hi_greedy, util_lo_greedy = np.zeros(alloc.shape[1]), np.zeros(alloc.shape[1]), np.zeros(alloc.shape[1])
# util_avg_unif, util_hi_unif, util_lo_unif = np.zeros(alloc.shape[1]), np.zeros(alloc.shape[1]), np.zeros(alloc.shape[1])
# util_avg_rudi, util_hi_rudi, util_lo_rudi = np.zeros(alloc.shape[1]), np.zeros(alloc.shape[1]), np.zeros(alloc.shape[1])
# np.save(os.path.join(storestr, 'util_avg_greedy_eff'), util_avg_greedy)
# np.save(os.path.join(storestr, 'util_hi_greedy_eff'), util_hi_greedy)
# np.save(os.path.join(storestr, 'util_lo_greedy_eff'), util_lo_greedy)
# np.save(os.path.join(storestr, 'util_avg_unif_eff'), util_avg_unif)
# np.save(os.path.join(storestr, 'util_hi_unif_eff'), util_hi_unif)
# np.save(os.path.join(storestr, 'util_lo_unif_eff'), util_lo_unif)
# np.save(os.path.join(storestr, 'util_avg_rudi_eff'), util_avg_rudi)
# np.save(os.path.join(storestr, 'util_hi_rudi_eff'), util_hi_rudi)
# np.save(os.path.join(storestr, 'util_lo_rudi_eff'), util_lo_rudi)


stop = False
while not stop:
    alloc = np.load(os.path.join(storestr, 'allprov_alloc.npy'))
    util_avg_greedy, util_hi_greedy, util_lo_greedy = np.load(os.path.join(storestr, 'util_avg_greedy_eff.npy')), \
        np.load(os.path.join(storestr, 'util_hi_greedy_eff.npy')), \
        np.load(os.path.join(storestr, 'util_lo_greedy_eff.npy'))
    util_avg_unif, util_hi_unif, util_lo_unif = np.load(os.path.join(storestr, 'util_avg_unif_eff.npy')), \
        np.load(os.path.join(storestr, 'util_hi_unif_eff.npy')), \
        np.load(os.path.join(storestr, 'util_lo_unif_eff.npy'))
    util_avg_rudi, util_hi_rudi, util_lo_rudi = np.load(os.path.join(storestr, 'util_avg_rudi_eff.npy')), \
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
        currlosslist = sampf.sampling_plan_loss_list_parallel(des_greedy, currbudget, csdict_allprov, paramdict)

        avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
        util_avg_greedy[currind + 1] = paramdict['baseloss'] - avg_loss
        util_lo_greedy[currind + 1] = paramdict['baseloss'] - avg_loss_CI[1]
        util_hi_greedy[currind + 1] = paramdict['baseloss'] - avg_loss_CI[0]
        print(des_greedy)
        print('Utility at ' + str(currbudget) + ' tests, Greedy: ' + str(util_avg_greedy[currind + 1]))

        # Uniform
        currlosslist = sampf.sampling_plan_loss_list_parallel(des_unif, currbudget, csdict_allprov, paramdict)

        avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
        util_avg_unif[currind + 1] = paramdict['baseloss'] - avg_loss
        util_lo_unif[currind + 1] = paramdict['baseloss'] - avg_loss_CI[1]
        util_hi_unif[currind + 1] = paramdict['baseloss'] - avg_loss_CI[0]
        print(des_unif)
        print('Utility at ' + str(currbudget) + ' tests, Uniform: ' + str(util_avg_unif[currind + 1]))

        # Rudimentary
        currlosslist = sampf.sampling_plan_loss_list_parallel(des_rudi, currbudget, csdict_allprov, paramdict)

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





#
# stop = False
# while not stop:
#     # Read in the current allocation
#     alloc = np.load(os.path.join(storestr, 'allprov_alloc.npy'))
#     util_avg = np.load(os.path.join(storestr, 'allprov_util_avg.npy'))
#     util_hi = np.load(os.path.join(storestr, 'allprov_util_hi.npy'))
#     util_lo = np.load(os.path.join(storestr, 'allprov_util_lo.npy'))
#     # Stop if the last allocation column is empty
#     if np.sum(alloc[:, -1]) > 0:
#         stop = True
#     else:
#         testnumind = np.argmax(np.sum(alloc, axis=0))
#         bestalloc = alloc[:, testnumind]
#         nextTN = -1 # Initialize next best node
#         currbestloss_avg, currbestloss_CI = -1, (-1, -1) # Initialize next best utility
#         for currTN in range(numTN):  # Loop through each test node and identify best direction via lowest avg loss
#             curralloc = bestalloc.copy()
#             curralloc[currTN] += 1  # Increment 1 at current test node
#             testnum = np.sum(curralloc) * testint
#             currdes = curralloc / np.sum(curralloc)  # Make a proportion design
#             currlosslist = sampf.sampling_plan_loss_list_importance(currdes, testnum, csdict_allprov, paramdict,
#                                                                     numimportdraws=60000,
#                                                                     numdatadrawsforimportance=5000,
#                                                                     impweightoutlierprop=0.005)
#             currloss_avg, currloss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
#             print('TN ' + str(currTN) + ' loss avg.: ' + str(currloss_avg))
#             if nextTN == -1 or currloss_avg < currbestloss_avg:  # Update with better loss
#                 nextTN = currTN
#                 currbestloss_avg = currloss_avg
#                 currbestloss_CI = currloss_CI
#         # Store best results
#         alloc[:, testnumind + 1] = bestalloc.copy()
#         alloc[nextTN, testnumind + 1] += 1
#         util_avg[testnumind + 1] = paramdict['baseloss'] - currbestloss_avg
#         util_hi[testnumind + 1] = paramdict['baseloss'] - currbestloss_CI[0]
#         util_lo[testnumind + 1] = paramdict['baseloss'] - currbestloss_CI[1]
#         # Save as numpy objects
#         np.save(os.path.join('..', 'allprovinces', 'allprov_alloc'), alloc)
#         np.save(os.path.join('..', 'allprovinces', 'allprov_util_avg'), util_avg)
#         np.save(os.path.join('..', 'allprovinces', 'allprov_util_hi'), util_hi)
#         np.save(os.path.join('..', 'allprovinces', 'allprov_util_lo'), util_lo)
#         print('TN ' + str(nextTN) + ' added, with utility CI of (' + str(util_lo[testnumind + 1]) + ', ' +
#               str(util_hi[testnumind + 1]) + ') for ' + str(testnum) + ' tests')
#         numint = util_avg.shape[0]
#         util.plot_marg_util_CI(util_avg.reshape(1, numint), util_hi.reshape(1, numint), util_lo.reshape(1, numint),
#                                testmax, testint, titlestr='')
#         util.plot_plan(alloc, np.arange(0, testmax + 1, testint), testint, titlestr='')
'''
END RESTARTABLE GREEDY ALLOCATION
'''






############################
############################
############################
############################
############################
############################



'''
5-JUL
DISREGARD EVERYTHING BELOW HERE, WHICH IS COPIED FROM BEFORE AND OLD
'''

alloc, util_avg, util_hi, util_lo = sampf.get_greedy_allocation(csdict_allprov, testmax, testint, paramdict,
                                                                numimpdraws=60000, numdatadrawsforimp=5000,
                                                                impwtoutlierprop=0.005,
                                                                printupdate=True, plotupdate=True,
                                                                plottitlestr='Exploratory Setting')


'''
25-SEP
[4, 6, 5, 1, 7, 7, 7, 5, 0, 5
 7, 4, 4, 6, 6, 6, 3, 2, 2, 6,
 ]

[(0.17292519603501644, 0.1765185733995387), (0.33400992816135666, 0.33827880128424326), (0.4428727535222894, 0.4483219432850918), (0.548368922617267, 0.5541303009419316), (0.6536904869826925, 0.6604919430997489)
 (0.7079470548279652, 0.7145842198883123), (0.7581483734176326, 0.7641238996313393), (0.8159737055554528, 0.8218362612259336), (0.8514697071128663, 0.8574239383134583), (0.8949279030608981, 0.9006034053837357),
 (0.908111654832028, 0.9138951324961444), (0.9310866366195478, 0.9367561268209803), (0.9599606087782844, 0.9656282382182848), (1.002549891406391, 1.0085190419883179), (1.0305208309164757, 1.036071034150901),
 (1.0556741075265523, 1.0613291280826194), (1.1023959156977847, 1.1075447642129106), (1.1070408289239597, 1.1124140441756425), (1.1180054941582118, 1.1233125539163304), (1.1373542226310782, 1.1430296172115222),
 
]

BEFORE 
[4, 6, 7, 5, 1, 1, 6, 7, 1,
 4, 5, 1, 5, 7, 4, 1, 7, 6,
 1, 0, 1, 4, 7, 0, 0, 4, 4,
 6, 1, 6, 7, 0, 0, 5, 7, 1,
 5, 
 ]
[(0.1367016267244403, 0.14085927615070082), (0.2730860941620148, 0.27919132150000125), (0.39063679190203304, 0.3982630162769749),
(0.4937371657371008, 0.5025369766810908), (0.5797701769098991, 0.5904628982206077), (0.6435399657617231, 0.6548095014569735) ,
(0.7035984264333188, 0.7150359885825144), (0.7630429842266984, 0.7754849725613293), (0.8138675058709159, 0.8273313904591584),
(0.8632173189124774, 0.8762833278807545), (0.9208709816851588, 0.9370829653622712), (0.9660348368455236, 0.9820397171801347),
(1.0099717070230145, 1.029485837638509), (1.0516800353628573, 1.0713606412379586), (1.0984662758942068, 1.1187483209268092),
(1.1371888083928368, 1.1597196574236759), (1.1931301498967222, 1.2184930683113113), (1.2229806017952112, 1.2495404144602023),
(1.2710414438207631, 1.2985547384347593), (1.3108090715736074, 1.3402217626129431), (1.3633556811172887, 1.3963611313232511),
(1.4016798615482982, 1.436398186751882), (1.4393042753068062, 1.4748892145149646), (1.4890790768002564, 1.5274438599175648),
(1.519932235306156, 1.5575358483752189), (1.5620460345901899, 1.6015239750366703), (1.6027552373713259, 1.6459329607449864),
(1.6404322716581319, 1.683345946308904), (1.684220795854632, 1.7283035451317759), (1.7275930982842347, 1.7731880043199109),
(1.7698988602504915, 1.8174798600229873), (1.8081982769540352, 1.8580261964908487), (1.830259412914371, 1.880282391623148),
(1.8871483209269466, 1.9379916219662428), (1.9192092860751633, 1.9700547554418255), (1.9662427840482677, 2.017996925568169),
(2.011433704156665, 2.0649481345628224), 
]
'''

np.save(os.path.join('../../casestudyoutputs', 'exploratory', 'expl_alloc'), alloc)
np.save(os.path.join('../../casestudyoutputs', 'exploratory', 'expl_util_avg'), util_avg)
np.save(os.path.join('../../casestudyoutputs', 'exploratory', 'expl_util_hi'), util_hi)
np.save(os.path.join('../../casestudyoutputs', 'exploratory', 'expl_util_lo'), util_lo)

''' 10-OCT-24
Want an interruptible/restartable utility estimation loop here
'''

def unif_design_mat(numTN, testmax, testint=1):
    """
    Generates a design matrix that allocates tests uniformly across all test nodes, for a max number of tests (testmax),
    a testing interval (testint), and a number of test nodes (numTN)
    """
    numcols = int(testmax/testint)
    testarr = np.arange(testint, testmax + testint, testint)
    des = np.zeros((numTN, int(testmax/testint)))
    for j in range(numcols):
        des[:, j] = np.ones(numTN) * np.floor(testarr[j]/numTN)
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
    numcols = int(testmax/testint)
    testarr = np.arange(testint, testmax + testint, testint)
    des = np.zeros((numTN, int(testmax/testint)))
    for j in range(numcols):
        des[:, j] = np.floor(np.divide(np.sum(Nexpl, axis=1), np.sum(Nexpl))*testarr[j])
        numtoadd = testarr[j] - np.sum(des[:, j])
        if numtoadd > 0:
            for k in range(int(numtoadd)):
                des[k, j] += 1

    return des / testarr


unif_mat = unif_design_mat(numTN, testmax, testint)
rudi_mat = rudi_design_mat(numTN, testmax, testint)

numreps = 10
stop = False
lastrep = 0
while not stop:
    # We want 10 evaluations of utility for each plan and testnum
    alloc = np.load(os.path.join('..', 'allprovinces', 'allprov_alloc.npy'))
    util_avg_greedy, util_hi_greedy, util_lo_greedy = np.load(os.path.join('..', 'allprovinces', 'util_avg_greedy_noout.npy')), \
        np.load(os.path.join('..', 'allprovinces', 'util_hi_greedy_noout.npy')), \
        np.load(os.path.join('..', 'allprovinces', 'util_lo_greedy_noout.npy'))
    util_avg_unif, util_hi_unif, util_lo_unif = np.load(
        os.path.join('..', 'allprovinces', 'util_avg_unif_noout.npy')), \
        np.load(os.path.join('..', 'allprovinces', 'util_hi_unif_noout.npy')), \
        np.load(os.path.join('..', 'allprovinces', 'util_lo_unif_noout.npy'))
    util_avg_rudi, util_hi_rudi, util_lo_rudi = np.load(
        os.path.join('..', 'allprovinces', 'util_avg_rudi_noout.npy')), \
        np.load(os.path.join('..', 'allprovinces', 'util_hi_rudi_noout.npy')), \
        np.load(os.path.join('..', 'allprovinces', 'util_lo_rudi_noout.npy'))

    # Stop if the last utility column isn't zero
    if util_avg_unif[-1, -1] > 0:
        stop = True
    else: # Do a set of utility estimates at the next zero
        # Index skips first column, which should be zeros for all
        currtup = (np.where(util_avg_unif[:, 1:] == 0)[0][0], np.where(util_avg_unif[:, 1:] == 0)[1][0])
        currrep = currtup[0]
        print("Current rep: "+str(currrep))
        currbudgetind = currtup[1]
        currbudget = testarr[currbudgetind]
        # Identify allocation to measure
        curralloc = alloc[:, currbudgetind + 1]
        des_greedy = curralloc / np.sum(curralloc)
        des_unif = unif_mat[:, currbudgetind]
        des_rudi = rudi_mat[:, currbudgetind]

        # Generate new base MCMC draws
        if lastrep != currrep: # Only generate again if we have moved to a new replication
            print("Generating base set of truth draws...")
            np.random.seed(1000+currrep)
            csdict_allprov = methods.GeneratePostSamples(csdict_allprov)
            lastrep = currrep

        '''
        # Greedy
        currlosslist = sampf.sampling_plan_loss_list_importance(des_greedy, currbudget, csdict_allprov, paramdict,
                                                                numimportdraws=10000,
                                                                numdatadrawsforimportance=5000,
                                                                impweightoutlierprop=0.005)
        avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
        util_avg_greedy[currrep, currbudgetind + 1] = paramdict['baseloss'] - avg_loss
        util_lo_greedy[currrep, currbudgetind + 1] = paramdict['baseloss'] - avg_loss_CI[1]
        util_hi_greedy[currrep, currbudgetind + 1] = paramdict['baseloss'] - avg_loss_CI[0]
        print(des_greedy)
        print('Utility at ' + str(currbudget) + ' tests, Greedy: ' + str(util_avg_greedy[currrep, currbudgetind + 1]))
        '''

        # Uniform
        currlosslist = sampf.sampling_plan_loss_list_importance(des_unif, currbudget, csdict_allprov, paramdict,
                                                                numimportdraws=10000,
                                                                numdatadrawsforimportance=5000,
                                                                impweightoutlierprop=0.000)
        avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
        util_avg_unif[currrep, currbudgetind + 1] = paramdict['baseloss'] - avg_loss
        util_lo_unif[currrep, currbudgetind + 1] = paramdict['baseloss'] - avg_loss_CI[1]
        util_hi_unif[currrep, currbudgetind + 1] = paramdict['baseloss'] - avg_loss_CI[0]
        print(des_unif)
        print('Utility at ' + str(currbudget) + ' tests, Uniform: ' + str(util_avg_unif[currrep, currbudgetind + 1]))

        # Rudimentary
        currlosslist = sampf.sampling_plan_loss_list_importance(des_rudi, currbudget, csdict_allprov, paramdict,
                                                                numimportdraws=10000,
                                                                numdatadrawsforimportance=5000,
                                                                impweightoutlierprop=0.000)
        avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
        util_avg_rudi[currrep, currbudgetind + 1] = paramdict['baseloss'] - avg_loss
        util_lo_rudi[currrep, currbudgetind + 1] = paramdict['baseloss'] - avg_loss_CI[1]
        util_hi_rudi[currrep, currbudgetind + 1] = paramdict['baseloss'] - avg_loss_CI[0]
        print(des_rudi)
        print('Utility at ' + str(currbudget) + ' tests, Rudimentary: ' + str(util_avg_rudi[currrep, currbudgetind + 1]))

        # Save updated objects
        #np.save(os.path.join('utilitypaper', 'allprovinces', 'util_avg_greedy_noout'), util_avg_greedy)
        #np.save(os.path.join('utilitypaper', 'allprovinces', 'util_hi_greedy_noout'), util_hi_greedy)
        #np.save(os.path.join('utilitypaper', 'allprovinces', 'util_lo_greedy_noout'), util_lo_greedy)
        np.save(os.path.join('..', 'allprovinces', 'util_avg_unif_noout'), util_avg_unif)
        np.save(os.path.join('..', 'allprovinces', 'util_hi_unif_noout'), util_hi_unif)
        np.save(os.path.join('..', 'allprovinces', 'util_lo_unif_noout'), util_lo_unif)
        np.save(os.path.join('..', 'allprovinces', 'util_avg_rudi_noout'), util_avg_rudi)
        np.save(os.path.join('..', 'allprovinces', 'util_hi_rudi_noout'), util_hi_rudi)
        np.save(os.path.join('..', 'allprovinces', 'util_lo_rudi_noout'), util_lo_rudi)
    # Plot utilities

    '''
    util_avg_arr = np.vstack(
        (np.concatenate((np.array([0]), np.true_divide(util_avg_greedy[:, 1:].sum(0), (util_avg_greedy[:, 1:]!=0).sum(0)))), 
         np.concatenate((np.array([0]), np.true_divide(util_avg_unif[:, 1:].sum(0), (util_avg_unif[:, 1:]!=0).sum(0)))), 
         np.concatenate((np.array([0]), np.true_divide(util_avg_rudi[:, 1:].sum(0), (util_avg_rudi[:, 1:]!=0).sum(0))))))
    util_hi_arr = np.vstack(
        (np.concatenate((np.array([0]), np.true_divide(util_hi_greedy[:, 1:].sum(0), (util_hi_greedy[:, 1:]!=0).sum(0)))), 
         np.concatenate((np.array([0]), np.true_divide(util_hi_unif[:, 1:].sum(0), (util_hi_unif[:, 1:]!=0).sum(0)))), 
         np.concatenate((np.array([0]), np.true_divide(util_hi_rudi[:, 1:].sum(0), (util_hi_rudi[:, 1:]!=0).sum(0))))))
    util_lo_arr = np.vstack(
        (np.concatenate((np.array([0]), np.true_divide(util_lo_greedy[:, 1:].sum(0), (util_lo_greedy[:, 1:]!=0).sum(0)))), 
         np.concatenate((np.array([0]), np.true_divide(util_lo_unif[:, 1:].sum(0), (util_lo_unif[:, 1:]!=0).sum(0)))), 
         np.concatenate((np.array([0]), np.true_divide(util_lo_rudi[:, 1:].sum(0), (util_lo_rudi[:, 1:]!=0).sum(0))))))
    '''

    '''
    util_avg_arr = np.vstack((np.average(util_avg_greedy, axis=0), np.average(util_avg_unif, axis=0), np.average(util_avg_rudi, axis=0)))
    util_hi_arr = np.vstack((np.average(util_hi_greedy, axis=0), np.average(util_hi_unif, axis=0), np.average(util_hi_rudi, axis=0)))
    util_lo_arr = np.vstack((np.average(util_lo_greedy, axis=0), np.average(util_lo_unif, axis=0), np.average(util_lo_rudi, axis=0)))
    

    # Plot
    util.plot_marg_util_CI(util_avg_arr, util_hi_arr, util_lo_arr, testmax=testmax, testint=testint,
                           titlestr='All-provinces setting, comparison with other approaches')
    '''

    for i in range(numreps):
        plt.plot(util_avg_greedy[i])
        plt.plot(util_avg_unif[i])
        plt.plot(util_avg_rudi[i])
    plt.show()




# PLot average
util_avg_rudi[4][37] = np.nan
plt.plot(np.average(util_avg_greedy, axis=0))
plt.plot(np.average(util_avg_unif, axis=0))
plt.plot(np.nanmean(util_avg_rudi, axis=0))
plt.show()

# Look at allocations at odd points
b = 19
np.average(util_avg_unif, axis=0)[b]
np.average(util_avg_greedy, axis=0)[b]
alloc[:, 19]
# unif
util.round_design_low(np.ones(numTN) / numTN, b*testint) / b*testint
# rudi
util.round_design_low(np.divide(np.sum(Nexpl, axis=1), np.sum(Nexpl)), b*testint) / b*testint





'''
END RESTARTABLE UTILITY ESTIMATION
'''

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
    currlosslist = sampf.sampling_plan_loss_list(des_unif, testarr[testind], csdict_allprov, paramdict)
    avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
    util_avg_unif[testind+1] = paramdict['baseloss'] - avg_loss
    util_lo_unif[testind+1] = paramdict['baseloss'] - avg_loss_CI[1]
    util_hi_unif[testind+1] = paramdict['baseloss'] - avg_loss_CI[0]
    print(des_unif)
    print('Utility at ' + str(testarr[testind]) + ' tests, Uniform: ' + str(util_avg_unif[testind+1]))
    # Rudimentary utility
    des_rudi = util.round_design_low(np.divide(np.sum(Nexpl, axis=1), np.sum(Nexpl)), testarr[testind]) / testarr[testind]
    currlosslist = sampf.sampling_plan_loss_list(des_rudi, testarr[testind], csdict_allprov, paramdict)
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
                               titlestr='Exploratory Setting, comparison with other approaches')

# Store matrices
np.save(os.path.join('../../casestudyoutputs', 'exploratory', 'util_avg_arr_expl'), util_avg_arr)
np.save(os.path.join('../../casestudyoutputs', 'exploratory', 'util_hi_arr_expl'), util_hi_arr)
np.save(os.path.join('../../casestudyoutputs', 'exploratory', 'util_lo_arr_expl'), util_lo_arr)