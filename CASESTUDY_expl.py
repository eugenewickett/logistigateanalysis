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

# Loss specification
paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN))

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

util.print_param_checks(paramdict) # Check of used parameters

''' 9-OCT-24
Want an interruptible/restartable greedy allocation loop here
'''
stop = False
while not stop:
    # Read in the current allocation
    alloc = np.load(os.path.join('utilitypaper', 'allprovinces', 'allprov_alloc.npy'))
    util_avg = np.load(os.path.join('utilitypaper', 'allprovinces', 'allprov_util_avg.npy'))
    util_hi = np.load(os.path.join('utilitypaper', 'allprovinces', 'allprov_util_hi.npy'))
    util_lo = np.load(os.path.join('utilitypaper', 'allprovinces', 'allprov_util_lo.npy'))
    # Stop if the last allocation column is empty
    if np.sum(alloc[:, -1]) > 0:
        stop = True
    else:
        testnumind = np.argmax(np.sum(alloc, axis=0))
        bestalloc = alloc[:, testnumind]
        nextTN = -1 # Initialize next best node
        currbestloss_avg, currbestloss_CI = -1, (-1, -1) # Initialize next best utility
        for currTN in range(numTN):  # Loop through each test node and identify best direction via lowest avg loss
            curralloc = bestalloc.copy()
            curralloc[currTN] += 1  # Increment 1 at current test node
            testnum = np.sum(curralloc) * testint
            currdes = curralloc / np.sum(curralloc)  # Make a proportion design
            currlosslist = sampf.sampling_plan_loss_list_importance(currdes, testnum, csdict_expl, paramdict,
                                                              numimportdraws=60000,
                                                              numdatadrawsforimportance=5000,
                                                              impweightoutlierprop=0.005)
            currloss_avg, currloss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
            print('TN ' + str(currTN) + ' loss avg.: ' + str(currloss_avg))
            if nextTN == -1 or currloss_avg < currbestloss_avg:  # Update with better loss
                nextTN = currTN
                currbestloss_avg = currloss_avg
                currbestloss_CI = currloss_CI
        # Store best results
        alloc[:, testnumind + 1] = bestalloc.copy()
        alloc[nextTN, testnumind + 1] += 1
        util_avg[testnumind + 1] = paramdict['baseloss'] - currbestloss_avg
        util_hi[testnumind + 1] = paramdict['baseloss'] - currbestloss_CI[0]
        util_lo[testnumind + 1] = paramdict['baseloss'] - currbestloss_CI[1]
        # Save as numpy objects
        np.save(os.path.join('utilitypaper', 'allprovinces', 'allprov_alloc'), alloc)
        np.save(os.path.join('utilitypaper', 'allprovinces', 'allprov_util_avg'), util_avg)
        np.save(os.path.join('utilitypaper', 'allprovinces', 'allprov_util_hi'), util_hi)
        np.save(os.path.join('utilitypaper', 'allprovinces', 'allprov_util_lo'), util_lo)
        print('TN ' + str(nextTN) + ' added, with utility CI of (' + str(util_lo[testnumind + 1]) + ', ' +
              str(util_hi[testnumind + 1]) + ') for ' + str(testnum) + ' tests')
        numint = util_avg.shape[0]
        util.plot_marg_util_CI(util_avg.reshape(1, numint), util_hi.reshape(1, numint), util_lo.reshape(1, numint),
                               testmax, testint, titlestr='')
        util.plot_plan(alloc, np.arange(0, testmax + 1, testint), testint, titlestr='')
'''
END RESTARTABLE GREEDY ALLOCATION
'''



alloc, util_avg, util_hi, util_lo = sampf.get_greedy_allocation(csdict_expl, testmax, testint, paramdict,
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

np.save(os.path.join('casestudyoutputs', 'exploratory', 'expl_alloc'), alloc)
np.save(os.path.join('casestudyoutputs', 'exploratory', 'expl_util_avg'), util_avg)
np.save(os.path.join('casestudyoutputs', 'exploratory', 'expl_util_hi'), util_hi)
np.save(os.path.join('casestudyoutputs', 'exploratory', 'expl_util_lo'), util_lo)

''' 10-OCT-24
Want an interruptible/restartable utility estimation loop here
'''
numreps = 10
stop = False
while not stop:
    # We want 10 evaluations of utility for each plan and testnum
    util_avg_greedy, util_hi_greedy, util_lo_greedy = np.empty((numreps, int(testmax / testint) + 1)), \
        np.empty((numreps, int(testmax / testint) + 1)), \
        np.empty((numreps, int(testmax / testint) + 1))
    util_avg_unif, util_hi_unif, util_lo_unif = np.empty((numreps, int(testmax / testint) + 1)), \
        np.empty((numreps, int(testmax / testint) + 1)), \
        np.empty((numreps, int(testmax / testint) + 1))
    util_avg_rudi, util_hi_rudi, util_lo_rudi = np.zeros((numreps, int(testmax / testint) + 1)), \
        np.empty((numreps, int(testmax / testint) + 1)), \
        np.empty((numreps, int(testmax / testint) + 1))

    np.save(os.path.join('utilitypaper', 'allprovinces', 'util_lo_rudi'), util_lo_rudi)

    # Read in the current allocation
    alloc = np.load(os.path.join('utilitypaper', 'allprovinces', 'allprov_alloc.npy'))
    util_avg = np.load(os.path.join('utilitypaper', 'allprovinces', 'allprov_util_avg.npy'))
    util_hi = np.load(os.path.join('utilitypaper', 'allprovinces', 'allprov_util_hi.npy'))
    util_lo = np.load(os.path.join('utilitypaper', 'allprovinces', 'allprov_util_lo.npy'))
    # Stop if the last allocation column is empty
    if np.sum(alloc[:, -1]) > 0:
        stop = True
    else:
        testnumind = np.argmax(np.sum(alloc, axis=0))
        bestalloc = alloc[:, testnumind]
        nextTN = -1 # Initialize next best node
        currbestloss_avg, currbestloss_CI = -1, (-1, -1) # Initialize next best utility
        for currTN in range(numTN):  # Loop through each test node and identify best direction via lowest avg loss
            curralloc = bestalloc.copy()
            curralloc[currTN] += 1  # Increment 1 at current test node
            testnum = np.sum(curralloc) * testint
            currdes = curralloc / np.sum(curralloc)  # Make a proportion design
            currlosslist = sampf.sampling_plan_loss_list_importance(currdes, testnum, csdict_expl, paramdict,
                                                              numimportdraws=60000,
                                                              numdatadrawsforimportance=5000,
                                                              impweightoutlierprop=0.005)
            currloss_avg, currloss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
            print('TN ' + str(currTN) + ' loss avg.: ' + str(currloss_avg))
            if nextTN == -1 or currloss_avg < currbestloss_avg:  # Update with better loss
                nextTN = currTN
                currbestloss_avg = currloss_avg
                currbestloss_CI = currloss_CI
        # Store best results
        alloc[:, testnumind + 1] = bestalloc.copy()
        alloc[nextTN, testnumind + 1] += 1
        util_avg[testnumind + 1] = paramdict['baseloss'] - currbestloss_avg
        util_hi[testnumind + 1] = paramdict['baseloss'] - currbestloss_CI[0]
        util_lo[testnumind + 1] = paramdict['baseloss'] - currbestloss_CI[1]
        # Save as numpy objects
        np.save(os.path.join('utilitypaper', 'allprovinces', 'allprov_alloc'), alloc)
        np.save(os.path.join('utilitypaper', 'allprovinces', 'allprov_util_avg'), util_avg)
        np.save(os.path.join('utilitypaper', 'allprovinces', 'allprov_util_hi'), util_hi)
        np.save(os.path.join('utilitypaper', 'allprovinces', 'allprov_util_lo'), util_lo)
        print('TN ' + str(nextTN) + ' added, with utility CI of (' + str(util_lo[testnumind + 1]) + ', ' +
              str(util_hi[testnumind + 1]) + ') for ' + str(testnum) + ' tests')
        numint = util_avg.shape[0]
        util.plot_marg_util_CI(util_avg.reshape(1, numint), util_hi.reshape(1, numint), util_lo.reshape(1, numint),
                               testmax, testint, titlestr='')
        util.plot_plan(alloc, np.arange(0, testmax + 1, testint), testint, titlestr='')
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
                               titlestr='Exploratory Setting, comparison with other approaches')

# Store matrices
np.save(os.path.join('casestudyoutputs', 'exploratory', 'util_avg_arr_expl'), util_avg_arr)
np.save(os.path.join('casestudyoutputs', 'exploratory', 'util_hi_arr_expl'), util_hi_arr)
np.save(os.path.join('casestudyoutputs', 'exploratory', 'util_lo_arr_expl'), util_lo_arr)