from logistigate.logistigate import utilities as util # Pull from the submodule "develop" branch
from logistigate.logistigate import methods
from logistigate.logistigate.priors import prior_normal_assort
from logistigate.logistigate import lossfunctions as lf
from logistigate.logistigate import samplingplanfunctions as sampf

import os
import numpy as np
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
csdict_fam = util.initDataDict(Nfam, Yfam) # Initialize necessary logistigate keys
csdict_fam['TNnames'] = ['MOD_39', 'MOD_17', 'MODHIGH_95', 'MODHIGH_26']
csdict_fam['SNnames'] = ['MNFR ' + str(i+1) for i in range(numSN)]

# Region catchment proportions, for market terms
TNcach = np.array([0.17646, 0.05752, 0.09275, 0.09488])
TNcach = TNcach[:4] / np.sum(TNcach[:4])
SNcach = np.matmul(TNcach, csdict_fam['Q'])

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
# Generate posterior draws
numdraws = 75000
csdict_fam['numPostSamples'] = numdraws
np.random.seed(1000) # To replicate draws later
csdict_fam = methods.GeneratePostSamples(csdict_fam)

# Loss specification
paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.concatenate((SNcach, TNcach)))

# Set limits of data collection and intervals for calculation
testmax, testint = 400, 10
testarr = np.arange(testint, testmax + testint, testint)

# Set MCMC draws to use in fast algorithm
numtruthdraws, numdatadraws = 75000, 2000
# Get random subsets for truth and data draws
np.random.seed(444)
truthdraws, datadraws = util.distribute_truthdata_draws(csdict_fam['postSamples'], numtruthdraws, numdatadraws)
paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
# Get base loss
paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)

util.print_param_checks(paramdict) # Check of used parameters

alloc, util_avg, util_hi, util_lo = sampf.get_greedy_allocation(csdict_fam, testmax, testint, paramdict,
                                                                printupdate=True, plotupdate=False)
'''23-JUN
[1, 0, 0, 0, 1, 0, 0, 1, 3,
 3, 0, 1, 0, 0, 2, 1, 0, 1,
 0, 0, 3, 0, 3, 1, 3, 0, 0,
 2, 3, 2, 1, 1, 3, 2, 1, 1,
 2, 0, 2, 1]
[(0.005342285170051753, 0.005626355417830631),(0.010000199034713747, 0.010664777128378072),(0.014322009776225247, 0.015100118325744288),
(0.018056740105196895, 0.01888031949615457),(0.02130529362972633, 0.02216762586322807),(0.024435828681957317, 0.025336115883716814),
(0.027224196900375558, 0.02811264405733907),(0.0300358770810078, 0.030960596328848122),(0.03273356444092884, 0.03367683973142965),
(0.0347883742545253, 0.035740089751312654),(0.03694847283298697, 0.03792889250186432),(0.03931236685695447, 0.04027098795569731),
(0.04110424939477186, 0.04207307731113591),(0.04322271214186285, 0.04420329757219832),(0.044931286500236645, 0.04594338308520768),
 (0.0467783620982096, 0.04780940198987074),(0.04858624623784205, 0.049598602214360604),(0.04992492938697654, 0.05096686992072791),
(0.051799482380469206, 0.05285014320747282),(0.05347908594652244, 0.05456732820824446),(0.05511620188897205, 0.05622691778857784),
(0.05693766613288316, 0.05808261292701643), (0.058492653341450915, 0.05962468443773562), (0.05968580635159576, 0.06082911626334053),
(0.061284579277427975, 0.06249198122354993), (0.06241892479416715, 0.06361967230542005), (0.06371678420219659, 0.06497822309570259),
(0.06529486186328179, 0.06666600834458766), (0.0666470658715127, 0.06798820520890111), (0.06784955844335837, 0.06923265386759443),
(0.06931165025192532, 0.07072323531149652), (0.0704073872485808, 0.07181945665762997), (0.07214706265170538, 0.0737002791869235),
(0.07308989469183409, 0.07459143801092673), (0.07485967997031646, 0.07646948643539432), (0.07570813486904208, 0.07729971617469121),
(0.07780930388860322, 0.07942828192463991), (0.07875109354011652, 0.08040265401131814), (0.07978567868256958, 0.08149820765597462),
(0.08144101149483135, 0.0832346775162082)]
'''
np.save(os.path.join('../../casestudyoutputs', 'familiar', 'fam_market_alloc'), alloc)
np.save(os.path.join('../../casestudyoutputs', 'familiar', 'fam_market_util_avg'), util_avg)
np.save(os.path.join('../../casestudyoutputs', 'familiar', 'fam_market_util_hi'), util_hi)
np.save(os.path.join('../../casestudyoutputs', 'familiar', 'fam_market_util_lo'), util_lo)

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
    currlosslist = sampf.sampling_plan_loss_list(des_unif, testarr[testind], csdict_fam, paramdict)
    avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
    util_avg_unif[testind+1] = paramdict['baseloss'] - avg_loss
    util_lo_unif[testind+1] = paramdict['baseloss'] - avg_loss_CI[1]
    util_hi_unif[testind+1] = paramdict['baseloss'] - avg_loss_CI[0]
    print(des_unif)
    print('Utility at ' + str(testarr[testind]) + ' tests, Uniform: ' + str(util_avg_unif[testind+1]))
    # Rudimentary utility
    des_rudi = util.round_design_low(np.divide(np.sum(Nfam, axis=1), np.sum(Nfam)), testarr[testind]) / testarr[testind]
    currlosslist = sampf.sampling_plan_loss_list(des_rudi, testarr[testind], csdict_fam, paramdict)
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
                               titlestr='Familiar Setting with Market Term, comparison with other approaches')

# Store matrices
np.save(os.path.join('../../casestudyoutputs', 'familiar', 'util_avg_arr_fam_market'), util_avg_arr)
np.save(os.path.join('../../casestudyoutputs', 'familiar', 'util_hi_arr_fam_market'), util_hi_arr)
np.save(os.path.join('../../casestudyoutputs', 'familiar', 'util_lo_arr_fam_market'), util_lo_arr)

targind = 5 # where do we want to gauge budget savings?
targval = util_avg_arr[0][targind]

# Uniform
kInd = next(x for x, val in enumerate(util_avg_arr[1].tolist()) if val > targval)
unif_saved = round((targval - util_avg_arr[1][kInd - 1]) / (util_avg_arr[1][kInd] - util_avg_arr[1][kInd - 1]) *\
                      testint) + (kInd - 1) * testint - targind*testint
print(unif_saved)  #
# Rudimentary
kInd = next(x for x, val in enumerate(util_avg_arr[2].tolist()) if val > targval)
rudi_saved = round((targval - util_avg_arr[2][kInd - 1]) / (util_avg_arr[2][kInd] - util_avg_arr[2][kInd - 1]) *\
                      testint) + (kInd - 1) * testint - targind*testint
print(rudi_saved)  #