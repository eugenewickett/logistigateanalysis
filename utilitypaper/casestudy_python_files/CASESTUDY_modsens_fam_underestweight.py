"""Script for investigating allocation/utility sensitivity to prior variance selection"""

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
csdict_fam = util.initDataDict(Nfam, Yfam) # Initialize necessary logistigate keys
csdict_fam['TNnames'] = ['MOD_39', 'MOD_17', 'MODHIGH_95', 'MODHIGH_26']
csdict_fam['SNnames'] = ['MNFR ' + str(i + 1) for i in range(numSN)]

# Use observed data to form Q for tested nodes; use bootstrap data for untested nodes
numBoot = 44  # Average across each TN in original data set
SNprobs = np.sum(csdict_fam['N'], axis=0) / np.sum(csdict_fam['N'])
np.random.seed(33)
Qvecs = np.random.multinomial(numBoot, SNprobs, size=4) / numBoot
csdict_fam['Q'] = np.vstack((csdict_fam['Q'][:4], Qvecs))

# Build prior
SNpriorMean = np.repeat(sps.logit(0.1), numSN)
# Establish test node priors according to assessment by regulators
TNpriorMean = sps.logit(np.array([0.1, 0.1, 0.15, 0.15]))
TNvar, SNvar = 2., 4.  # Variances for use with prior; supply nodes are wide due to uncertainty
csdict_fam['prior'] = prior_normal_assort(np.concatenate((SNpriorMean, TNpriorMean)),
                               np.diag(np.concatenate((np.repeat(SNvar, numSN), np.repeat(TNvar, numTN)))))

# Set up MCMC
csdict_fam['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
# Generate posterior draws
numdraws = 75000
csdict_fam['numPostSamples'] = numdraws
np.random.seed(1000) # To replicate draws later
csdict_fam = methods.GeneratePostSamples(csdict_fam)

# Loss specification
paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=1., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN))

# Set limits of data collection and intervals for calculation
testmax, testint = 180, 10
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
# Store results
np.save(os.path.join('../../casestudyoutputs', 'modeling_sensitivity', 'fam_MS_underestweight_1_alloc'), alloc)
np.save(os.path.join('../../casestudyoutputs', 'modeling_sensitivity', 'fam_MS_underestweight_1_util_avg'), util_avg)
np.save(os.path.join('../../casestudyoutputs', 'modeling_sensitivity', 'fam_MS_underestweight_1_util_hi'), util_hi)
np.save(os.path.join('../../casestudyoutputs', 'modeling_sensitivity', 'fam_MS_underestweight_1_util_lo'), util_lo)

# Key comparison points
alloc90 = util_avg[9]
alloc180 = util_avg[18]
# Compare with Uniform and Rudimentary allocations
util_avg_unif_90 = []
util_avg_rudi_90 = []
# Do by 10 for uniform, by 50 for rudimentary
testarr_unif = np.arange(90, 131, 10)
for testnum in testarr_unif:
    if len(util_avg_unif_90) == 0 or not util_avg_unif_90[-1] > alloc90:
        des_unif = util.round_design_low(np.ones(numTN) / numTN, testnum) / testnum
        currlosslist = sampf.sampling_plan_loss_list(des_unif, testnum, csdict_fam, paramdict)
        avg_loss, _ = sampf.process_loss_list(currlosslist, zlevel=0.95)
        util_avg_unif_90.append(paramdict['baseloss'] - avg_loss)
        print('Unif at ' + str(testnum) + ' tests: ' + str(paramdict['baseloss'] - avg_loss))
testarr_rudi = np.arange(90, 491, 50)
for testnum in testarr_rudi:
    if len(util_avg_rudi_90) == 0 or not util_avg_rudi_90[-1] > alloc90: # Don't add more if we've exceeded the heuristic comparison
        des_rudi = util.round_design_low(np.divide(np.sum(Nfam, axis=1), np.sum(Nfam)), testnum) / testnum
        currlosslist = sampf.sampling_plan_loss_list(des_rudi, testnum, csdict_fam, paramdict)
        avg_loss, _ = sampf.process_loss_list(currlosslist, zlevel=0.95)
        util_avg_rudi_90.append(paramdict['baseloss'] - avg_loss)
        print('Rudi at ' + str(testnum) + ' tests: ' + str(paramdict['baseloss'] - avg_loss))

# Store
np.save(os.path.join('../../casestudyoutputs', 'modeling_sensitivity', 'fam_MS_underestweight_1_util_avg_unif_90'), util_avg_unif_90)
np.save(os.path.join('../../casestudyoutputs', 'modeling_sensitivity', 'fam_MS_underestweight_1_util_avg_rudi_90'), util_avg_rudi_90)

# Now for comparison with 180; do by 10 for both uniform and rudimentary
util_avg_unif_180 = []
util_avg_rudi_180 = []
testarr_unif = np.arange(180, 221, 10)
for testnum in testarr_unif:
    if len(util_avg_unif_180) == 0 or not util_avg_unif_180[-1] > alloc180: # Don't add more if we've exceeded the heuristic comparison
        des_unif = util.round_design_low(np.ones(numTN) / numTN, testnum) / testnum
        currlosslist = sampf.sampling_plan_loss_list(des_unif, testnum, csdict_fam, paramdict)
        avg_loss, _ = sampf.process_loss_list(currlosslist, zlevel=0.95)
        util_avg_unif_180.append(paramdict['baseloss'] - avg_loss)
        print('Unif at ' + str(testnum) + ' tests: ' + str(paramdict['baseloss'] - avg_loss))
testarr_rudi = np.arange(180, 681, 50)
for testnum in testarr_rudi:
    if len(util_avg_rudi_180) == 0 or not util_avg_rudi_180[-1] > alloc180: # Don't add more if we've exceeded the heuristic comparison
        des_rudi = util.round_design_low(np.divide(np.sum(Nfam, axis=1), np.sum(Nfam)), testnum) / testnum
        currlosslist = sampf.sampling_plan_loss_list(des_rudi, testnum, csdict_fam, paramdict)
        avg_loss, _ = sampf.process_loss_list(currlosslist, zlevel=0.95)
        util_avg_rudi_180.append(paramdict['baseloss'] - avg_loss)
        print('Rudi at ' + str(testnum) + ' tests: ' + str(paramdict['baseloss'] - avg_loss))

np.save(os.path.join('../../casestudyoutputs', 'modeling_sensitivity', 'fam_MS_underestweight_1_util_avg_unif_180'), util_avg_unif_180)
np.save(os.path.join('../../casestudyoutputs', 'modeling_sensitivity', 'fam_MS_underestweight_1_util_avg_rudi_180'), util_avg_rudi_180)

# Locate closest sample point for uniform and rudimentary to alloc90 and alloc180
kInd = next(x for x, val in enumerate(util_avg_unif_90) if val > alloc90)
unif90saved = round((alloc90 - util_avg_unif_90[kInd - 1]) / (util_avg_unif_90[kInd] - util_avg_unif_90[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_unif_180) if val > alloc180)
unif180saved = round((alloc180 - util_avg_unif_180[kInd-1]) / (util_avg_unif_180[kInd]-util_avg_unif_180[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_rudi_90) if val > alloc90)
rudi90saved = round((alloc90 - util_avg_rudi_90[kInd - 1]) / (util_avg_rudi_90[kInd] - util_avg_rudi_90[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
kInd = next(x for x, val in enumerate(util_avg_rudi_180) if val > alloc180)
rudi180saved = round((alloc180 - util_avg_rudi_180[kInd-1]) / (util_avg_rudi_180[kInd]-util_avg_rudi_180[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
print('Saved vs Unif at 90: '+str(unif90saved))
print('Saved vs Rudi at 90: '+str(rudi90saved))
print('Saved vs Unif at 180: '+str(unif180saved))
print('Saved vs Rudi at 180: '+str(rudi180saved))

###########################
# Use different underestweight
###########################
paramdict['scoredict'].update({'underestweight': 10.})

# Get base loss
paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)

util.print_param_checks(paramdict) # Check of used parameters
alloc, util_avg, util_hi, util_lo = sampf.get_greedy_allocation(csdict_fam, testmax, testint, paramdict,
                                                                printupdate=True, plotupdate=False)
# Store results
np.save(os.path.join('../../casestudyoutputs', 'modeling_sensitivity', 'fam_MS_underestweight_10_alloc'), alloc)
np.save(os.path.join('../../casestudyoutputs', 'modeling_sensitivity', 'fam_MS_underestweight_10_util_avg'), util_avg)
np.save(os.path.join('../../casestudyoutputs', 'modeling_sensitivity', 'fam_MS_underestweight_10_util_hi'), util_hi)
np.save(os.path.join('../../casestudyoutputs', 'modeling_sensitivity', 'fam_MS_underestweight_10_util_lo'), util_lo)

# Key comparison points
alloc90 = util_avg[9]
alloc180 = util_avg[18]
# Compare with Uniform and Rudimentary allocations
util_avg_unif_90 = []
util_avg_rudi_90 = []
# Do by 10 for uniform, by 50 for rudimentary
testarr_unif = np.arange(90, 131, 10)
for testnum in testarr_unif:
    if len(util_avg_unif_90) == 0 or not util_avg_unif_90[-1] > alloc90:
        des_unif = util.round_design_low(np.ones(numTN) / numTN, testnum) / testnum
        currlosslist = sampf.sampling_plan_loss_list(des_unif, testnum, csdict_fam, paramdict)
        avg_loss, _ = sampf.process_loss_list(currlosslist, zlevel=0.95)
        util_avg_unif_90.append(paramdict['baseloss'] - avg_loss)
        print('Unif at ' + str(testnum) + ' tests: ' + str(paramdict['baseloss'] - avg_loss))
testarr_rudi = np.arange(90, 491, 50)
for testnum in testarr_rudi:
    if len(util_avg_rudi_90) == 0 or not util_avg_rudi_90[-1] > alloc90: # Don't add more if we've exceeded the heuristic comparison
        des_rudi = util.round_design_low(np.divide(np.sum(Nfam, axis=1), np.sum(Nfam)), testnum) / testnum
        currlosslist = sampf.sampling_plan_loss_list(des_rudi, testnum, csdict_fam, paramdict)
        avg_loss, _ = sampf.process_loss_list(currlosslist, zlevel=0.95)
        util_avg_rudi_90.append(paramdict['baseloss'] - avg_loss)
        print('Rudi at ' + str(testnum) + ' tests: ' + str(paramdict['baseloss'] - avg_loss))

# Store
np.save(os.path.join('../../casestudyoutputs', 'modeling_sensitivity', 'fam_MS_underestweight_10_util_avg_unif_90'), util_avg_unif_90)
np.save(os.path.join('../../casestudyoutputs', 'modeling_sensitivity', 'fam_MS_underestweight_10_util_avg_rudi_90'), util_avg_rudi_90)

# Now for comparison with 180; do by 10 for both uniform and rudimentary
util_avg_unif_180 = []
util_avg_rudi_180 = []
testarr_unif = np.arange(180, 221, 10)
for testnum in testarr_unif:
    if len(util_avg_unif_180) == 0 or not util_avg_unif_180[-1] > alloc180: # Don't add more if we've exceeded the heuristic comparison
        des_unif = util.round_design_low(np.ones(numTN) / numTN, testnum) / testnum
        currlosslist = sampf.sampling_plan_loss_list(des_unif, testnum, csdict_fam, paramdict)
        avg_loss, _ = sampf.process_loss_list(currlosslist, zlevel=0.95)
        util_avg_unif_180.append(paramdict['baseloss'] - avg_loss)
        print('Unif at ' + str(testnum) + ' tests: ' + str(paramdict['baseloss'] - avg_loss))
testarr_rudi = np.arange(180, 681, 50)
for testnum in testarr_rudi:
    if len(util_avg_rudi_180) == 0 or not util_avg_rudi_180[-1] > alloc180: # Don't add more if we've exceeded the heuristic comparison
        des_rudi = util.round_design_low(np.divide(np.sum(Nfam, axis=1), np.sum(Nfam)), testnum) / testnum
        currlosslist = sampf.sampling_plan_loss_list(des_rudi, testnum, csdict_fam, paramdict)
        avg_loss, _ = sampf.process_loss_list(currlosslist, zlevel=0.95)
        util_avg_rudi_180.append(paramdict['baseloss'] - avg_loss)
        print('Rudi at ' + str(testnum) + ' tests: ' + str(paramdict['baseloss'] - avg_loss))

np.save(os.path.join('../../casestudyoutputs', 'modeling_sensitivity', 'fam_MS_underestweight_10_util_avg_unif_180'), util_avg_unif_180)
np.save(os.path.join('../../casestudyoutputs', 'modeling_sensitivity', 'fam_MS_underestweight_10_util_avg_rudi_180'), util_avg_rudi_180)

# Locate closest sample point for uniform and rudimentary to alloc90 and alloc180
kInd = next(x for x, val in enumerate(util_avg_unif_90) if val > alloc90)
unif90saved = round((alloc90 - util_avg_unif_90[kInd - 1]) / (util_avg_unif_90[kInd] - util_avg_unif_90[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_unif_180) if val > alloc180)
unif180saved = round((alloc180 - util_avg_unif_180[kInd-1]) / (util_avg_unif_180[kInd]-util_avg_unif_180[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_rudi_90) if val > alloc90)
rudi90saved = round((alloc90 - util_avg_rudi_90[kInd - 1]) / (util_avg_rudi_90[kInd] - util_avg_rudi_90[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
kInd = next(x for x, val in enumerate(util_avg_rudi_180) if val > alloc180)
rudi180saved = round((alloc180 - util_avg_rudi_180[kInd-1]) / (util_avg_rudi_180[kInd]-util_avg_rudi_180[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
print('Saved vs Unif at 90: '+str(unif90saved))
print('Saved vs Rudi at 90: '+str(rudi90saved))
print('Saved vs Unif at 180: '+str(unif180saved))
print('Saved vs Rudi at 180: '+str(rudi180saved))





