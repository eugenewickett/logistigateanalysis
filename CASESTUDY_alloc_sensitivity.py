"""
This script illustrates that although heuristic allocations may differ slightly depending on MCMC draws, the resulting
utility of these allocations are nearly identical
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
csdict_fam = util.initDataDict(Nfam, Yfam) # Initialize necessary logistigate keys
csdict_fam['TNnames'] = ['MOD_39', 'MOD_17', 'MODHIGH_95', 'MODHIGH_26']
csdict_fam['SNnames'] = ['MNFR ' + str(i+1) for i in range(numSN)]

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

# Loss specification
paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN))

# Set limits of data collection and intervals for calculation
testmax, testint = 400, 10
testarr = np.arange(testint, testmax + testint, testint)

# How many data draws should we use for importance sampling? Plot proportion of data set for each trace vs.
    # number of data points, across the data set size, for a uniform allocation across test nodes

des = np.ones(numTN) / numTN
datasim_max, datasim_by = 20000,500
flatarr = np.empty((numTN*numSN, int(datasim_max/datasim_by)))
for reps in range(20):
    for numdatadrawsforimp_ind, numdatadrawsforimp in enumerate(range(datasim_by, datasim_max, datasim_by)):
        sampMat = util.generate_sampling_array(des, testmax, roundalg = 'lo')
        NMat = np.moveaxis(np.array([np.random.multinomial(sampMat[tnInd], csdict_fam['Q'][tnInd], size=numdatadrawsforimp)
                                     for tnInd in range(numTN)]), 1, 0).astype(int)
        # Get average rounded data set from these few draws
        NMatAvg = np.round(np.average(NMat, axis=0)).astype(int)
        flatarr[:, numdatadrawsforimp_ind] = NMatAvg.flatten()

    plt.plot(range(datasim_by,datasim_max+datasim_by,datasim_by), flatarr.T, color="black", linewidth=0.2)

plt.show() # 10k seems like enough

# Set MCMC draws to use in fast algorithm
numtruthdraws, numdatadraws = 75000, 3000

numReps = 10
for rep in range(1, numReps):
    print('Rep: '+str(rep))
    # Get new MCMC draws
    np.random.seed(2000+rep)
    csdict_fam = methods.GeneratePostSamples(csdict_fam)
    # Get random subsets for truth and data draws
    truthdraws, datadraws = util.distribute_truthdata_draws(csdict_fam['postSamples'], numtruthdraws, numdatadraws)
    paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
    # Get base loss
    paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)
    util.print_param_checks(paramdict) # Check of used parameters
    alloc, util_avg, util_hi, util_lo = sampf.get_greedy_allocation(csdict_fam, testmax, testint, paramdict,
                                                                    numimpdraws=35000, numdatadrawsforimp=10000,
                                                                    impwtoutlierprop=0.01,
                                                                    plotupdate=False)
    # Store
    np.save(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'allocsens_alloc_'+str(rep)), alloc)
    np.save(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'allocsens_util_avg_'+str(rep)), util_avg)
    np.save(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'allocsens_util_hi_'+str(rep)), util_hi)
    np.save(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'allocsens_util_lo_'+str(rep)), util_lo)

##################
# Compare utilities and allocations
##################
alloc_list, util_avg_list, util_hi_list, util_lo_list = [], [], [], []
for i in range(9):
    alloc_list.append(np.load(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'allocsens_alloc_'+str(i)+'.npy')))
    util_avg_list.append(np.load(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'allocsens_util_avg_'+str(i)+'.npy')))
    util_hi_list.append(np.load(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'allocsens_util_hi_' + str(i) + '.npy')))
    util_lo_list.append(np.load(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'allocsens_util_lo_' + str(i) + '.npy')))

util_avg_list = np.array(util_avg_list)
util_hi_list = np.array(util_hi_list)
util_lo_list = np.array(util_lo_list)

### PLOT OF UTILITIES
colors = ['g', 'r']
dashes = [[5,2], [2,4]]
x1 = range(0, 401, 10)
yMax = 1.4
for desind in range(util_avg_list.shape[0]):
    if desind == 0:
        plt.plot(x1, util_hi_list[desind], dashes=dashes[0],
                 linewidth=0.7, color=colors[0],label='Upper 95% CI')
        plt.plot(x1, util_lo_list[desind], dashes=dashes[1], label='Lower 95% CI',
                 linewidth=0.7, color=colors[1])
    else:
        plt.plot(x1, util_hi_list[desind], dashes=dashes[0],
                 linewidth=0.7, color=colors[0])
        plt.plot(x1, util_lo_list[desind], dashes=dashes[1],
                 linewidth=0.7, color=colors[1])
    #plt.fill_between(x1, margutilarr_lo[desind], margutilarr_hi[desind],
    #                color=colors[desind], alpha=0.3 * al)
plt.legend()
plt.ylim([0., yMax])
plt.xlabel('Number of Tests')
plt.ylabel('Utility')
plt.title('Utility vs. Sampling Budget\n10 Replications of Heuristic in Regular Setting')
plt.show()
plt.close()

### PLOT OF ALLOCATIONS
alloc_list = np.array(alloc_list)

colorsset = plt.get_cmap('Set1')
colorinds = [6,1,2,3]
colors = np.array([colorsset(i) for i in colorinds])
x1 = range(0, 401, 10)

_ = plt.figure(figsize=(13,8))
labels = ['Moderate(39)', 'Moderate(17)', 'ModeratelyHigh(95)', 'ModeratelyHigh(26)']
for allocarr_ind in range(alloc_list.shape[0]):
    allocarr = alloc_list[allocarr_ind]
    if allocarr_ind == 0:
        for tnind in range(allocarr.shape[0]):
            plt.plot(x1, allocarr[tnind]*testint,
                     linewidth=3, color=colors[tnind],
                     label=labels[tnind], alpha=0.2)
    else:
        for tnind in range(allocarr.shape[0]):
            plt.plot(x1, allocarr[tnind]*testint,
                     linewidth=3, color=colors[tnind],  alpha=0.2)
allocmax = 200
plt.legend(fontsize=12)
plt.ylim([0., allocmax])
plt.xlabel('Number of Tests', fontsize=14)
plt.ylabel('Test Node Allocation', fontsize=14)
plt.title('Test Node Allocation\n10 Replications of Heuristic in Regular Setting',
          fontsize=18)
plt.tight_layout()
plt.show()
plt.close()

#####################################
### PLOT OF BIAS FROM LOW TRUTH DRAWS
#####################################
numdatadraws = 2000
# Reduce the number of reps
numReps = 5
# 15k
for rep in range(numReps):
    print('Rep: '+str(rep)+' for 15k')
    # Get new MCMC draws
    np.random.seed(2000+rep)
    numdraws, numtruthdraws = 15000, 15000
    csdict_fam['numPostSamples'] = numdraws
    csdict_fam = methods.GeneratePostSamples(csdict_fam)
    # Get random subsets for truth and data draws
    truthdraws, datadraws = util.distribute_truthdata_draws(csdict_fam['postSamples'], numtruthdraws, numdatadraws)
    paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
    # Get base loss
    paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)
    util.print_param_checks(paramdict) # Check of used parameters
    alloc, util_avg, util_hi, util_lo = sampf.get_greedy_allocation(csdict_fam, testmax, testint, paramdict,
                                                                    plotupdate=False)
    # Store
    np.save(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'truthbias_15k_util_avg_'+str(rep)), util_avg)
    np.save(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'truthbias_15k_util_hi_'+str(rep)), util_hi)
    np.save(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'truthbias_15k_util_lo_'+str(rep)), util_lo)
# 50k
for rep in range(numReps):
    print('Rep: '+str(rep)+' for 50k')
    # Get new MCMC draws
    np.random.seed(2000+rep)
    numdraws, numtruthdraws = 50000, 50000
    csdict_fam['numPostSamples'] = numdraws
    csdict_fam = methods.GeneratePostSamples(csdict_fam)
    # Get random subsets for truth and data draws
    truthdraws, datadraws = util.distribute_truthdata_draws(csdict_fam['postSamples'], numtruthdraws, numdatadraws)
    paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
    # Get base loss
    paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)
    util.print_param_checks(paramdict) # Check of used parameters
    alloc, util_avg, util_hi, util_lo = sampf.get_greedy_allocation(csdict_fam, testmax, testint, paramdict,
                                                                    plotupdate=False)
    # Store
    np.save(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'truthbias_50k_util_avg_'+str(rep)), util_avg)
    np.save(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'truthbias_50k_util_hi_'+str(rep)), util_hi)
    np.save(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'truthbias_50k_util_lo_'+str(rep)), util_lo)
# 100k
for rep in range(numReps):
    print('Rep: '+str(rep)+' for 100k')
    # Get new MCMC draws
    np.random.seed(2000+rep)
    numdraws, numtruthdraws = 100000, 100000
    csdict_fam['numPostSamples'] = numdraws
    csdict_fam = methods.GeneratePostSamples(csdict_fam)
    # Get random subsets for truth and data draws
    truthdraws, datadraws = util.distribute_truthdata_draws(csdict_fam['postSamples'], numtruthdraws, numdatadraws)
    paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
    # Get base loss
    paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)
    util.print_param_checks(paramdict) # Check of used parameters
    alloc, util_avg, util_hi, util_lo = sampf.get_greedy_allocation(csdict_fam, testmax, testint, paramdict,
                                                                    plotupdate=False)
    # Store
    np.save(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'truthbias_100k_util_avg_'+str(rep)), util_avg)
    np.save(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'truthbias_100k_util_hi_'+str(rep)), util_hi)
    np.save(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'truthbias_100k_util_lo_'+str(rep)), util_lo)

#####################################
### PLOT OF VARIANCE FROM DATA DRAWS
#####################################
numdraws, numtruthdraws = 75000, 75000

numReps = 5

# 500
for rep in range(numReps):
    print('Rep: '+str(rep)+' for 500')
    # Get new MCMC draws
    np.random.seed(2000+rep)
    numdatadraws = 500
    csdict_fam['numPostSamples'] = numdraws
    csdict_fam = methods.GeneratePostSamples(csdict_fam)
    # Get random subsets for truth and data draws
    truthdraws, datadraws = util.distribute_truthdata_draws(csdict_fam['postSamples'], numtruthdraws, numdatadraws)
    paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
    # Get base loss
    paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)
    util.print_param_checks(paramdict) # Check of used parameters
    alloc, util_avg, util_hi, util_lo = sampf.get_greedy_allocation(csdict_fam, testmax, testint, paramdict,
                                                                    plotupdate=False)
    # Store
    np.save(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'datavar_500_util_avg_'+str(rep)), util_avg)
    np.save(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'datavar_500_util_hi_'+str(rep)), util_hi)
    np.save(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'datavar_500_util_lo_'+str(rep)), util_lo)
# 1000
for rep in range(numReps):
    print('Rep: ' + str(rep) + ' for 1000')
    # Get new MCMC draws
    np.random.seed(2000 + rep)
    numdatadraws = 1000
    csdict_fam['numPostSamples'] = numdraws
    csdict_fam = methods.GeneratePostSamples(csdict_fam)
    # Get random subsets for truth and data draws
    truthdraws, datadraws = util.distribute_truthdata_draws(csdict_fam['postSamples'], numtruthdraws, numdatadraws)
    paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
    # Get base loss
    paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)
    util.print_param_checks(paramdict)  # Check of used parameters
    alloc, util_avg, util_hi, util_lo = sampf.get_greedy_allocation(csdict_fam, testmax, testint, paramdict,
                                                                    plotupdate=False)
    # Store
    np.save(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'datavar_1000_util_avg_' + str(rep)), util_avg)
    np.save(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'datavar_1000_util_hi_' + str(rep)), util_hi)
    np.save(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'datavar_1000_util_lo_' + str(rep)), util_lo)
# 3000
for rep in range(numReps):
    print('Rep: ' + str(rep) + ' for 3000')
    # Get new MCMC draws
    np.random.seed(2000 + rep)
    numdatadraws = 3000
    csdict_fam['numPostSamples'] = numdraws
    csdict_fam = methods.GeneratePostSamples(csdict_fam)
    # Get random subsets for truth and data draws
    truthdraws, datadraws = util.distribute_truthdata_draws(csdict_fam['postSamples'], numtruthdraws, numdatadraws)
    paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
    # Get base loss
    paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)
    util.print_param_checks(paramdict)  # Check of used parameters
    alloc, util_avg, util_hi, util_lo = sampf.get_greedy_allocation(csdict_fam, testmax, testint, paramdict,
                                                                    plotupdate=False)
    # Store
    np.save(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'datavar_3000_util_avg_' + str(rep)), util_avg)
    np.save(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'datavar_3000_util_hi_' + str(rep)), util_hi)
    np.save(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'datavar_3000_util_lo_' + str(rep)), util_lo)