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
import time
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
numdraws = 5000
csdict_fam['numPostSamples'] = numdraws

paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN))

# Set limits of data collection and intervals for calculation
testmax, testint = 400, 10
testarr = np.arange(testint, testmax + testint, testint)

numdatadraws = 20
des = np.array([1., 0., 0., 0.])

np.random.seed(1050)  # To replicate draws later
csdict_fam = methods.GeneratePostSamples(csdict_fam)

numtruthdraws = 1000
truthdraws, datadraws = util.distribute_truthdata_draws(csdict_fam['postSamples'], numtruthdraws, numdatadraws)
paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
# Get base loss
paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)
print(paramdict['baseloss'])

paramdict['riskdict'].update({'slope':0.2})
paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)
print(paramdict['baseloss'])

# Use critical ratio and optimizer to find bayes estimates
q = paramdict['scoredict']['underestweight'] / (1+paramdict['scoredict']['underestweight'])
Wvec =  np.ones(truthdraws.shape[0]) / truthdraws.shape[0]
critratio_est = sampf.bayesest_critratio(truthdraws, Wvec, q)
opt_est = sampf.get_bayes_min(truthdraws, Wvec, paramdict)
print(critratio_est)
print(opt_est.x)
print(np.linalg.norm(critratio_est-opt_est.x))

paramdict['riskdict'].update({'slope': 0.8})
opt_est = sampf.get_bayes_min(truthdraws, Wvec, paramdict)
print(critratio_est)
print(opt_est.x)
print(np.linalg.norm(critratio_est-opt_est.x))






##############
##############
##############

numreps=8
truth15arr = np.zeros((numreps, testarr.shape[0]+1))
truth15arr_lo, truth15arr_hi = np.zeros((numreps, testarr.shape[0]+1)), np.zeros((numreps, testarr.shape[0]+1))
truth50arr = np.zeros((numreps, testarr.shape[0]+1))
truth50arr_lo, truth50arr_hi = np.zeros((numreps, testarr.shape[0]+1)), np.zeros((numreps, testarr.shape[0]+1))
truth15times, truth50times = [], []
for rep in range(numreps):
    np.random.seed(1050 + rep)  # To replicate draws later
    csdict_fam = methods.GeneratePostSamples(csdict_fam)
    # 50k truthdraws
    print('On 50k truth draws...')
    np.random.seed(200 + rep)
    numtruthdraws = 50000
    truthdraws, datadraws = util.distribute_truthdata_draws(csdict_fam['postSamples'], numtruthdraws, numdatadraws)
    paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
    # Get base loss
    paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)
    util.print_param_checks(paramdict) # Check of used parameters
    for testind in range(testarr.shape[0]):
        print(str(testarr[testind])+' tests...')
        time0 = time.time()
        currlosslist = sampf.sampling_plan_loss_list(des, testarr[testind], csdict_fam, paramdict)
        truth50times.append(time.time() - time0)
        print('Time: ' + str(truth50times[-1]))
        avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
        truth50arr[rep][testind+1] = paramdict['baseloss'] - avg_loss
        truth50arr_lo[rep][testind+1] = paramdict['baseloss'] - avg_loss_CI[1]
        truth50arr_hi[rep][testind+1] = paramdict['baseloss'] - avg_loss_CI[0]
    # 15k truthdraws
    print('On 15k truth draws...')
    np.random.seed(200 + rep)
    numtruthdraws = 15000
    truthdraws, datadraws = util.distribute_truthdata_draws(csdict_fam['postSamples'], numtruthdraws, numdatadraws)
    paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
    # Get base loss
    paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)
    util.print_param_checks(paramdict)  # Check of used parameters
    for testind in range(testarr.shape[0]):
        print(str(testarr[testind]) + ' tests...')
        time0 = time.time()
        currlosslist = sampf.sampling_plan_loss_list(des, testarr[testind], csdict_fam, paramdict)
        truth15times.append(time.time() - time0)
        print('Time: ' + str(truth15times[-1]))
        avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
        truth15arr[rep][testind + 1] = paramdict['baseloss'] - avg_loss
        truth15arr_lo[rep][testind + 1] = paramdict['baseloss'] - avg_loss_CI[1]
        truth15arr_hi[rep][testind + 1] = paramdict['baseloss'] - avg_loss_CI[0]
    # Plot
    util.plot_marg_util(np.vstack((truth15arr,truth50arr)),testmax,testint,al=0.2,type='delta',
                        labels=['15k' for i in range(numreps)]+['50k' for i in range(numreps)],
                        colors=['red' for i in range(numreps)]+['blue' for i in range(numreps)],
                        dashes=[[1,0] for i in range(numreps)]+[[2,1] for i in range(numreps)])

util.plot_marg_util_CI(truth15arr,truth15arr_hi,truth15arr_lo,400,10,utilmax=0.75)
util.plot_marg_util_CI(truth50arr,truth50arr_hi,truth50arr_lo,400,10,utilmax=0.75)

'''
np.save(os.path.join('studies', 'truthdraws_10JUN', 'truth15arr'), truth15arr)
np.save(os.path.join('studies', 'truthdraws_10JUN', 'truth15arr_hi'), truth15arr_hi)
np.save(os.path.join('studies', 'truthdraws_10JUN', 'truth15arr_lo'), truth15arr_lo)
np.save(os.path.join('studies', 'truthdraws_10JUN', 'truth50arr'), truth50arr)
np.save(os.path.join('studies', 'truthdraws_10JUN', 'truth50arr_hi'), truth50arr_hi)
np.save(os.path.join('studies', 'truthdraws_10JUN', 'truth50arr_lo'), truth50arr_lo)
np.save(os.path.join('studies', 'truthdraws_10JUN', 'truth15times'), np.array(truth15times))
np.save(os.path.join('studies', 'truthdraws_10JUN', 'truth50times'), np.array(truth50times))

truth15arr = np.load(os.path.join('studies', 'truthdraws_10JUN', 'truth15arr.npy'))
truth15arr_hi = np.load(os.path.join('studies', 'truthdraws_10JUN', 'truth15arr_hi.npy'))
truth15arr_lo = np.load(os.path.join('studies', 'truthdraws_10JUN', 'truth15arr_lo.npy'))
truth50arr = np.load(os.path.join('studies', 'truthdraws_10JUN', 'truth50arr.npy'))
truth50arr_hi = np.load(os.path.join('studies', 'truthdraws_10JUN', 'truth50arr_hi.npy'))
truth50arr_lo = np.load(os.path.join('studies', 'truthdraws_10JUN', 'truth50arr_lo.npy'))
truth15times = np.load(os.path.join('studies', 'truthdraws_10JUN', 'truth15times.npy'))
truth50times = np.load(os.path.join('studies', 'truthdraws_10JUN', 'truth50times.npy'))
'''

# Boxplot for times
plt.boxplot(np.vstack((truth15times,truth50times[:320])).T,labels=['15k', '50k'])
plt.ylim([0,380])
plt.title('Time needed to compile 2000 data draws')
plt.xlabel('Number of truth draws')
plt.ylabel('Seconds')
plt.show()
plt.close()
print(np.average(truth50times)/np.average(truth15times))
print(50/15)

# Show bias
truth15arr = truth15arr[:-2]
truth50arr = truth50arr[:-2]
truth15arr_hi = truth15arr_hi[:-2]
truth50arr_hi = truth50arr_hi[:-2]
truth15arr_lo = truth15arr_lo[:-2]
truth50arr_lo = truth50arr_lo[:-2]
margutilarr_avg = np.vstack((truth15arr, truth50arr))
margutilarr_hi = np.vstack((truth15arr_hi, truth50arr_hi))
margutilarr_lo = np.vstack((truth15arr_lo, truth50arr_lo))

'''
if len(colors) == 0:
    colors = cm.rainbow(np.linspace(0, 1, margutilarr_avg.shape[0]))
if len(dashes) == 0:
    dashes = [[2, desind + 1] for desind in range(margutilarr_avg.shape[0])]
if len(labels) == 0:
    labels = ['Design ' + str(desind + 1) for desind in range(margutilarr_avg.shape[0])]
'''
labels = ['15k', '50k']
al=0.1
x1 = range(0, testmax + 1, testint)
yMax = margutilarr_hi.max() * 1.1
for desind in range(margutilarr_avg.shape[0]):
    if desind == 0:
        plt.plot(x1, margutilarr_avg[desind],
             linewidth=1, color='blue',
             label=labels[0], alpha=al)
        #plt.fill_between(x1, margutilarr_lo[desind], margutilarr_hi[desind],
        #                 color='blue', alpha=0.3 * al)
    elif desind==8:
        plt.plot(x1, margutilarr_avg[desind],
             linewidth=1, color='red',
             label=labels[1], alpha=al)
        #plt.fill_between(x1, margutilarr_lo[desind], margutilarr_hi[desind],
        #                 color='red', alpha=0.3 * al)
    elif desind<8:
        plt.plot(x1, margutilarr_avg[desind],
                 linewidth=1, color='blue', alpha=al)
        plt.fill_between(x1, margutilarr_lo[desind], margutilarr_hi[desind],
                         color='blue', alpha=0.3 * al)
    elif desind>8:
        plt.plot(x1, margutilarr_avg[desind],
             linewidth=1, color='red', alpha=al)
        plt.fill_between(x1, margutilarr_lo[desind], margutilarr_hi[desind],
                         color='red', alpha=0.3 * al)
plt.legend()
plt.ylim([0., yMax])
plt.xlabel('Number of Tests')
plt.ylabel('Utility Gain')
plt.title('Utility with Increasing Tests at Test Node 1\n15k or 50k truth draws')
plt.show()
plt.close()

# Get sample standard deviation estimates
range15 = np.average(truth15arr_hi - truth15arr_lo,axis=0)
range50 = np.average(truth50arr_hi - truth50arr_lo,axis=0)
plt.plot(x1,range15,label='15k',color='blue')
plt.plot(x1,range50,label='50k', color='red')
plt.legend()
plt.title('Avg. 95% CI width vs. increasing tests\n15k or 50k truth draws')
plt.show()
plt.close()

############
# Do same analysis when increasing data draws by a commensurate amount
############
numdatadraws = 6666

numreps=8
data6667arr = np.zeros((numreps, testarr.shape[0]+1))
data6667arr_lo, data6667arr_hi = np.zeros((numreps, testarr.shape[0]+1)), np.zeros((numreps, testarr.shape[0]+1))
for rep in range(numreps):
    print('Rep '+str(rep)+'...')
    np.random.seed(1050 + rep)  # To replicate draws later
    csdict_fam = methods.GeneratePostSamples(csdict_fam)
    # 6667 datadraws
    np.random.seed(200 + rep)
    numtruthdraws = 15000
    truthdraws, datadraws_big = util.distribute_truthdata_draws(csdict_fam['postSamples'], numtruthdraws, numdatadraws)
    paramdict.update({'truthdraws': truthdraws})
    # Get base loss
    paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)
    util.print_param_checks(paramdict) # Check of used parameters
    for testind in range(testarr.shape[0]):
        paramdict.update({'datadraws': datadraws_big[:2222]})
        currlosslist = sampf.sampling_plan_loss_list(des, testarr[testind], csdict_fam, paramdict)
        paramdict.update({'datadraws': datadraws_big[2222:4444]})
        currlosslist = currlosslist + sampf.sampling_plan_loss_list(des, testarr[testind], csdict_fam, paramdict)
        paramdict.update({'datadraws': datadraws_big[4444:]})
        currlosslist = currlosslist + sampf.sampling_plan_loss_list(des, testarr[testind], csdict_fam, paramdict)
        avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
        data6667arr[rep][testind+1] = paramdict['baseloss'] - avg_loss
        data6667arr_lo[rep][testind+1] = paramdict['baseloss'] - avg_loss_CI[1]
        data6667arr_hi[rep][testind+1] = paramdict['baseloss'] - avg_loss_CI[0]
        print(str(testarr[testind]) + ' tests: '+str(paramdict['baseloss'] - avg_loss))
    # Plot
    util.plot_marg_util(np.vstack((truth15arr, data6667arr)), testmax, testint,al=0.2,type='delta',
                        labels=['2k' for i in range(numreps)]+['6.7k' for i in range(numreps)],
                        colors=['red' for i in range(numreps)]+['blue' for i in range(numreps)],
                        dashes=[[1,0] for i in range(numreps)]+[[2,1] for i in range(numreps)])

np.save(os.path.join('studies', 'truthdraws_10JUN', 'data6667arr'), data6667arr)
np.save(os.path.join('studies', 'truthdraws_10JUN', 'data6667arr_hi'), data6667arr_hi)
np.save(os.path.join('studies', 'truthdraws_10JUN', 'data6667arr_lo'), data6667arr_lo)

data6667arr = np.load(os.path.join('studies', 'truthdraws_10JUN', 'data6667arr.npy'))
data6667arr_hi = np.load(os.path.join('studies', 'truthdraws_10JUN', 'data6667arr_hi.npy'))
data6667arr_lo = np.load(os.path.join('studies', 'truthdraws_10JUN', 'data6667arr_lo.npy'))

margutilarr_avg = np.vstack((truth15arr, data6667arr))
margutilarr_hi = np.vstack((truth15arr_hi, data6667arr_hi))
margutilarr_lo = np.vstack((truth15arr_lo, data6667arr_lo))

labels = ['2k', '6.6k']
al=0.1
x1 = range(0, testmax + 1, testint)
yMax = margutilarr_hi.max() * 1.1
for desind in range(margutilarr_avg.shape[0]):
    if desind == 0:
        plt.plot(x1, margutilarr_avg[desind],
             linewidth=1, color='blue',
             label=labels[0], alpha=al)
        #plt.fill_between(x1, margutilarr_lo[desind], margutilarr_hi[desind],
        #                 color='blue', alpha=0.3 * al)
    elif desind==8:
        plt.plot(x1, margutilarr_avg[desind],
             linewidth=1, color='red',
             label=labels[1], alpha=al)
        #plt.fill_between(x1, margutilarr_lo[desind], margutilarr_hi[desind],
        #                 color='red', alpha=0.3 * al)
    elif desind<8:
        plt.plot(x1, margutilarr_avg[desind],
                 linewidth=1, color='blue', alpha=al)
        plt.fill_between(x1, margutilarr_lo[desind], margutilarr_hi[desind],
                         color='blue', alpha=0.3 * al)
    elif desind>8:
        plt.plot(x1, margutilarr_avg[desind],
             linewidth=1, color='red', alpha=al)
        plt.fill_between(x1, margutilarr_lo[desind], margutilarr_hi[desind],
                         color='red', alpha=0.3 * al)
plt.legend()
plt.ylim([0., yMax])
plt.xlabel('Number of Tests')
plt.ylabel('Utility Gain')
plt.title('Utility with Increasing Tests at Test Node 1\n2k or 6.6k data draws')
plt.show()
plt.close()

# Plot of CI widths
range2 = np.average(truth15arr_hi - truth15arr_lo,axis=0)
range6 = np.average(data6667arr_hi - data6667arr_lo,axis=0)
range50 = np.average(truth50arr_hi - truth50arr_lo,axis=0)
plt.plot(x1,range2,label='15k truth, 2k data',color='blue')
plt.plot(x1,range6,label='15k truth, 6.6k data', color='red')
plt.plot(x1,range50, 'r--', label='50k truth, 2k data')
plt.legend()
plt.title('Avg. 95% CI width vs. increasing tests')
plt.xlabel('Number of Tests')
plt.show()
plt.close()


#############
# How much bias is there at higher truth draws?
#############
numdatadraws = 50
minCI = 0.02

numreps=8
truth75arr = np.zeros((numreps, testarr.shape[0]+1))
truth75arr_lo, truth75arr_hi = np.zeros((numreps, testarr.shape[0]+1)), np.zeros((numreps, testarr.shape[0]+1))
truth100arr = np.zeros((numreps, testarr.shape[0]+1))
truth100arr_lo, truth100arr_hi = np.zeros((numreps, testarr.shape[0]+1)), np.zeros((numreps, testarr.shape[0]+1))
for rep in range(numreps):
    np.random.seed(1050 + rep)  # To replicate draws later
    csdict_fam = methods.GeneratePostSamples(csdict_fam)
    # 50k truthdraws
    print('On 75k truth draws...')
    np.random.seed(200 + rep)
    numtruthdraws = 75000
    truthdraws, datadraws = util.distribute_truthdata_draws(csdict_fam['postSamples'], numtruthdraws, numdatadraws)
    paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
    # Get base loss
    paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)
    util.print_param_checks(paramdict) # Check of used parameters
    for testind in range(testarr.shape[0]):
        print(str(testarr[testind])+' tests...')
        avg_loss_CI = (0, minCI*2)
        currlosslist = []
        while avg_loss_CI[1]-avg_loss_CI[0] > minCI:
            paramdict.update({'datadraws': truthdraws[choice(np.arange(numtruthdraws),size=numdatadraws,replace=False)]})
            currlosslist = currlosslist + sampf.sampling_plan_loss_list(des, testarr[testind], csdict_fam, paramdict)
            avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
            print('Current CI width: ' + str(avg_loss_CI[1]-avg_loss_CI[0]))
        truth75arr[rep][testind+1] = paramdict['baseloss'] - avg_loss
        truth75arr_lo[rep][testind+1] = paramdict['baseloss'] - avg_loss_CI[1]
        truth75arr_hi[rep][testind+1] = paramdict['baseloss'] - avg_loss_CI[0]
    # 100k truthdraws
    print('On 100k truth draws...')
    np.random.seed(200 + rep)
    numtruthdraws = 100000
    truthdraws, datadraws = util.distribute_truthdata_draws(csdict_fam['postSamples'], numtruthdraws, numdatadraws)
    paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
    # Get base loss
    paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)
    util.print_param_checks(paramdict)  # Check of used parameters
    for testind in range(testarr.shape[0]):
        print(str(testarr[testind]) + ' tests...')
        avg_loss_CI = (0, minCI * 2)
        currlosslist = []
        while avg_loss_CI[1] - avg_loss_CI[0] > minCI:
            paramdict.update({'datadraws': truthdraws[choice(np.arange(numtruthdraws), size=numdatadraws, replace=False)]})
            currlosslist = currlosslist + sampf.sampling_plan_loss_list(des, testarr[testind], csdict_fam, paramdict)
            avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
            print('Current CI width: ' + str(avg_loss_CI[1] - avg_loss_CI[0]))
        truth100arr[rep][testind + 1] = paramdict['baseloss'] - avg_loss
        truth100arr_lo[rep][testind + 1] = paramdict['baseloss'] - avg_loss_CI[1]
        truth100arr_hi[rep][testind + 1] = paramdict['baseloss'] - avg_loss_CI[0]
    # Plot
    margutilarr_avg = np.vstack((truth15arr, truth50arr, truth75arr, truth100arr))
    margutilarr_hi = np.vstack((truth15arr_hi, truth50arr_hi, truth75arr_hi, truth100arr_hi))
    margutilarr_lo = np.vstack((truth15arr_lo, truth50arr_lo, truth75arr_lo, truth100arr_lo))

    labels = ['15k', '50k', '75k', '100k']
    al = 0.1
    x1 = range(0, testmax + 1, testint)
    yMax = margutilarr_hi.max() * 1.1
    for desind in range(margutilarr_avg.shape[0]):
        if desind == 0:
            plt.plot(x1, margutilarr_avg[desind],
                     linewidth=1, color='blue',
                     label=labels[0], alpha=al)
            # plt.fill_between(x1, margutilarr_lo[desind], margutilarr_hi[desind],
            #                 color='blue', alpha=0.3 * al)
        elif desind == 8:
            plt.plot(x1, margutilarr_avg[desind],
                     linewidth=1, color='red',
                     label=labels[1], alpha=al)
            # plt.fill_between(x1, margutilarr_lo[desind], margutilarr_hi[desind],
            #                 color='red', alpha=0.3 * al)
        elif desind == 16:
            plt.plot(x1, margutilarr_avg[desind],
                     linewidth=1, color='purple',
                     label=labels[2], alpha=al)
        elif desind == 24:
            plt.plot(x1, margutilarr_avg[desind],
                     linewidth=1, color='orange',
                     label=labels[3], alpha=al)
        elif desind < 8:
            plt.plot(x1, margutilarr_avg[desind],
                     linewidth=1, color='blue', alpha=al)
            plt.fill_between(x1, margutilarr_lo[desind], margutilarr_hi[desind],
                             color='blue', alpha=0.3 * al)
        elif desind > 8 and desind < 16:
            plt.plot(x1, margutilarr_avg[desind],
                     linewidth=1, color='red', alpha=al)
            plt.fill_between(x1, margutilarr_lo[desind], margutilarr_hi[desind],
                             color='red', alpha=0.3 * al)
        elif desind > 16 and desind < 24:
            plt.plot(x1, margutilarr_avg[desind],
                     linewidth=1, color='purple', alpha=al)
            plt.fill_between(x1, margutilarr_lo[desind], margutilarr_hi[desind],
                             color='purple', alpha=0.3 * al)
        elif desind > 24:
            plt.plot(x1, margutilarr_avg[desind],
                     linewidth=1, color='purple', alpha=al)
            plt.fill_between(x1, margutilarr_lo[desind], margutilarr_hi[desind],
                             color='purple', alpha=0.3 * al)
    plt.legend()
    plt.ylim([0., yMax])
    plt.xlabel('Number of Tests')
    plt.ylabel('Utility Gain')
    plt.title('Utility with Increasing Tests at Test Node 1\n15k or 50k truth draws')
    plt.show()
    plt.close()