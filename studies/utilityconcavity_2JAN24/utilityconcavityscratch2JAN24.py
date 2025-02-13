from logistigate.logistigate import utilities as util # Pull from the submodule "develop" branch
from logistigate.logistigate import methods
from logistigate.logistigate.priors import prior_normal_assort
from logistigate.logistigate import lossfunctions as lf
from logistigate.logistigate import samplingplanfunctions as sampf

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
import scipy.stats as sps
import scipy.special as spsp
import scipy.optimize as spo

numTN, numSN = 2, 2
N = np.array([[1.,0.], [1.,0.]])
Y = np.array([[0., 0.], [0., 0.]])
TNnames = ['TN1', 'TN2']
SNnames = ['SN1', 'SN2']
dataTbl = [['TN1', 'SN1', 0],['TN2', 'SN1', 0]]
testdatadict = {'dataTbl':dataTbl, 'type':'Tracked', 'TNnames':TNnames, 'SNnames':SNnames}
# Set up logistigate dictionary
lgdict = util.initDataDict(N, Y)
lgdict.update({'TNnames':TNnames, 'SNnames':SNnames})

SNpriorMean = np.repeat(spsp.logit(0.1), numSN)
TNpriorMean = np.repeat(spsp.logit(0.1), numTN)
# Concatenate prior means
priorMean = np.concatenate((SNpriorMean, TNpriorMean))
TNvar, SNvar = 2., 2.  # Variances for use with prior; supply nodes are wider due to unknown risk assessments
priorCovar = np.diag(np.concatenate((np.repeat(SNvar, numSN), np.repeat(TNvar, numTN))))
priorObj = prior_normal_assort(priorMean, priorCovar)
lgdict['prior'] = priorObj

# MCMC
numdraws = 10000
lgdict['numPostSamples'] = numdraws
lgdict['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
lgdict = methods.GeneratePostSamples(lgdict, maxTime=5000)

util.plotPostSamples(lgdict, 'int90')

lgdict.update({'Q':np.array([[1.,0.],[1.,0.]])})

# Utility
paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN))

# Set MCMC draws to use in fast algorithm
numtruthdraws, numdatadraws = 10000, 1000
# Get random subsets for truth and data draws
np.random.seed(56)
truthdraws, datadraws = util.distribute_truthdata_draws(lgdict['postSamples'], numtruthdraws, numdatadraws)
paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
# Get base loss
paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)

util.print_param_checks(paramdict)


def getUtilityEstimate(n, lgdict, paramdict, zlevel=0.95):
    """
    Return a utility estimate average and confidence interval for allocation array n
    """
    testnum = int(np.sum(n))
    des = n/testnum
    currlosslist = sampf.sampling_plan_loss_list(des, testnum, lgdict, paramdict)
    currloss_avg, currloss_CI = sampf.process_loss_list(currlosslist, zlevel=zlevel)
    return paramdict['baseloss'] - currloss_avg, (paramdict['baseloss']-currloss_CI[1], paramdict['baseloss']-currloss_CI[0])

n = np.array([1,0])
util, util_CI = getUtilityEstimate(n, lgdict, paramdict)

# Iterate through pairs of samples and plot
utiltotallist, utiltotalCIlist = [0], [(0,0)]
utilsumlist, utilsumCIlist = [0], [(0,0)]
for i in range(1, 100):
    n = np.array([i,0])
    util1, util1_CI = getUtilityEstimate(n, lgdict, paramdict)
    n = np.array([0, i])
    util2, util2_CI = getUtilityEstimate(n, lgdict, paramdict)
    utilsumlist.append(util1+util2)
    utilsumCIlist.append((util1_CI[0]+util2_CI[0], util1_CI[1]+util2_CI[1]))
    # Holistic utility
    n = np.array([i, i])
    utiltotal, utiltotal_CI = getUtilityEstimate(n, lgdict, paramdict)
    utiltotallist.append(utiltotal)
    utiltotalCIlist.append(utiltotal_CI)
    # Update plotting lists
    utilsumCIlistlower = [x[0] for x in utilsumCIlist]
    utilsumCIlistupper = [x[1] for x in utilsumCIlist]
    utiltotalCIlistlower = [x[0] for x in utiltotalCIlist]
    utiltotalCIlistupper = [x[1] for x in utiltotalCIlist]
    # Plot
    plt.plot(range(0,i + 1), utilsumlist, color='blue',linewidth=3)
    plt.plot(range(0,i + 1), utiltotallist, color='black',linewidth=3)
    plt.plot(range(0,i + 1), utilsumCIlistlower, color='lightblue',linestyle='dashed')
    plt.plot(range(0,i + 1), utilsumCIlistupper, color='lightblue',linestyle='dashed')
    plt.plot(range(0,i + 1), utiltotalCIlistlower, color='gray',linestyle='dashed')
    plt.plot(range(0,i + 1), utiltotalCIlistupper, color='gray',linestyle='dashed')
    plt.legend(['$U(n_1)+U(n_2)$','$U(n_1+n_2)$'])
    plt.title('2 TNs, 1 SN')
    plt.show()

##################
##################
##################
# Now do with 5 supply nodes
numTN, numSN = 2, 7
N = np.vstack((np.ones(numSN),np.ones(numSN)))
Y = np.vstack((np.zeros(numSN),np.zeros(numSN)))
TNnames = ['TN1', 'TN2']
SNnames = ['SN'+str(i) for i in range(1,numSN+1)]
dataTbl = [ [tn, sn, 0] for tn in TNnames for sn in SNnames]
testdatadict = {'dataTbl':dataTbl, 'type':'Tracked', 'TNnames':TNnames, 'SNnames':SNnames}
# Set up logistigate dictionary
lgdict = util.initDataDict(N, Y)
lgdict.update({'TNnames':TNnames, 'SNnames':SNnames})

SNpriorMean = np.repeat(spsp.logit(0.1), numSN)
TNpriorMean = np.repeat(spsp.logit(0.1), numTN)
# Concatenate prior means
priorMean = np.concatenate((SNpriorMean, TNpriorMean))
TNvar, SNvar = 2., 2.  # Variances for use with prior; supply nodes are wider due to unknown risk assessments
priorCovar = np.diag(np.concatenate((np.repeat(SNvar, numSN), np.repeat(TNvar, numTN))))
priorObj = prior_normal_assort(priorMean, priorCovar)
lgdict['prior'] = priorObj

# MCMC
numdraws = 10000
lgdict['numPostSamples'] = numdraws
lgdict['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
lgdict = methods.GeneratePostSamples(lgdict, maxTime=5000)

util.plotPostSamples(lgdict, 'int90')

Qvec = np.ones(numSN)/numSN
lgdict.update({'Q':np.vstack((Qvec,Qvec)) })

# Utility
paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN))

# Set MCMC draws to use in fast algorithm
numtruthdraws, numdatadraws = 10000, 1000
# Get random subsets for truth and data draws
np.random.seed(56)
truthdraws, datadraws = util.distribute_truthdata_draws(lgdict['postSamples'], numtruthdraws, numdatadraws)
paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
# Get base loss
paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)

util.print_param_checks(paramdict)


n = np.array([1,0])
util, util_CI = getUtilityEstimate(n, lgdict, paramdict)

# Iterate through pairs of samples and plot
utiltotallist, utiltotalCIlist = [0], [(0,0)]
utilsumlist, utilsumCIlist = [0], [(0,0)]
for i in range(5, 101, 5):
    print('On '+str(i) +' tests...')
    n = np.array([i,0])
    util1, util1_CI = getUtilityEstimate(n, lgdict, paramdict)
    n = np.array([0, i])
    util2, util2_CI = getUtilityEstimate(n, lgdict, paramdict)
    utilsumlist.append(util1+util2)
    utilsumCIlist.append((util1_CI[0]+util2_CI[0], util1_CI[1]+util2_CI[1]))
    # Holistic utility
    n = np.array([i, i])
    utiltotal, utiltotal_CI = getUtilityEstimate(n, lgdict, paramdict)
    utiltotallist.append(utiltotal)
    utiltotalCIlist.append(utiltotal_CI)
    # Update plotting lists
    utilsumCIlistlower = [x[0] for x in utilsumCIlist]
    utilsumCIlistupper = [x[1] for x in utilsumCIlist]
    utiltotalCIlistlower = [x[0] for x in utiltotalCIlist]
    utiltotalCIlistupper = [x[1] for x in utiltotalCIlist]
    # Plot
    plt.plot(range(0,i + 1,5), utilsumlist, color='blue',linewidth=3)
    plt.plot(range(0,i + 1,5), utiltotallist, color='black',linewidth=3)
    plt.plot(range(0,i + 1,5), utilsumCIlistlower, color='lightblue',linestyle='dashed')
    plt.plot(range(0,i + 1,5), utilsumCIlistupper, color='lightblue',linestyle='dashed')
    plt.plot(range(0,i + 1,5), utiltotalCIlistlower, color='gray',linestyle='dashed')
    plt.plot(range(0,i + 1,5), utiltotalCIlistupper, color='gray',linestyle='dashed')
    plt.legend(['$U(n_1)+U(n_2)$','$U(n_1+n_2)$'])
    plt.title('2 TNs, 7 SNs')
    plt.show()

########################
########################
########################
# todo: WHY DONT' THE COMPARATIVE UTILITIES MAKE SENSE
# Put answer into interpolated functions and check that the modified objective is working as intended
tempobjval = 0
for ind, row in util_df.iterrows():
    currBound, loval, hival = row[1], row[2], row[4]
    # Get interpolation values
    retx, retf, l, k, m1, m2 = GetTriangleInterpolation([0, 1, currBound], [0, loval, hival])
    tempobjval += retf[int(n_init[ind])]
# MATCHES
# tempobjval: 2.7690379399483893

# Take hi vals of CIs and see how much the interpolations change
tempobjval = 0
for ind, row in util_df.iterrows():
    currBound, loval, loval_CI, hival, hival_CI = row[1], row[2], row[3], row[4], row[5]
    # Get interpolation values
    retx, retf, l, k, m1, m2 = GetTriangleInterpolation([0, 1, currBound], [0, loval_CI[0], hival_CI[0]])
    tempobjval += retf[int(n_init[ind])]
print(tempobjval)
'''
HI-HI: 2.943311577936248
LO-HI: 
HI-LO: 2.6811757278907646
LO-LO: 
'''

# Choose two correlated districts and check sum of utilities
# iterate through Q
currminnorm, currmaxnorm = 1e4, 0.
ind1min, ind2min = 0, 0
ind1max, ind2max = 0, 0
for i, qvec in enumerate(Qvecs):
    for j, qvec2 in enumerate(Qvecs):
        if np.linalg.norm((qvec-qvec2))<currminnorm and i != j:
            currminnorm = np.linalg.norm((qvec-qvec2))
            ind1min, ind2min = i, j
        if np.linalg.norm((qvec - qvec2)) > currmaxnorm and i != j:
            currmaxnorm = np.linalg.norm((qvec - qvec2))
            ind1max, ind2max = i, j

ntemp = np.zeros(numTN)
ntemp[ind1] = 100
ntemp[ind2] = 100
realutil, realutil_CI = getUtilityEstimate(ntemp, lgdict, paramdict)
print(realutil, realutil_CI) #0.7081823056941321 (0.6771539227322343, 0.7392106886560299)

ntemp = np.zeros(numTN)
ntemp[ind1] = 100
tildeutil1, tildutil1_CI = getUtilityEstimate(ntemp, lgdict, paramdict)
print(tildeutil1, tildutil1_CI) #0.3484752939904734 (0.3414514488824043, 0.35549913909854247)

ntemp = np.zeros(numTN)
ntemp[ind2] = 100
tildeutil2, tildutil2_CI = getUtilityEstimate(ntemp, lgdict, paramdict)
print(tildeutil2, tildutil2_CI) #0.22789067301994415 (0.22005794709540005, 0.23572339894448824)

# Equal to 81% of real combined utility

# Look at just 2 tests
ntemp = np.zeros(numTN)
ntemp[ind1] = 1
ntemp[ind2] = 1
realutil, realutil_CI = getUtilityEstimate(ntemp, lgdict, paramdict)
# realutil: 0.038604538610504946 (0.03658020869356271, 0.04062886852744718), for ind1, ind2 = 9, 16
util_df.iloc[ind1][2] + util_df.iloc[ind2][2] # 0.036228213316286784
util_df.iloc[ind1][3][0] + util_df.iloc[ind2][3][0] # 0.03333885619967347
util_df.iloc[ind1][3][1] + util_df.iloc[ind2][3][1] # 0.039117570432900095

# Look at *least* correlated districts
# indices 1 and 15
ntemp = np.zeros(numTN)
ntemp[ind1max] = 100
ntemp[ind2max] = 100
realutil, realutil_CI = getUtilityEstimate(ntemp, lgdict, paramdict)
print(realutil, realutil_CI) # 0.6592539560234485 (0.6296673529836117, 0.6888405590632853)

ntemp = np.zeros(numTN)
ntemp[ind1max] = 100
tildeutil1, tildutil1_CI = getUtilityEstimate(ntemp, lgdict, paramdict)
print(tildeutil1, tildutil1_CI) # 0.3904876268310815 (0.3826728966658255, 0.39830235699633754)

ntemp = np.zeros(numTN)
ntemp[ind2max] = 100
tildeutil2, tildutil2_CI = getUtilityEstimate(ntemp, lgdict, paramdict)
print(tildeutil2, tildutil2_CI) # 0.16721586913591047 (0.16076033432742953, 0.1736714039443914)
# Equal to 85% of real combined utility

# Iterate through a pair of samples and plot
utilcomblist, utilcombCIlist = [0], [(0,0)]
utilsumlist, utilsumCIlist = [0], [(0,0)]
for i in range(20, 101, 20):
    print('On ' + str(i) + ' tests...')
    print('Location 8...')
    n = np.zeros(numTN)
    n[8] = i
    util1, util1_CI = getUtilityEstimate(n, lgdict, paramdict)
    print('Location 9...')
    n = np.zeros(numTN)
    n[9] = i
    util2, util2_CI = getUtilityEstimate(n, lgdict, paramdict)
    utilsumlist.append(util1+util2)
    utilsumCIlist.append((util1_CI[0]+util2_CI[0], util1_CI[1]+util2_CI[1]))
    # Holistic utility
    print('Holistic...')
    n = np.zeros(numTN)
    n[8], n[9] = i, i
    utilcomb, utilcomb_CI = getUtilityEstimate(n, lgdict, paramdict)
    utilcomblist.append(utilcomb)
    utilcombCIlist.append(utilcomb_CI)
    # Update plotting lists
    utilsumCIlistlower = [x[0] for x in utilsumCIlist]
    utilsumCIlistupper = [x[1] for x in utilsumCIlist]
    utilcombCIlistlower = [x[0] for x in utilcombCIlist]
    utilcombCIlistupper = [x[1] for x in utilcombCIlist]
    # Plot
    plt.plot(range(0,i + 1,20), utilsumlist, color='darkgreen',linewidth=3)
    plt.plot(range(0,i + 1,20), utilcomblist, color='black',linewidth=3)
    plt.plot(range(0,i + 1,20), utilsumCIlistlower, color='lightgreen',linestyle='dashed')
    plt.plot(range(0,i + 1,20), utilsumCIlistupper, color='lightgreen',linestyle='dashed')
    plt.plot(range(0,i + 1,20), utilcombCIlistlower, color='gray',linestyle='dashed')
    plt.plot(range(0,i + 1,20), utilcombCIlistupper, color='gray',linestyle='dashed')
    plt.legend(['$U(n_1)+U(n_2)$','$U(n_1+n_2)$'])
    plt.title('Equal tests at Locations 8 and 9\n150k truth, 500 data')
    plt.xlabel('Number of tests at each location')
    plt.show()

# Focus *only* at 100 tests for each location; compare the bound and holistic utility under different levels of
#   truth draws
truthdrawslist = [5000, 20000, 50000, 75000, 100000, 125000, 150000]
utilcomblist, utilcombCIlist = [0 for x in range(len(truthdrawslist))], [(0,0) for x in range(len(truthdrawslist))]
utilsumlist, utilsumCIlist = [0 for x in range(len(truthdrawslist))], [(0,0) for x in range(len(truthdrawslist))]
tn1, tn2 = 9, 16
for i, currtruthdraws in enumerate(truthdrawslist):
    print('On ' + str(currtruthdraws) + ' draws...')
    # Set MCMC draws to use in fast algorithm
    numtruthdraws, numdatadraws = currtruthdraws, 300
    # Get random subsets for truth and data draws
    np.random.seed(58)
    truthdraws, datadraws = util.distribute_truthdata_draws(lgdict['postSamples'], numtruthdraws, numdatadraws)
    paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
    # Get base loss
    paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)
    util.print_param_checks(paramdict)

    # Get bounds and holistic utility
    print('Location '+str(tn1) +'...')
    n = np.zeros(numTN)
    n[tn1] = 100
    util1, util1_CI = getUtilityEstimate(n, lgdict, paramdict)
    print('Location '+str(tn2) +'...')
    n = np.zeros(numTN)
    n[tn2] = 100
    util2, util2_CI = getUtilityEstimate(n, lgdict, paramdict)
    utilsumlist[i] = util1 + util2
    utilsumCIlist[i] = (util1_CI[0] + util2_CI[0], util1_CI[1] + util2_CI[1])
    # Holistic utility
    print('Holistic...')
    n = np.zeros(numTN)
    n[tn1], n[tn2] = 100, 100
    utilcomb, utilcomb_CI = getUtilityEstimate(n, lgdict, paramdict)
    utilcomblist[i] = utilcomb
    utilcombCIlist[i] = utilcomb_CI
    # Update plotting lists
    utilsumCIlistlower = [x[0] for x in utilsumCIlist]
    utilsumCIlistupper = [x[1] for x in utilsumCIlist]
    utilcombCIlistlower = [x[0] for x in utilcombCIlist]
    utilcombCIlistupper = [x[1] for x in utilcombCIlist]
    # Plot
    plt.plot(truthdrawslist, utilsumlist, color='orange', linewidth=3)
    plt.plot(truthdrawslist, utilcomblist, color='black', linewidth=3)
    plt.plot(truthdrawslist, utilsumCIlistlower, color='bisque', linestyle='dashed')
    plt.plot(truthdrawslist, utilsumCIlistupper, color='bisque', linestyle='dashed')
    plt.plot(truthdrawslist, utilcombCIlistlower, color='gray', linestyle='dashed')
    plt.plot(truthdrawslist, utilcombCIlistupper, color='gray', linestyle='dashed')
    plt.legend(['$U(n_1)+U(n_2)$', '$U(n_1+n_2)$'])
    plt.title('Bounds and real utility vs. truth draws\n100 tests at Locations 8 and 9, 300 MCMC data draws')
    plt.xlabel('Number of truth draws')
    plt.show()


# todo: (END OF COMPARING UTILITIES)