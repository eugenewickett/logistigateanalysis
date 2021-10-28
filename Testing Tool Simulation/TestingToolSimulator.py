# -*- coding: utf-8 -*-
'''
Script that generates a dictionary of scenarios to be used with DiagnosticTool.

Each scenario is a dictionary of diagnostics and SFP environments.

'''
from importlib import reload
import numpy as np
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, '../logistigate')))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, '../logistigate', 'logistigate')))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, '../logistigate', 'logistigate', 'mcmcsamplers')))

import utilities as util # Pull from the submodule "develop" branch
import methods # Pull from the submodule "develop" branch
import lg # Pull from the submodule "develop" branch
reload(methods)
reload(util)
import statistics
import random
import scipy.stats as sps
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
import pickle

# Generate lists for each parameter used in the simulator
lst_dataType = ['Tracked'] # 'Tracked' or 'Untracked'
lst_numTN = [50] # Number of test nodes
lst_rho = [1.] # Ratio of supply nodes to test nodes; num. SNs = rho*num. of TNs; [0.5,1.0,2.0]
lst_credInt_alpha = [0.9] # Alpha level for credible intervals
lst_u = [0.05] # Exoneration threshold
lst_t = [0.3] # Suspect threshold
'''
SFPDist_1: SFP rates of [0.01, 0.1, 0.25, 0.5, 0.75] w probabilities [0.5, 0.2, 0.15, 0.1, 0.05]
SFPBeta_1-5: SFP rates generated from beta(1,5)
'''
lst_trueSFPratesSet = ['SFPDist_1'] # ['SFPBeta_1-5'}
lst_lamb = [1.] # Pareto scale parameter characterizing the sourcing probability matrix; [0.8, 1.0, 1.2]
lst_zeta = [4] # Number of non-zero entries of the sourcing probability matrix, in multiples of numTN; [5,4,3,2]
lst_priorMean = [-5] # Prior mean
lst_priorVar = [5] # Prior variance
lst_numSamples = [3000] # How many samples to test per iteration

m = 500 # How frequently (in samples) to recalculate the MCMC samples
N = 50 # Number of systems to generate

# Generate list of testing tools, which are lists of [name, sensitivity, specificity]
lst_TT = [['HiDiag',1.0,1.0],  ['LoDiag',0.6,0.9]] # ['MdDiag',0.8,0.95],

currSysTime = time.time() # For tracking how long it takes to run a batch of simulations
# Generate dictionary of scenarios to run
scenList = {}
scenNum = 0
for dataType in lst_dataType:
    for numTN in lst_numTN:
        for rho in lst_rho:
            for alpha in lst_credInt_alpha:
                for u in lst_u:
                    for t in lst_t:
                        for trueSFPratesSet in lst_trueSFPratesSet:
                            for lamb in lst_lamb:
                                for zeta in lst_zeta:
                                    for priorMean in lst_priorMean:
                                        for priorVar in lst_priorVar:
                                            for numSamples in lst_numSamples:
                                                scenList[scenNum] = {'dataType':dataType,'numTN':numTN, 'rho':rho,
                                                                     'alpha':alpha, 'u':u, 't':t,
                                                                     'trueSFPratesSet':trueSFPratesSet, 'lamb':lamb,
                                                                     'zeta':zeta, 'priorMean':priorMean,
                                                                     'priorVar':priorVar, 'numSamples':numSamples}
                                                scenNum += 1
                                                #scenList.append([numTN,rho,alpha,u,t,trueSFPratesSet,lamb,zeta,priorMean,priorVar,numSamples])

#####################################


# Run iterations for each testing tool and collect data, per the entered scenarios
# Store each iteration output row in a list (output functions should be able to process data in this way), one row for each metric
outputDict = {}
outputRow = 0 # For iterating the outputDict rows
# Set random seeds
'''
randSeed = 3
if randSeed >= 0:
    random.seed(randSeed + 2)
    np.random.seed(randSeed)
'''
# Set MCMC dict specs
MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 2000, 'delta': 0.4}

for scenInd in scenList: # scenario loop
    scen = scenList[scenInd]
    for simIter in range(N): # system iteration loop
        # todo: REPLACE WITH SINGLE FUNCTION CALL LATER, WITH THE FOLLOWING INPUTS: scen, m
        # Generate a new system
        curr_numTN = scen['numTN'] # number of test nodes
        curr_numSN = int(scen['rho']*curr_numTN) # number of supply nodes

        # Generate true SFP rates
        if scen['trueSFPratesSet'] == 'SFPDist_1':
            # [0.01, 0.1, 0.25, 0.5, 0.75] w probabilities [0.5, 0.2, 0.15, 0.1, 0.05]
            curr_trueRates_TN = random.choices([0.01, 0.1, 0.25, 0.5, 0.75],weights=[0.5, 0.2, 0.15, 0.1, 0.05],
                                               k=curr_numTN)
            curr_trueRates_SN = random.choices([0.01, 0.1, 0.25, 0.5, 0.75],weights=[0.5, 0.2, 0.15, 0.1, 0.05],
                                               k=curr_numSN)
        elif scen['trueSFPratesSet'] == 'SFPBeta_1-5':
            # beta(1,5) distibution for all node SFP rates; ~17% suspect, ~23% exonerated
            curr_trueRates_TN = sps.beta.rvs(1,5,size=curr_numTN).tolist()
            curr_trueRates_SN = sps.beta.rvs(1,5,size=curr_numSN).tolist()
        else:
            print('Enter a valid argument for the true SFP rates.')
            break
        # How many suspects and exonerated do we have?
        suspNum_TN = len([i for i in curr_trueRates_TN if i>=scen['t']])
        suspNum_SN = len([i for i in curr_trueRates_SN if i >= scen['t']])
        exonNum_TN = len([i for i in curr_trueRates_TN if i <= scen['u']])
        exonNum_SN = len([i for i in curr_trueRates_SN if i <= scen['u']])
        # Generate a sourcing matrix, Q
        # todo: Make function in logistigate utilities for generating a random Q using size, number of pos. entries, and distribution
        Q = np.zeros(shape=(curr_numTN, curr_numSN))
        # First decide the positive entries of Q; need at least positive entry for each row so pick those first
        for TNind in range(curr_numTN):
            randSNind = random.choice(range(curr_numSN))
            Q[TNind, randSNind] = 1.0
        # Next choose (zeta-1)*curr_numTN entries randomly and set to 1
        numEntries = int(np.floor((scen['zeta']-1)*curr_numTN))
        if numEntries > (curr_numTN*curr_numSN)-curr_numTN: # zeta is too big
            print('zeta argument is too large, choose another.')
            break
        for entryInd in range(numEntries):
            entryBool = False
            while entryBool == False:
                # Choose a row with zeros
                entryRowInd = random.choice(range(curr_numTN))
                if np.sum(Q[entryRowInd]) < curr_numSN:
                    entryBool = True
            TNcandidates = np.arange(curr_numTN)
            QrowZeroInds = [i for i in range(curr_numSN) if (Q[entryRowInd,i]==0)] # Eligible indices
            Q[entryRowInd, random.choice(QrowZeroInds)] = 1.0
        # Move entries so that every supply node has at least one test node it provides for
        for SNind in range(curr_numSN):
            if np.sum(Q[:, SNind]) == 0.0:
                # Find eligible SNs for swapping
                QcolNonZeroInds = [i for i in range(curr_numSN) if (np.sum(Q[:,i])>1.0)]
                # Choose random eligible column
                randColInd = random.choice(QcolNonZeroInds)
                QrowPosInds = [i for i in range(curr_numTN) if Q[i,randColInd]==1.0]
                randRowInd = random.choice(QrowPosInds)
                # Swap
                Q[randRowInd, randColInd] = 0.
                Q[randRowInd, SNind] = 1.
        # We now have a Q with a 1.0 in every row and column, with zeta*curr_numTN total positive entries
        # Now make each row of Q Pareto-distributed, characterized by the lamb parameter
        for TNind in range(curr_numTN):
            paretoRow = np.array([random.paretovariate(scen['lamb']) for i in range(curr_numSN)])
            newRow = [paretoRow[i]*Q[TNind][i] for i in range(curr_numSN)]
            rowSum = np.sum(newRow)
            newRow = [newRow[i]/rowSum for i in range(curr_numSN)] # Normalize
            Q[TNind] = newRow
        # Now generate data for each testing tool
        TTdataArr = [[] for TT in lst_TT] # Initialize data list for each testing tool
        metricsArr = [[[[],[]], [[],[]], [[],[]], [[],[]], [[],[]], [[],[]]] for TT in lst_TT] # Initialize metrics list
            # For each testing tool, we have TN and SN lists for metrics of basic interval scoring, Gneiting loss,
            #       suspect Type I error, suspect Type II error, exoneration Type I error, exoneration Type II error
        for currSamp in range(scen['numSamples']):
            currTN = random.choice(range(curr_numTN))
            currSN = random.choices(range(curr_numSN), weights=Q[currTN], k=1)[0]
            consolRate = curr_trueRates_TN[currTN] + (1 - curr_trueRates_TN[currTN]) * curr_trueRates_SN[currSN]
            realResult = np.random.binomial(1, p=consolRate)
            if realResult == 1:
                for TTind, currTT in enumerate(lst_TT): # What happens when we use different testing tools
                    testResult = np.random.binomial(1,p=currTT[1])
                    if scen['dataType'] == 'Tracked':
                        TTdataArr[TTind].append([currTN, currSN, testResult])
                    elif scen['dataType'] == 'Untracked':
                        TTdataArr[TTind].append([currTN, testResult])
                    else:
                        print('dataType misspecified, try again.')
                        break
            if realResult == 0:
                for TTind, currTT in enumerate(lst_TT):  # What happens when we use different testing tools
                    testResult = np.random.binomial(1, p=1.-currTT[2])
                    if scen['dataType'] == 'Tracked':
                        TTdataArr[TTind].append([currTN, currSN, testResult])
                    elif scen['dataType'] == 'Untracked':
                        TTdataArr[TTind].append([currTN, testResult])
                    else:
                        print('dataType misspecified, try again.')
                        break
            # If we have reached a recalculation point, we do that here
            if np.mod(currSamp+1,m)==0 and currSamp>0:
                print('sample number ' + str(currSamp) + ' for iteration ' + str(simIter))
                TNnames = np.arange(curr_numTN).tolist()
                SNnames = np.arange(curr_numSN).tolist()
                # Do for each testing tool
                for TTind, currTT in enumerate(lst_TT):
                    currDataDict = {'type':scen['dataType'], 'dataTbl':TTdataArr[TTind],
                                    'outletNames':TNnames, 'importerNames':SNnames}
                    currDataDict = util.GetVectorForms(currDataDict)
                    currDataDict.update({'diagSens':currTT[1]-1e-3, 'diagSpec':currTT[2]-1e-3,
                                         'MCMCdict':MCMCdict,
                                         'prior':methods.prior_normal(scen['priorMean'],scen['priorVar']),
                                         'numPostSamples':500})
                    # Generate MCMC samples
                    currDataDict = methods.GeneratePostSamples(currDataDict)
                    print(currDataDict['postSamplesGenTime'])
                    # Evaluate MCMC samples vs. true SFP rates
                    curr_trueRates_TN, curr_trueRates_SN
                    MCMCsamples = currDataDict['postSamples']

                    basicInt_TN, basicInt_SN = 0, 0 # Basic interval scoring
                    gnLoss_TN, gnLoss_SN = 0, 0  # Gneiting loss
                    suspErr1_TN, suspErr1_SN = 0, 0 # Suspect Type I error
                    suspErr2_TN, suspErr2_SN  = 0, 0 # Suspect Type II error
                    exonErr1_TN, exonErr1_SN = 0, 0 # Exonerate Type I error
                    exonErr2_TN, exonErr2_SN = 0, 0 # Exonerate Type II error
                    for TNind in range(curr_numTN): # Iterate through test nodes
                        currMCMCInt = [np.quantile(MCMCsamples[:, curr_numSN + TNind], (1-scen['alpha'])/2),
                                       np.quantile(MCMCsamples[:, curr_numSN + TNind], 1-((1-scen['alpha'])/2))]
                        # Evaluate different metrics
                        currTR = curr_trueRates_TN[TNind]
                        if currTR >= currMCMCInt[0] and currTR <= currMCMCInt[1]: # Interval contains true SFP rate
                            basicInt_TN += 1
                            gnLoss_TN += (currMCMCInt[1] - currMCMCInt[0])
                        else: # Interval does not contain true SFP rate
                            gnLoss_TN += (currMCMCInt[1] - currMCMCInt[0]) + (2 / (1-scen['alpha'])) * \
                                         min(np.abs(currTR - currMCMCInt[1]), np.abs(currTR - currMCMCInt[0]))
                        # Suspect classifiction
                        if currTR < scen['t']: # Node is below suspect threshold
                            if currMCMCInt[0] >= scen['t']: # Type I error
                                suspErr1_TN += 1
                        else: # Node is above suspect threshold
                            if currMCMCInt[0] < scen['t']: # Type II error
                                suspErr2_TN += 1
                        # Exoneration classification
                        if currTR > scen['u']: # Node is above exoneration threshold
                            if currMCMCInt[1] <= scen['u']: # Type I error
                                exonErr1_TN += 1
                        else: # Node is below exoneration threshold
                            if currMCMCInt[1] > scen['u']: # Type II error
                                exonErr2_TN += 1

                    for SNind in range(curr_numSN): # Iterate through supply nodes
                        currMCMCInt = [np.quantile(MCMCsamples[:, SNind], (1-scen['alpha'])/2),
                                       np.quantile(MCMCsamples[:, SNind], 1-((1-scen['alpha'])/2))]
                        # Evaluate different metrics
                        currTR = curr_trueRates_SN[SNind]
                        if currTR >= currMCMCInt[0] and currTR <= currMCMCInt[1]: # Interval contains true SFP rate
                            basicInt_SN += 1
                            gnLoss_SN += (currMCMCInt[1] - currMCMCInt[0])
                        else: # Interval does not contain true SFP rate
                            gnLoss_SN += (currMCMCInt[1] - currMCMCInt[0]) + (2 / (1-scen['alpha'])) * \
                                         min(np.abs(currTR - currMCMCInt[1]), np.abs(currTR - currMCMCInt[0]))
                        # Suspect classifiction
                        if currTR < scen['t']: # Node is below suspect threshold
                            if currMCMCInt[0] >= scen['t']: # Type I error
                                suspErr1_SN += 1
                        else: # Node is above suspect threshold
                            if currMCMCInt[0] < scen['t']: # Type II error
                                suspErr2_SN += 1
                        # Exoneration classification
                        if currTR > scen['u']: # Node is above exoneration threshold
                            if currMCMCInt[1] <= scen['u']: # Type I error
                                exonErr1_SN += 1
                        else: # Node is below exoneration threshold
                            if currMCMCInt[1] > scen['u']: # Type II error
                                exonErr2_SN += 1
                    # Update the metrics array; test nodes then supply nodes within each metric
                    metricsArr[TTind][0][0].append(basicInt_TN / curr_numTN) # Basic interval scoring
                    metricsArr[TTind][0][1].append(basicInt_SN / curr_numSN)
                    metricsArr[TTind][1][0].append(gnLoss_TN) # Gneiting loss
                    metricsArr[TTind][1][1].append(gnLoss_SN)
                    if suspNum_TN > 0: # Suspect Type I and II error
                        metricsArr[TTind][2][0].append(suspErr1_TN / suspNum_TN)
                        metricsArr[TTind][3][0].append(suspErr2_TN / suspNum_TN)
                    if suspNum_SN > 0:
                        metricsArr[TTind][2][1].append(suspErr1_SN / suspNum_SN)
                        metricsArr[TTind][3][1].append(suspErr2_SN / suspNum_SN)
                    if exonNum_TN > 0: # Exoneration Type I and II error
                        metricsArr[TTind][4][0].append(exonErr1_TN / exonNum_TN)
                        metricsArr[TTind][5][0].append(exonErr2_TN / exonNum_TN)
                    if exonNum_SN > 0:
                        metricsArr[TTind][4][1].append(exonErr1_SN / exonNum_SN)
                        metricsArr[TTind][5][1].append(exonErr2_SN / exonNum_SN)

                # END recalculation if statement
            # END data collection loop

        # STORE LINES OF OUTPUT FOR THIS ITERATION
        rowDict = {'lst_TT': lst_TT, 'scen':scen, 'simIter':simIter, 'Q':Q, 'trueRates_TN':curr_trueRates_TN,
                   'trueRates_SN':curr_trueRates_SN, 'metricsArr':metricsArr}
        outputDict[outputRow] = rowDict
        outputRow += 1
        # END system iteration loop
    # END scenario loop
# Write the outputDict to a file
outputFilePath  = os.getcwd() + '\\output dictionaries'
if not os.path.exists(outputFilePath): # Generate this folder if one does not already exist
    os.makedirs(outputFilePath)
    #outputFileName = os.path.basename(sys.argv[0])[:-3] + '_OUTPUT' # Current file name
outputFileName = os.path.join(outputFilePath, 'OP_'+str(time.time())[:10])
pickle.dump(outputDict, open(outputFileName,'wb'))

# Total run time
batchRunTime = time.time() - currSysTime
print('Batch took ' + str(batchRunTime)[:4] + ' seconds')




### OUTPUT ANALYSIS HERE
# How to read back a stored dictionary
openFileName = os.path.join(os.getcwd() + '\\output dictionaries', 'OP_1635437686') # Change to desired output file
openFile = open(openFileName,'rb') # Read the file
openDict = pickle.load(openFile)

print(openDict.keys())
print(openDict[0]['scen'])
print(openDict[51]['scen'])
print(openDict[144]['scen'])


lst_OPdicts = []
for f in os.listdir(os.getcwd()+'\\output dictionaries'):
    openFileName = os.path.join(os.getcwd() + '\\output dictionaries', f)  # Change to desired output file
    openFile = open(openFileName, 'rb')  # Read the file
    openDict = pickle.load(openFile)
    lst_OPdicts.append(openDict.copy())




scen1 = {'dataType': 'Tracked', 'numTN': 50, 'rho': 2.0, 'alpha': 0.9, 'u': 0.05, 't': 0.3,
         'trueSFPratesSet': 'SFPDist_1', 'lamb': 0.8, 'zeta': 5, 'priorMean': -5, 'priorVar': 5, 'numSamples': 3000}
scen2 = {'dataType': 'Tracked', 'numTN': 50, 'rho': 2.0, 'alpha': 0.9, 'u': 0.05, 't': 0.3,
         'trueSFPratesSet': 'SFPDist_1', 'lamb': 1.2, 'zeta': 5, 'priorMean': -5, 'priorVar': 5, 'numSamples': 3000}
lst_scens = [scen1]#[scen1, scen2]
lst_TTs = [['HiDiag', 1.0, 1.0], ['LoDiag', 0.6, 0.9]]
subtitle='(1,1) tool vs. (0.6,0.9) tool'


# TEST FOR OUTPUT READER
scen1 = {'dataType': 'Tracked', 'numTN': 50, 'rho': 2.0, 'alpha': 0.9, 'u': 0.05, 't': 0.3,
         'trueSFPratesSet': 'SFPDist_1', 'lamb': 0.8, 'zeta': 5, 'priorMean': -5, 'priorVar': 5, 'numSamples': 3000}
lst_TTs = [['HiDiag', 1.0, 1.0], ['LoDiag', 0.6, 0.9]]
name = '(1,1) tool vs. (0.6,0.9) tool'

GenerateChart(lst_OPdicts, lst_scens, lst_TTs, subtitle=name)

# OUTPUT READER
def GenerateChart(lst_OPdicts,lst_scens, lst_TTs, subtitle=''):
    '''
    This function finds all simulation output satisfying the characteristics entered and produces charts for each
    metric.

    INPUTS:
        - lst_OPdicts:      LIST of output dictionaries with the required keys
        - lst_scens:        LIST of scenario dictionaries to match with the 'scen' key of the replications in the output dictionaries
        - lst_TTs:          LIST of testing tools, with each testing tool given as [name, sensitivity, specificity]
        - subtitle:         STRING for desired subtitle on charts
    OUTPUTS:
        12 charts, one for each metric-node class combination, with 95% confidence intervals on the metrics as found
        in the replications stored in the respective output dictionaries
    '''
    tempTToutputList = [[[[],[]], [[],[]], [[],[]], [[],[]], [[],[]], [[],[]]] for TT in lst_TTs]
    scenOutputList = [tempTToutputList for scen in lst_scens]

    for currOPdict in lst_OPdicts: # Loop through each output dictionary
        for iterInd in currOPdict.keys():
            currOutput = currOPdict[iterInd]
            # Check that 'scen' matches one in the list
            if currOutput['scen'] in lst_scens:
                currScenInd = lst_scens.index(currOutput['scen'])
            else:
                currScenInd = -1
            # Check that the testing tools are in the list
            currTTinds = []
            for TTind, TTval in enumerate(lst_TTs):
                foundTT = False
                for opTTind, opTT in enumerate(currOutput['lst_TT']):
                    if opTT == TTval:
                        currTTinds.append(opTTind)
                        foundTT = True
                if foundTT == False:
                    currTTinds.append(-1)
            # We now know where our metrics should go.
            currOPmetrics = currOutput['metricsArr']
            if currScenInd >= 0:
                for lstTTind, TTind in enumerate(currTTinds):
                    numSamples = currOutput['scen']['numSamples'] # Need this for plotting
                    if TTind >= 0:
                        currTTmetrics = currOPmetrics[TTind] # We have a set testing tool metrics for this scenario
                        for metricInd, metric in enumerate(currTTmetrics):
                            scenOutputList[currScenInd][lstTTind][metricInd][0].append(metric[0]) # Test node metric
                            scenOutputList[currScenInd][lstTTind][metricInd][1].append(metric[1]) # Supply node metric

    # scenOutputList now has all the information we need
    # Now generate lists for producing charts
    plotVec_TN_intScore = [[[[], []] for TT in lst_TTs] for scen in lst_scens]
    plotVec_SN_intScore = [[[[], []] for TT in lst_TTs] for scen in lst_scens]
    plotVec_TN_gneitLoss = [[[[], []] for TT in lst_TTs] for scen in lst_scens]
    plotVec_SN_gneitLoss = [[[[], []] for TT in lst_TTs] for scen in lst_scens]
    plotVec_TN_suspErr1 = [[[[], []] for TT in lst_TTs] for scen in lst_scens]
    plotVec_SN_suspErr1 = [[[[], []] for TT in lst_TTs] for scen in lst_scens]
    plotVec_TN_suspErr2 = [[[[], []] for TT in lst_TTs] for scen in lst_scens]
    plotVec_SN_suspErr2 = [[[[], []] for TT in lst_TTs] for scen in lst_scens]
    plotVec_TN_exonErr1 = [[[[], []] for TT in lst_TTs] for scen in lst_scens]
    plotVec_SN_exonErr1 = [[[[], []] for TT in lst_TTs] for scen in lst_scens]
    plotVec_TN_exonErr2 = [[[[], []] for TT in lst_TTs] for scen in lst_scens]
    plotVec_SN_exonErr2 = [[[[], []] for TT in lst_TTs] for scen in lst_scens]

    num_m_steps = len(scenOutputList[0][0][0][0][0])
    for scenInd, scen in enumerate(lst_scens):
        for TTind, TT in enumerate(lst_TTs):

            # Make n/m lists so we can form the 95% CIs
            tempList_Met1_TN = [[] for i in range(num_m_steps)]
            tempList_Met1_SN = [[] for i in range(num_m_steps)]
            tempList_Met2_TN = [[] for i in range(num_m_steps)]
            tempList_Met2_SN = [[] for i in range(num_m_steps)]
            tempList_Met3_TN = [[] for i in range(num_m_steps)]
            tempList_Met3_SN = [[] for i in range(num_m_steps)]
            tempList_Met4_TN = [[] for i in range(num_m_steps)]
            tempList_Met4_SN = [[] for i in range(num_m_steps)]
            tempList_Met5_TN = [[] for i in range(num_m_steps)]
            tempList_Met5_SN = [[] for i in range(num_m_steps)]
            tempList_Met6_TN = [[] for i in range(num_m_steps)]
            tempList_Met6_SN = [[] for i in range(num_m_steps)]

            for item in scenOutputList[scenInd][TTind][0][0]: #1st metric, test nodes
                for j in range(num_m_steps):
                    tempList_Met1_TN[j].append(item[j])
            for item in scenOutputList[scenInd][TTind][0][1]: #1st metric, supply nodes
                for j in range(num_m_steps):
                    tempList_Met1_SN[j].append(item[j])
            for item in scenOutputList[scenInd][TTind][1][0]: #2nd metric, test nodes
                for j in range(num_m_steps):
                    tempList_Met2_TN[j].append(item[j])
            for item in scenOutputList[scenInd][TTind][1][1]: #2nd metric, supply nodes
                for j in range(num_m_steps):
                    tempList_Met2_SN[j].append(item[j])
            for item in scenOutputList[scenInd][TTind][2][0]: #3rd metric, test nodes
                for j in range(num_m_steps):
                    tempList_Met3_TN[j].append(item[j])
            for item in scenOutputList[scenInd][TTind][2][1]: #3rd metric, supply nodes
                for j in range(num_m_steps):
                    tempList_Met3_SN[j].append(item[j])
            for item in scenOutputList[scenInd][TTind][3][0]: #4th metric, test nodes
                for j in range(num_m_steps):
                    tempList_Met4_TN[j].append(item[j])
            for item in scenOutputList[scenInd][TTind][3][1]: #4th metric, supply nodes
                for j in range(num_m_steps):
                    tempList_Met4_SN[j].append(item[j])
            for item in scenOutputList[scenInd][TTind][4][0]: #5th metric, test nodes
                for j in range(num_m_steps):
                    tempList_Met5_TN[j].append(item[j])
            for item in scenOutputList[scenInd][TTind][4][1]: #5th metric, supply nodes
                for j in range(num_m_steps):
                    tempList_Met5_SN[j].append(item[j])
            for item in scenOutputList[scenInd][TTind][5][0]: #6th metric, test nodes
                for j in range(num_m_steps):
                    tempList_Met6_TN[j].append(item[j])
            for item in scenOutputList[scenInd][TTind][5][1]: #6th metric, supply nodes
                for j in range(num_m_steps):
                    tempList_Met6_SN[j].append(item[j])
            # Put 95% lower and upper bounds in plot vectors
            plotVec_TN_intScore[scenInd][TTind][0] = [(np.quantile(tempList_Met1_TN[j],0.05)) for j in
                                                      range(num_m_steps)] # Lower first
            plotVec_TN_intScore[scenInd][TTind][1] = [(np.quantile(tempList_Met1_TN[j], 0.95)) for j in
                                                      range(num_m_steps)] # Upper next
            plotVec_SN_intScore[scenInd][TTind][0] = [(np.quantile(tempList_Met1_SN[j], 0.05)) for j in
                                                      range(num_m_steps)]
            plotVec_SN_intScore[scenInd][TTind][1] = [(np.quantile(tempList_Met1_SN[j], 0.95)) for j in
                                                      range(num_m_steps)]
            plotVec_TN_gneitLoss[scenInd][TTind][0] = [(np.quantile(tempList_Met2_TN[j], 0.05)) for j in
                                                      range(num_m_steps)]
            plotVec_TN_gneitLoss[scenInd][TTind][1] = [(np.quantile(tempList_Met2_TN[j], 0.95)) for j in
                                                      range(num_m_steps)]
            plotVec_SN_gneitLoss[scenInd][TTind][0] = [(np.quantile(tempList_Met2_SN[j], 0.05)) for j in
                                                       range(num_m_steps)]
            plotVec_SN_gneitLoss[scenInd][TTind][1] = [(np.quantile(tempList_Met2_SN[j], 0.95)) for j in
                                                       range(num_m_steps)]
            plotVec_TN_suspErr1[scenInd][TTind][0] = [(np.quantile(tempList_Met3_TN[j], 0.05)) for j in
                                                       range(num_m_steps)]
            plotVec_TN_suspErr1[scenInd][TTind][1] = [(np.quantile(tempList_Met3_TN[j], 0.95)) for j in
                                                       range(num_m_steps)]
            plotVec_SN_suspErr1[scenInd][TTind][0] = [(np.quantile(tempList_Met3_SN[j], 0.05)) for j in
                                                       range(num_m_steps)]
            plotVec_SN_suspErr1[scenInd][TTind][1] = [(np.quantile(tempList_Met3_SN[j], 0.95)) for j in
                                                       range(num_m_steps)]
            plotVec_TN_suspErr2[scenInd][TTind][0] = [(np.quantile(tempList_Met4_TN[j], 0.05)) for j in
                                                      range(num_m_steps)]
            plotVec_TN_suspErr2[scenInd][TTind][1] = [(np.quantile(tempList_Met4_TN[j], 0.95)) for j in
                                                      range(num_m_steps)]
            plotVec_SN_suspErr2[scenInd][TTind][0] = [(np.quantile(tempList_Met4_SN[j], 0.05)) for j in
                                                      range(num_m_steps)]
            plotVec_SN_suspErr2[scenInd][TTind][1] = [(np.quantile(tempList_Met4_SN[j], 0.95)) for j in
                                                      range(num_m_steps)]
            plotVec_TN_exonErr1[scenInd][TTind][0] = [(np.quantile(tempList_Met5_TN[j], 0.05)) for j in
                                                      range(num_m_steps)]
            plotVec_TN_exonErr1[scenInd][TTind][1] = [(np.quantile(tempList_Met5_TN[j], 0.95)) for j in
                                                      range(num_m_steps)]
            plotVec_SN_exonErr1[scenInd][TTind][0] = [(np.quantile(tempList_Met5_SN[j], 0.05)) for j in
                                                      range(num_m_steps)]
            plotVec_SN_exonErr1[scenInd][TTind][1] = [(np.quantile(tempList_Met5_SN[j], 0.95)) for j in
                                                      range(num_m_steps)]
            plotVec_TN_exonErr2[scenInd][TTind][0] = [(np.quantile(tempList_Met6_TN[j], 0.05)) for j in
                                                      range(num_m_steps)]
            plotVec_TN_exonErr2[scenInd][TTind][1] = [(np.quantile(tempList_Met6_TN[j], 0.95)) for j in
                                                      range(num_m_steps)]
            plotVec_SN_exonErr2[scenInd][TTind][0] = [(np.quantile(tempList_Met6_SN[j], 0.05)) for j in
                                                      range(num_m_steps)]
            plotVec_SN_exonErr2[scenInd][TTind][1] = [(np.quantile(tempList_Met6_SN[j], 0.95)) for j in
                                                      range(num_m_steps)]

    # Now plot
    import matplotlib.pyplot as plt
    import matplotlib.colors as col
    xTickLabels = np.arange(numSamples/num_m_steps,numSamples+numSamples/num_m_steps,numSamples/num_m_steps).tolist()
    colorInd = np.linspace(0,1,len(lst_scens)*len(lst_TTs))

    fig, ax = plt.subplots(3, 2, sharex='col')
    fig.text(0.29, 0.01, 'Number of samples', ha='center')
    fig.text(0.77, 0.01, 'Number of samples', ha='center')
    # First metric, test nodes
    for scenInd in range(len(lst_scens)):
        for TTind in range(len(lst_TTs)):
            currInts = plotVec_TN_intScore[scenInd][TTind]
            currCol = plt.cm.RdYlBu(colorInd[TTind+scenInd*len(lst_TTs)])

            ax[0, 0].plot(xTickLabels,currInts[0], color=currCol,label='SCEN'+str(scenInd+1) + ', TT'+str(TTind+1))
            ax[0, 0].plot(xTickLabels,currInts[1], color=currCol)
            ax[0, 0].set_ylim(0.5, 1.1)
            ax[0, 0].title.set_text('Interval Score')
    ax[0, 0].legend(loc='lower left',fontsize=6)
    # Second metric, test nodes
    for scenInd in range(len(lst_scens)):
        for TTind in range(len(lst_TTs)):
            currInts = plotVec_TN_gneitLoss[scenInd][TTind]
            currCol = plt.cm.RdYlBu(colorInd[TTind+scenInd*len(lst_TTs)])
            ax[0, 1].plot(xTickLabels,currInts[0], color=currCol,label='SCEN'+str(scenInd+1)+', TT'+str(TTind+1))
            ax[0, 1].plot(xTickLabels,currInts[1], color=currCol)
            #ax[0, 1].set_ylim(0, 1.)
            ax[0, 1].title.set_text('Gneiting Loss')
    ax[0, 1].legend(loc='upper right',fontsize=6)
    # Third metric, test nodes
    for scenInd in range(len(lst_scens)):
        for TTind in range(len(lst_TTs)):
            currInts = plotVec_TN_suspErr1[scenInd][TTind]
            currCol = plt.cm.RdYlBu(colorInd[TTind + scenInd * len(lst_TTs)])
            ax[1, 0].plot(xTickLabels, currInts[0], color=currCol, label='SCEN' + str(scenInd+1) + ', TT' + str(TTind+1))
            ax[1, 0].plot(xTickLabels, currInts[1], color=currCol)
            ax[1, 0].set_ylim(0, 1.1)
            ax[1, 0].title.set_text('Susp. Type I Err.')
    ax[1, 0].legend(loc='upper left', fontsize=6)
    # Fourth metric, test nodes
    for scenInd in range(len(lst_scens)):
        for TTind in range(len(lst_TTs)):
            currInts = plotVec_TN_suspErr2[scenInd][TTind]
            currCol = plt.cm.RdYlBu(colorInd[TTind + scenInd * len(lst_TTs)])
            ax[1, 1].plot(xTickLabels, currInts[0], color=currCol,
                          label='SCEN' + str(scenInd + 1) + ', TT' + str(TTind + 1))
            ax[1, 1].plot(xTickLabels, currInts[1], color=currCol)
            ax[1, 1].set_ylim(0, 1.1)
            ax[1, 1].title.set_text('Susp. Type II Err.')
    ax[1, 1].legend(loc='lower left', fontsize=6)
    # Fifth metric, test nodes
    for scenInd in range(len(lst_scens)):
        for TTind in range(len(lst_TTs)):
            currInts = plotVec_TN_exonErr1[scenInd][TTind]
            currCol = plt.cm.RdYlBu(colorInd[TTind + scenInd * len(lst_TTs)])
            ax[2, 0].plot(xTickLabels, currInts[0], color=currCol,
                          label='SCEN' + str(scenInd + 1) + ', TT' + str(TTind + 1))
            ax[2, 0].plot(xTickLabels, currInts[1], color=currCol)
            ax[2, 0].set_ylim(0, 1.1)
            ax[2, 0].title.set_text('Exon. Type I Err.')
    ax[2, 0].legend(loc='upper left', fontsize=6)
    # Sixth metric, test nodes
    for scenInd in range(len(lst_scens)):
        for TTind in range(len(lst_TTs)):
            currInts = plotVec_TN_exonErr2[scenInd][TTind]
            currCol = plt.cm.RdYlBu(colorInd[TTind + scenInd * len(lst_TTs)])
            ax[2, 1].plot(xTickLabels, currInts[0], color=currCol,
                          label='SCEN' + str(scenInd + 1) + ', TT' + str(TTind + 1))
            ax[2, 1].plot(xTickLabels, currInts[1], color=currCol)
            ax[2, 1].set_ylim(0, 1.1)
            ax[2, 1].title.set_text('Exon. Type II Err.')
    ax[2, 1].legend(loc='lower left', fontsize=6)

    plt.suptitle('95% CIs on Metrics for TEST NODES\n'+subtitle)
    plt.tight_layout()
    #plt.figure(figsize=(8, 10))
    plt.show()

    # Now supply nodes
    fig, ax = plt.subplots(3, 2, sharex='col')
    fig.text(0.29, 0.01, 'Number of samples', ha='center')
    fig.text(0.77, 0.01, 'Number of samples', ha='center')
    # First metric, test nodes
    for scenInd in range(len(lst_scens)):
        for TTind in range(len(lst_TTs)):
            currInts = plotVec_SN_intScore[scenInd][TTind]
            currCol = plt.cm.RdYlBu(colorInd[TTind + scenInd * len(lst_TTs)])

            ax[0, 0].plot(xTickLabels, currInts[0], color=currCol,
                          label='SCEN' + str(scenInd + 1) + ', TT' + str(TTind + 1))
            ax[0, 0].plot(xTickLabels, currInts[1], color=currCol)
            ax[0, 0].set_ylim(0.5, 1.1)
            ax[0, 0].title.set_text('Interval Score')
    ax[0, 0].legend(loc='lower left', fontsize=6)
    # Second metric, test nodes
    for scenInd in range(len(lst_scens)):
        for TTind in range(len(lst_TTs)):
            currInts = plotVec_SN_gneitLoss[scenInd][TTind]
            currCol = plt.cm.RdYlBu(colorInd[TTind + scenInd * len(lst_TTs)])
            ax[0, 1].plot(xTickLabels, currInts[0], color=currCol,
                          label='SCEN' + str(scenInd + 1) + ', TT' + str(TTind + 1))
            ax[0, 1].plot(xTickLabels, currInts[1], color=currCol)
            # ax[0, 1].set_ylim(0, 1.)
            ax[0, 1].title.set_text('Gneiting Loss')
    ax[0, 1].legend(loc='upper right', fontsize=6)
    # Third metric, test nodes
    for scenInd in range(len(lst_scens)):
        for TTind in range(len(lst_TTs)):
            currInts = plotVec_SN_suspErr1[scenInd][TTind]
            currCol = plt.cm.RdYlBu(colorInd[TTind + scenInd * len(lst_TTs)])
            ax[1, 0].plot(xTickLabels, currInts[0], color=currCol,
                          label='SCEN' + str(scenInd + 1) + ', TT' + str(TTind + 1))
            ax[1, 0].plot(xTickLabels, currInts[1], color=currCol)
            ax[1, 0].set_ylim(0, 1.1)
            ax[1, 0].title.set_text('Susp. Type I Err.')
    ax[1, 0].legend(loc='upper left', fontsize=6)
    # Fourth metric, test nodes
    for scenInd in range(len(lst_scens)):
        for TTind in range(len(lst_TTs)):
            currInts = plotVec_SN_suspErr2[scenInd][TTind]
            currCol = plt.cm.RdYlBu(colorInd[TTind + scenInd * len(lst_TTs)])
            ax[1, 1].plot(xTickLabels, currInts[0], color=currCol,
                          label='SCEN' + str(scenInd + 1) + ', TT' + str(TTind + 1))
            ax[1, 1].plot(xTickLabels, currInts[1], color=currCol)
            ax[1, 1].set_ylim(0, 1.1)
            ax[1, 1].title.set_text('Susp. Type II Err.')
    ax[1, 1].legend(loc='lower left', fontsize=6)
    # Fifth metric, test nodes
    for scenInd in range(len(lst_scens)):
        for TTind in range(len(lst_TTs)):
            currInts = plotVec_SN_exonErr1[scenInd][TTind]
            currCol = plt.cm.RdYlBu(colorInd[TTind + scenInd * len(lst_TTs)])
            ax[2, 0].plot(xTickLabels, currInts[0], color=currCol,
                          label='SCEN' + str(scenInd + 1) + ', TT' + str(TTind + 1))
            ax[2, 0].plot(xTickLabels, currInts[1], color=currCol)
            ax[2, 0].set_ylim(0, 1.1)
            ax[2, 0].title.set_text('Exon. Type I Err.')
    ax[2, 0].legend(loc='upper left', fontsize=6)
    # Sixth metric, test nodes
    for scenInd in range(len(lst_scens)):
        for TTind in range(len(lst_TTs)):
            currInts = plotVec_SN_exonErr2[scenInd][TTind]
            currCol = plt.cm.RdYlBu(colorInd[TTind + scenInd * len(lst_TTs)])
            ax[2, 1].plot(xTickLabels, currInts[0], color=currCol,
                          label='SCEN' + str(scenInd + 1) + ', TT' + str(TTind + 1))
            ax[2, 1].plot(xTickLabels, currInts[1], color=currCol)
            ax[2, 1].set_ylim(0, 1.1)
            ax[2, 1].title.set_text('Exon. Type II Err.')
    ax[2, 1].legend(loc='lower left', fontsize=6)

    plt.suptitle('95% CIs on Metrics for SUPPLY NODES\n' + subtitle)
    plt.tight_layout()
    # plt.figure(figsize=(8, 10))
    plt.show()

    return















