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

# Generate lists for each parameter used in the simulator
lst_numTN = [50] # Number of test nodes
lst_rho = [0.5,1.0,2.0] # Ratio of supply nodes to test nodes; num. SNs = rho*num. of TNs
lst_credInt_alpha = [0.9] # Alpha level for credible intervals
lst_u = [0.05] # Exoneration threshold
lst_t = [0.3] # Suspect threshold
'''
SFPDist_1: SFP rates of [0.01, 0.1, 0.25, 0.5, 0.75] w probabilities [0.5, 0.2, 0.15, 0.1, 0.05]
SFPBeta_1-5: SFP rates generated from beta(1,5)
'''
lst_trueSFPratesSet = ['SFPDist_1', 'SFPBeta_1-5']
lst_lamb = [0.8, 1.0, 1.2] # Pareto scale parameter characterizing the sourcing probability matrix
lst_zeta = [2, 3, 4, 5] # Number of non-zero entries of the sourcing probability matrix, in multiples of numTN
lst_priorMean = [-5] # Prior mean
lst_priorVar = [5] # Prior variance
lst_numSamples = [50*100] # How many samples to test per iteration

m = 500 # How frequently to recalculate the MCMC samples
N = 1000 # Number of systems to generate

# Generate list of testing tools, which are lists of [name, sensitivity, specificity]
lst_TT = [['HiDiag',1.0,1.0], ['MdDiag',0.8,0.95], ['LoDiag',0.6,0.9]]

# Generate dictionary of scenarios to run
scenList = {}
scenNum = 0
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
                                            scenList[scenNum] = {'numTN':numTN, 'rho':rho, 'alpha':alpha, 'u':u, 't':t,
                                                 'trueSFPratesSet':trueSFPratesSet, 'lamb':lamb, 'zeta':zeta,
                                                 'priorMean':priorMean, 'priorVar':priorVar, 'numSamples':numSamples}
                                            scenNum += 1
                                            #scenList.append([numTN,rho,alpha,u,t,trueSFPratesSet,lamb,zeta,priorMean,priorVar,numSamples])

#####################################


# Run iterations for each testing tool and collect data, per the entered scenarios
# Store each iteration output row in a list (output functions should be able to process data in this way), one row for each metric
outputDict = {}
for scen in scenList: # scenario loop
    for iter in range(N): # system iteration loop
        # REPLACE WITH SINGLE FUNCTION CALL LATER, WITH THE FOLLOWING INPUTS:
        #   scen

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
        # Generate a sourcing matrix, Q
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
        
        print(Q)

        '''
        for outInd in range(numOut):
            rowRands = [random.paretovariate(transMatLambda) for i in range(numImp)]
            if numImp > 10:  # Only keep 10 randomly chosen importers, if numImp > 10
                rowRands[10:] = [0.0 for i in range(numImp - 10)]
                random.shuffle(rowRands)

            normalizedRands = [rowRands[i] / sum(rowRands) for i in range(numImp)]
            # only keep transition probabilities above 2%
            # normalizedRands = [normalizedRands[i] if normalizedRands[i]>0.02 else 0.0 for i in range(numImp)]

            # normalizedRands = [normalizedRands[i] / sum(normalizedRands) for i in range(numImp)]
            transMat[outInd, :] = normalizedRands
        '''
        for TT in lst_TT: # testing tool loop
            curr_TTname = TT[0]
            curr_sens = TT[1]
            curr_spec = TT[2]




            # END testing tool loop
        # END system iteration loop
    # END scenario loop



