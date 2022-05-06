# -*- coding: utf-8 -*-
'''
Script that generates and analyzes a synthetic set of PMS data. These data differ from the data used in the paper but
capture the important elements of what is presented in the paper.
Inference generation requires use of the logistigate package, available at https://logistigate.readthedocs.io/en/main/.
Running the generateSyntheticData() function generates Figures 2, 3, and 4, as well as the interval widths for Tables
1 and 2, that are analagous to the items produced using the de-identified data.
'''

from logistigate.logistigate import utilities as util # Pull from the submodule "develop" branch
from logistigate.logistigate import methods
from logistigate.logistigate import lg
import numpy as np
import scipy.special as sps


def bayesianexample():
    '''
    Use a small example to find the utility from different sampling designs.
    '''

    # Define squared loss function
    def lossfunc1(est,param):
        return np.linalg.norm(est-param,2)

    # Designate number of test and supply nodes
    numTN = 3
    numSN = 2
    s, r = 1., 1.

    # Generate a supply chain
    exampleDict = util.generateRandDataDict(numImp=numSN, numOut=numTN, diagSens=s, diagSpec=r, numSamples=40,
                                            dataType='Tracked',randSeed=24,trueRates=[0.5,0.05,0.1,0.08,0.02])
    exampleDict['diagSens'] = s # bug from older version of logistigate that doesn't affect the data
    exampleDict['diagSpec'] = r

    exampleDict = util.GetVectorForms(exampleDict)

    # Add a prior
    exampleDict['prior'] = methods.prior_normal()
    exampleDict['numPostSamples'] = 1000
    exampleDict['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    exampleDict['importerNum'] = numSN
    exampleDict['outletNum'] = numTN

    # Store the sourcing probability matrix; assume Q is known, but it could be estimated otherwise
    #Q = exampleDict['transMat']

    # Summarize the data results
    #N_init = exampleDict['N']
    #Y_init = exampleDict['Y']

    # Generate posterior draws
    exampleDict = methods.GeneratePostSamples(exampleDict)

    # Different designs
    design1 = np.array([1., 0., 0.])
    design2 = np.array([0., 1., 0.])
    design3 = np.array([0., 0., 1.])
    design4 = np.array([0.4, 0.3, 0.3])
    design5 = np.array([0., 0.5, 0.5])

    ##################################
    ########## REMOVE LATER ##########
    ##################################
    priordatadict = exampleDict.copy()
    estdecision = 'mean'
    numtests = 8
    design = design5.copy()
    lossfunc = lossfunc1

    def bayesutility(priordatadict, lossfunc, estdecision, design, numtests, omeganum):
        '''
        priordatadict: should have posterior draws from initial data set already included
        estdecision: how to form a decision from the posterior samples; one of 'mean', 'mode', or 'median'
        design: a sampling probability vector along all test nodes
        numtests: how many samples will be obtained under the design
        '''

        omeganum = 100    # UPDATE

        (numTN, numSN) = priordatadict['N'].shape
        Q = priordatadict['transMat']
        s, r = priordatadict['diagSens'], priordatadict['diagSpec']

        # Retrieve prior draws
        priordraws = priordatadict['postSamples']

        # Store utility for each omega in an array
        utilityvec = []

        for omega in range(omeganum):
            # Initialize samples to be drawn from test nodes, per the design
            TNsamps = np.round(numtests * design)
            # Grab a draw from the prior
            currpriordraw = priordraws[np.random.choice(priordraws.shape[0], size=1)[0]] # [SN rates, TN rates]
            # Initialize Ntilde and Ytilde
            Ntilde = np.zeros(shape = priordatadict['N'].shape)
            Ytilde = Ntilde.copy()

            while np.sum(TNsamps) > 0.:
                index = [i for i, x in enumerate(TNsamps) if x > 0]
                currTNind = index[0]
                TNsamps[currTNind] -= 1
                # Randomly choose the supply node, per Q
                currSNind = np.random.choice(np.array(list(range(numSN))),size=1,p=Q[currTNind])[0]
                # Generate test result
                currTNrate = currpriordraw[numSN+currTNind]
                currSNrate = currpriordraw[currSNind]
                currrealrate = currTNrate + (1-currTNrate)*currSNrate # z_star for this sample
                currposrate = s*currrealrate+(1-r)*(1-currrealrate) # z for this sample
                result = np.random.binomial(1, p=currposrate)
                Ntilde[currTNind, currSNind] += 1
                Ytilde[currTNind, currSNind] += result

            # We have a new set of data d_tilde
            Nomega = priordatadict['N'] + Ntilde
            Yomega = priordatadict['Y'] + Ytilde

            postdatadict = priordatadict.copy()
            postdatadict['N'] = Nomega
            postdatadict['Y'] = Yomega

            postdatadict = methods.GeneratePostSamples(postdatadict)
            # Get mean of samples as estimate
            currSamps = sps.logit(postdatadict['postSamples'])

            if estdecision == 'mean':
                logitmeans = np.average(currSamps,axis=0)
                currEst = sps.expit(logitmeans)

            # Average loss for all postpost samples
            avgloss = 0
            for currsamp in postdatadict['postSamples']:
                currloss = lossfunc(currEst,currsamp)
                avgloss += currloss
            avgloss = avgloss/len(postdatadict['postSamples'])

            #Append to utility storage vector
            utilityvec.append(avgloss)

        utilvalue = np.average(utilityvec)
        utilsd = np.std(utilityvec)



        return utilvalue, utilsd, utilvalue-2*utilsd,utilvalue+2*utilsd







    return


'''
uvecavg = []
for i in range(len(utilityvec)):
    uvecavg.append(np.average(utilityvec[:(i+1)]))

import matplotlib.pyplot as plt
plt.plot(utilityvec)
plt.show()

plt.plot(uvecavg)
plt.show()
'''