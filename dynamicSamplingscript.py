import numpy as np
import scipy.optimize as spo
import scipy.special as sps

# Workaround for the 'methods' file not being able to locate the 'mcmcsamplers' folder for importing
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, 'logistigate','logistigate')))

import logistigate.logistigate.utilities as util # Pull from the submodule "develop" branch
import logistigate.logistigate.methods as methods # Pull from the submodule "develop" branch
import logistigate.logistigate.lg as lg # Pull from the submodule "develop" branch

def testDynamicSamplingPolicies(numOutlets, numImporters, numSystems, numSamples,
                                batchSize, diagSens, diagSpec):
    '''
    This function/script uses randomly generated systems to test different sampling
    policies' ability to form inferences on entities in the system.
    The goal is to understand the space with respect to SFP manifestation. This goal is
    measured by the average 90% interval size of each entity's inferred SFP rate.

    For each generated supply-chain system, samples are collected according to the
    sampling policy according to a set batch size. Once each batch is collected, all
    samples collected to that point are used to generate MCMC samples. The intervals
    formed by these samples are measured, recorded, and the next batch of samples is
    collected.
    '''
    import numpy as np
    import logistigate.logistigate.samplingpolicies as policies
    numOutlets, numImporters = 100, 20
    numSystems = 2
    numSamples = 2000
    batchSize = 100
    diagSens = 1.0
    diagSpec = 1.0
    # Initialize the matrix for recording measurements
    numBatches = int(numSamples/batchSize)
    measureMat = np.zeros(shape=(numSystems,numBatches))
    for systemInd in range(numSystems): # Loop through each system
        print('Working on system ' + str(systemInd+1))
        # Generate a new system of the desired size
        currSystemDict = util.generateRandSystem(numImp=numImporters,numOut=numOutlets,randSeed=systemInd+10)
        # Collect samples according to the sampling policy and regenerate MCMC samples if
        # it's been 'batchsize' samples since the last MCMC generation

        # UNIFORM RANDOM SAMPLING
        resultsList = [] # Initialize the testing results list
        MCMCdict_NUTS = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
        for batchInd in range(numBatches):
            resultsList = policies.SampPol_Uniform(currSystemDict,testingDataList=resultsList,
                                                   numSamples=batchSize,dataType='Tracked',
                                                   sens=diagSens,spec=diagSpec)
            # Generae new MCMC samples with current resultsList
            samplesDict = {}  # Used for generating MCMC samples
            # Update with current results and items needed for MCMC sampling
            samplesDict.update({'type': 'Tracked', 'outletNames': currSystemDict['outletNames'],
                             'importerNames': currSystemDict['importerNames'],
                             'dataTbl':resultsList,'diagSens':diagSens,'diagSpec':diagSpec,
                             'prior': methods.prior_normal(),'numPostSamples':500,
                             'MCMCdict':MCMCdict_NUTS})
            samplesDict = util.GetVectorForms(samplesDict) # Add vector forms, needed for generating samples
            samplesDict = methods.GeneratePostSamples(samplesDict)
            # Once samples are generated, perform measurement of the interval widths
            tempListOfIntervalWidths = []
            for entityInd in range(numImporters+numOutlets):
                curr90IntWidth = np.quantile(samplesDict['postSamples'][:, entityInd], 0.975)-\
                                 np.quantile(samplesDict['postSamples'][:, entityInd], 0.025)
                tempListOfIntervalWidths.append(curr90IntWidth)
            measureMat[systemInd,batchInd] = np.mean(tempListOfIntervalWidths)
    # Now there are measurements for each batch and generated system
    # Next is to plot the measurements across systems
    import matplotlib.pyplot as plt
    x = np.arange(1,numBatches+1)*batchSize
    y = [np.mean(measureMat[:,i]) for i in range(numBatches)]
    yLower = [np.quantile(measureMat[:, i], 0.025) for i in range(numBatches)]
    yUpper = [np.quantile(measureMat[:, i], 0.975) for i in range(numBatches)]
    fig = plt.figure()
    plt.suptitle(
        'Average 90% interval width vs. number of samples\n100 outlets, 20 importers\nUNIFORM RANDOM SAMPLING',
        size=10)
    plt.xlabel('Number of samples', size=14)
    plt.ylabel('Average 90% interval width', size=14)
    plt.plot(x,y,'black',label='Mean')
    plt.plot(x,yLower,'b--',label='Lower 95% of systems')
    plt.plot(x,yUpper,'g--',label='Upper 95% of systems')
    plt.legend()
    plt.show()

    return