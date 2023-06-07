from logistigate.logistigate import utilities as util  # Pull from the submodule "develop" branch
from logistigate.logistigate import methods, lg
from logistigate.logistigate import lossfunctions as lf
from logistigate.logistigate import samplingplanfunctions as sampf
from logistigate.logistigate.priors import prior_normal_assort
import os
import numpy as np
from numpy.random import choice
import scipy.special as sps
import scipy.stats as spstat
import matplotlib.pyplot as plt
import random
import time
from math import comb
import matplotlib.cm as cm

def STUDYsourcingEffects():
    '''
    Study on how/if utility at one TN is affected by sourcing changes at other TNs; use case study to evaluate
    '''
    rd3_N = np.array([[1., 1., 10., 1., 3., 0., 1., 6., 7., 5., 0., 0., 4.],
                      [1., 1., 4., 2., 0., 1., 1., 2., 0., 4., 0., 0., 1.],
                      [3., 17., 31., 4., 2., 0., 1., 6., 0., 23., 1., 2., 5.],
                      [1., 1., 15., 2., 0., 0., 0., 1., 0., 6., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    rd3_Y = np.array([[0., 0., 7., 0., 3., 0., 1., 0., 1., 0., 0., 0., 4.],
                      [0., 0., 2., 2., 0., 1., 1., 0., 0., 1., 0., 0., 1.],
                      [0., 0., 15., 3., 2., 0., 0., 2., 0., 1., 1., 2., 5.],
                      [0., 0., 5., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

    (numTN, numSN) = rd3_N.shape
    s, r = 1., 1.
    CSdict3 = util.generateRandDataDict(numImp=numSN, numOut=numTN, diagSens=s, diagSpec=r,
                                        numSamples=0, dataType='Tracked', randSeed=2)
    CSdict3['diagSens'], CSdict3['diagSpec'] = s, r
    CSdict3 = util.GetVectorForms(CSdict3)
    CSdict3['N'], CSdict3['Y'] = rd3_N, rd3_Y

    SNpriorMean = np.repeat(sps.logit(0.1), numSN)
    # Establish test nodes according to assessment by regulators
    # REMOVE LATER
    # ASHANTI: Moderate; BRONG AHAFO: Moderate; CENTRAL: Moderately High; EASTERN REGION: Moderately High
    # GREATER ACCRA: Moderately High; NORTHERN SECTOR: Moderate; VOLTA: Moderately High; WESTERN: Moderate
    TNpriorMean = sps.logit(np.array([0.1, 0.1, 0.15, 0.15, 0.15, 0.1, 0.15, 0.1]))
    TNvar, SNvar = 2., 4.
    CSdict3['prior'] = prior_normal_assort(np.concatenate((SNpriorMean, TNpriorMean)),
                                           np.diag(np.concatenate((np.repeat(SNvar, numSN), np.repeat(TNvar, numTN)))))

    ##### REMOVE LATER
    # CSdict3['TNnames'] = ['ASHANTI', 'BRONG AHAFO', 'CENTRAL', 'EASTERN REGION', 'GREATER ACCRA', 'NORTHERN SECTOR', 'VOLTA', 'WESTERN']
    CSdict3['TNnames'] = ['MOD_39', 'MOD_17', 'MODHIGH_95', 'MODHIGH_26',
                          'MODHIGH_EXPL_1', 'MOD_EXPL_1', 'MODHIGH_EXPL_2', 'MOD_EXPL_2']
    CSdict3['SNnames'] = ['ACME FORMULATION PVT. LTD.', 'AS GRINDEKS', 'BELCO PHARMA', 'BHARAT PARENTERALS LTD',
                          'HUBEI TIANYAO PHARMACEUTICALS CO LTD.', 'MACIN REMEDIES INDIA LTD',
                          'NORTH CHINA PHARMACEUTICAL CO. LTD', 'NOVARTIS PHARMA', 'PFIZER',
                          'PIRAMAL HEALTHCARE UK LIMITED', 'PUSHKAR PHARMA',
                          'SHANDOND SHENGLU PHARMACEUTICAL CO.LTD.', 'SHANXI SHUGUANG PHARM']

    # Region catchment proportions
    TNcach = np.array([0.17646, 0.05752, 0.09275, 0.09488, 0.17695, 0.22799, 0.07805, 0.0954])
    tempQ = CSdict3['N'][:4] / np.sum(CSdict3['N'][:4], axis=1).reshape(4, 1)
    tempTNcach = TNcach[:4] / np.sum(TNcach[:4])
    SNcach = np.matmul(tempTNcach, tempQ)
    # Normalize market weights s.t. sum of TN terms equals sum of SN terms equals number of TNs
    # TNcach = TNcach * TNcach.shape[0] / TNcach.sum()
    # SNcach = SNcach * TNcach.sum() / SNcach.sum()
    ###################

    CSdict3['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    CSdict3['SNnum'], CSdict3['TNnum'] = numSN, numTN
    # Generate posterior draws
    numdraws = 50000  # Evaluate choice here
    CSdict3['numPostSamples'] = numdraws
    CSdict3 = methods.GeneratePostSamples(CSdict3)

    util.plotPostSamples(CSdict3, 'int90')

    # Draws for Bayes estimates and data
    # setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=10000, replace=False)]
    # np.save('bayesDraws_untestedNodes', setDraws)
    setDraws = np.load('bayesDraws_untestedNodes.npy')
    numSetDraws = 10000

    # Use single boostrap sample from observed supply nodes to establish Q for each new test node
    numBoot = 44  # Average across each TN in original data set
    SNprobs = np.sum(CSdict3['N'], axis=0) / np.sum(CSdict3['N'])
    np.random.seed(33)
    Qvecs = np.random.multinomial(numBoot, SNprobs, size=numTN - 4) / numBoot
    CSdict3['Q'] = np.vstack((CSdict3['N'][:4] / np.sum(CSdict3['N'][:4], axis=1).reshape(4, 1), Qvecs))
    '''Sourcing vectors for new regions:
    array([[0.02272727, 0.11363636, 0.31818182, 0.02272727, 0.06818182, 0.        , 0.        , 0.15909091, 0.04545455, 0.18181818, 0.02272727, 0.        , 0.04545455],
       [0.        , 0.09090909, 0.29545455, 0.13636364, 0.06818182, 0.        , 0.        , 0.06818182, 0.04545455, 0.25      , 0.        , 0.        , 0.04545455],
       [0.        , 0.04545455, 0.5       , 0.        , 0.02272727, 0.        , 0.06818182, 0.11363636, 0.02272727, 0.18181818, 0.        , 0.        , 0.04545455],
       [0.02272727, 0.06818182, 0.31818182, 0.09090909, 0.02272727, 0.        , 0.02272727, 0.02272727, 0.04545455, 0.27272727, 0.02272727, 0.        , 0.09090909]])
    '''
    '''
    Q1 = CSdict3['Q'][4:]
    np.random.seed(52)
    Qvecs = np.random.multinomial(numBoot, SNprobs, size=numTN - 4) / numBoot
    CSdict3['Q'] = np.vstack((CSdict3['N'][:4] / np.sum(CSdict3['N'][:4], axis=1).reshape(4, 1), Qvecs))
    Q2 = CSdict3['Q'][4:]
    tempSum = 0
    for i in range(4):
        print(np.linalg.norm(Q1[i]-Q2[i]))
        tempSum += np.linalg.norm(Q1[i]-Q2[i])
    print(tempSum)
    # 36: 0.909; 52: 0.823
    '''

    # Loss specification
    paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                                  marketvec=np.ones(numTN + numSN), candneighnum=1000)

    # Set limits of data collection and intervals for calculation
    testMax, testInt = 400, 10
    numtargetdraws = 5100

    numDataDraws = 5000
    utilDict = {'method': 'weightsNodeDraw3linear'}
    utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})

    # To avoid seed issues, use a list of Q matrices
    numQ = 5
    Qlist = []
    for Qind in range(numQ):
        np.random.seed(Qind + 33)
        Qvecs = np.random.multinomial(numBoot, SNprobs, size=numTN - 4) / numBoot
        Qlist.append(np.vstack((CSdict3['N'][:4] / np.sum(CSdict3['N'][:4], axis=1).reshape(4, 1), Qvecs)))
    np.random.seed(33)

    # Minimize variance by averaging over multiple runs
    numReps = 10
    sampBudget = 60
    tnMat = np.empty((4 * numQ, numReps))
    for rep in range(numReps):
        print('Replication ' + str(rep) + '...')
        # Get new MCMC draws
        # CSdict3 = methods.GeneratePostSamples(CSdict3)
        # Withdraw a subset of MCMC prior draws
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New loss draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        # Get a new set of data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        # Get neighbors
        print('Generating loss matrix...')
        tempLossMat = lf.lossMatSetBayesDraws(dictTemp['postSamples'], lossDict.copy(), lossDict['bayesDraws'])
        tempLossDict = lossDict.copy()
        tempLossDict.update({'lossMat': tempLossMat})
        newBayesDraws, newLossMat = lf.add_cand_neighbors(tempLossDict.copy(), CSdict3['postSamples'],
                                                          dictTemp['postSamples'])
        tempLossDict.update({'bayesDraws': newBayesDraws, 'lossMat': newLossMat})
        baseLoss = (np.sum(newLossMat, axis=1) / newLossMat.shape[1]).min()
        print('Base loss: ' + str(baseLoss))
        for Qind in range(numQ):
            print('Sourcing matrix ' + str(Qind) + '...')
            dictTemp['Q'] = Qlist[Qind]
            # print(dictTemp['Q'])
            # Calculate utility for each node at sampBudget
            for tnInd in range(4):
                print('Calculating TN ' + str(tnInd) + '...')
                desArr = np.zeros(8)
                desArr[tnInd] = 1.
                currUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[desArr], numtests=sampBudget,
                                                               utildict=utilDict)[0]
                tnMat[(tnInd * numQ) + Qind, rep] = currUtil
        # Create boxplot
        plt.boxplot(tnMat[:, :rep + 1].T)
        plt.title('Boxplot of utiliies at tested TNs for different sourcing at untested TNs\nSample budget=60')
        xtickstr = ['TN ' + str(i) + '\n$Q_' + str(j + 1) + '$' for i in range(4) for j in range(numQ)]
        plt.xticks(np.arange(4 * numQ), xtickstr, fontsize=6)
        plt.show()
        plt.close()
    '''14-APR run
    tnMat = 
    '''
    '''    
    utilMatList = [u1, u2, u3, u4, u5]
    '''
    avgUtilMat = np.average(np.array(utilMatList), axis=0)
    plotMargUtil(avgUtilMat, testMax, testInt, labels=dictTemp['TNnames'], type='delta',
                 titleStr='Untested Nodes, $t=0.15$, $m=0.6$', lineLabels=True,  # utilMax=0.2,
                 colors=cm.rainbow(np.linspace(0, 1., numTN)),
                 dashes=[[1, 0] for tn in range(4)] + [[1, 1] for tn in range(4)])
    allocArr, objValArr = smoothAllocationForward(avgUtilMat)
    plotAlloc(allocArr, paramList=[str(i) for i in np.arange(testInt, testMax + 1, testInt)], testInt=testInt,
              labels=dictTemp['TNnames'], titleStr='Untested Nodes, $t=0.15$, $m=0.6$',  # allocMax=250,
              colors=cm.rainbow(np.linspace(0, 1., numTN)),
              dashes=[[1, 0] for tn in range(4)] + [[1, 1] for tn in range(4)])

    allocArr = np.array([])

    return