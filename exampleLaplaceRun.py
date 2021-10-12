import numpy as np
import scipy.optimize as spo
import scipy.special as sps

# Workaround for the 'methods' file not being able to locate the 'mcmcsamplers' folder for importing
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, 'logistigate','logistigate')))

import utilities as util # Pull from the submodule "develop" branch
import methods as methods # Pull from the submodule "develop" branch

def examiningLaplaceApprox():
    '''
    This script constitutes a detailed breakdown of what's happening in the Laplace
    approximation process. The goal is to understand why negative Hessian diagonal
    values are so common.
    '''

    # First generate a random system using a fixed seed
    newSysDict = util.generateRandDataDict(numImp=3, numOut=10, numSamples=50 * 20,
                                           diagSens=0.9, diagSpec=0.99,
                                           dataType='Tracked', randSeed=5)
    _ = util.GetVectorForms(newSysDict)
    newSysDict.update({'prior': methods.prior_normal(var=3)}) # Set prior variance here
    #import inspect
    #lines = inspect.getsource(methods.prior_normal)
    #print(lines)

    # Form Laplace approximation estimates
    outDict = methods.FormEstimates(newSysDict, retOptStatus=True)
    print(np.diag(outDict['hess'])) # Negative diagonals present
    print(np.diag(outDict['hessinv']))
    soln = np.append(outDict['impEst'],outDict['outEst'])
    soln_trans = sps.logit(soln) # Transformed solution
    # Check Jacobian + Hessian at this solution point
    soln_jac = methods.Tracked_LogPost_Grad(soln_trans,newSysDict['N'], newSysDict['Y'],
                                            newSysDict['diagSens'], newSysDict['diagSpec'],
                                            prior=newSysDict['prior'])
    soln_hess = methods.Tracked_LogPost_Hess(soln_trans, newSysDict['N'], newSysDict['Y'],
                                            newSysDict['diagSens'], newSysDict['diagSpec'],
                                            prior=newSysDict['prior'])
    print(soln_jac) # Gradient seems within tolerance of 0
    print(np.diag(soln_hess))

    # Check 2nd-order derivatives at this point
    (nOut, nImp) = newSysDict['N'].shape
    # Use a non-default prior
    # prior = methods.prior_normal(mu=1, var=2)
    # Grab the likelihood and gradient at beta0
    dL0 = methods.Tracked_LogPost_Grad(soln_trans, newSysDict['N'], newSysDict['Y'],
                                            newSysDict['diagSens'], newSysDict['diagSpec'],
                                            prior=newSysDict['prior'])
    ddL0 = methods.Tracked_LogPost_Hess(soln_trans, newSysDict['N'], newSysDict['Y'],
                                            newSysDict['diagSens'], newSysDict['diagSpec'],
                                            prior=newSysDict['prior'])
    print(np.diag(ddL0))
    # Move in every direction and flag if the difference from the gradient is more than epsilon
    for k in range(nImp + nOut):
        beta1 = 1 * soln_trans[:]
        beta1[k] = beta1[k] + 10 ** (-5)
        dL1 = methods.Tracked_LogPost_Grad(beta1, newSysDict['N'], newSysDict['Y'],
                                            newSysDict['diagSens'], newSysDict['diagSpec'],
                                            prior=newSysDict['prior'])
        print((dL1 - dL0) * (10 ** (5)))
        print(ddL0[k])
        #print((dL1 - dL0) * (10 ** (5)) - ddL0[k])
        print(np.linalg.norm((dL1 - dL0) * (10 ** (5)) - ddL0[k]))

    # Do it line by line here (from methods.FormEstimates)
    N, Y = newSysDict['N'], newSysDict['Y']
    Sens, Spec = newSysDict['diagSens'], newSysDict['diagSpec']
    prior = newSysDict['prior']
    (numOut, numImp) = N.shape

    beta0_List = []
    for sampNum in range(10):  # Generate 10 random samples via the prior
        beta0_List.append(prior.rand(numImp + numOut))

    # Loop through each possible initial point and store the optimal solution likelihood values
    likelihoodsList = []
    solsList = []
    OptStatusList = []
    bds = spo.Bounds(np.zeros(numImp + numOut) - 8, np.zeros(numImp + numOut) + 8)
    for curr_beta0 in beta0_List:
        opVal = spo.minimize(methods.Tracked_NegLogPost, curr_beta0,
                             args=(N, Y, Sens, Spec, prior), method='L-BFGS-B',
                             jac=methods.Tracked_NegLogPost_Grad,
                             options={'disp': False}, bounds=bds)
        likelihoodsList.append(opVal.fun)
        solsList.append(opVal.x)
        OptStatusList.append(opVal.status) # 0 means convergence; alternatively, use opVal.message
    print(likelihoodsList) # Check that our restarts are giving similar solutions
    best_x = solsList[np.argmin(likelihoodsList)]
    jac = methods.Tracked_LogPost_Grad(best_x, N, Y, Sens, Spec, prior)
    hess = methods.Tracked_NegLogPost_Hess(best_x, N, Y, Sens, Spec, prior)
    print(jac)
    print(np.diag(hess))
    print(np.diag(soln_hess))

    # Check 2nd-order derivatives at this point
    (nOut, nImp) = N.shape
    # Use a non-default prior
    #prior = methods.prior_normal(mu=1, var=2)
    # Grab the likelihood and gradient at beta0
    dL0 = methods.Tracked_LogPost_Grad(best_x, N, Y, Sens, Spec, prior)
    ddL0 = methods.Tracked_LogPost_Hess(best_x, N, Y, Sens, Spec, prior)
    print(np.diag(ddL0))
    # Move in every direction and flag if the difference from the gradient is more than epsilon
    for k in range(nImp + nOut):
        beta1 = 1 * best_x[:]
        beta1[k] = beta1[k] + 10 ** (-5)
        dL1 = methods.Tracked_LogPost_Grad(beta1, N, Y, Sens, Spec, prior)
        print((dL1 - dL0) * (10 ** (5)) - ddL0[k])
        print(np.linalg.norm((dL1 - dL0) * (10 ** (5)) - ddL0[k]))

    return

def laplaceTests():
    '''
    A script for checking the Laplace approximation in producing credible intervals.
    '''
    # Check L-BFGS exit flags for some random systems
    import numpy as np
    for randSys in range(10):
        newSysDict = util.generateRandDataDict(numImp=10, numOut=100, numSamples=100 * 20,
                                                 dataType='Tracked')
        _ = util.GetVectorForms(newSysDict)
        newSysDict.update({'prior': methods.prior_normal()})
        outDict = methods.FormEstimates(newSysDict, retOptStatus=True)
        print(np.sum(outDict['optStatus']))
    # Ran for 100 systems of size 10/100; no instance of a non-successful optimizer exit

    # Check the generated Hessian diagonals WRT the prior variance; try for 3 different system sizes
    priorVarList = [1,5,9]
    numSystems = 100

    resultsMat_5_20 = np.zeros((len(priorVarList), numSystems)) # for proportion of Hessian diagonals that are negative
    avgdevsMat_5_20 = np.zeros((len(priorVarList), numSystems)) # for SIZE of negative diagonals
    percOutMat_5_20 = np.zeros((len(priorVarList), numSystems)) # for proportion of negative diagonals that are outlets
    percNegEigMat_5_20 = np.zeros((len(priorVarList), numSystems)) # for proportion negative eigenvalues
    for currVarInd, currVar in enumerate(priorVarList):
        print('Working on variance of ' + str(currVar) + '...')
        for randSysInd in range(numSystems): # Systems of size 5, 20
            newSysDict = util.generateRandDataDict(numImp=5, numOut=20, numSamples=20 * 20,
                                                   dataType='Tracked')
            totalEnts = len(newSysDict['importerNames']) + len(newSysDict['outletNames'])
            _ = util.GetVectorForms(newSysDict)
            newSysDict.update({'prior': methods.prior_normal(var=currVar)})
            outDict = methods.FormEstimates(newSysDict, retOptStatus=True, printUpdate=False)
            currHessDiags = np.diag(outDict['hess'])
            negDiags = [i for i in currHessDiags if i<0]
            resultsMat_5_20[currVarInd, randSysInd] = len(negDiags)/totalEnts
            avgdevsMat_5_20[currVarInd, randSysInd] = np.average(negDiags)
            percOutMat_5_20[currVarInd, randSysInd] = len([i for i in currHessDiags[5:] if i<0])/len(negDiags)
            percNegEigMat_5_20[currVarInd, randSysInd] = len([i for i in np.linalg.eigvals(outDict['hess']) if i<0])/totalEnts

    resultsMat_10_40 = np.zeros((len(priorVarList), numSystems))
    avgdevsMat_10_40 = np.zeros((len(priorVarList), numSystems))
    percOutMat_10_40 = np.zeros((len(priorVarList), numSystems))
    percNegEigMat_10_40 = np.zeros((len(priorVarList), numSystems))
    for currVarInd, currVar in enumerate(priorVarList):
        print('Working on variance of ' + str(currVar) + '...')
        for randSysInd in range(numSystems):  # Systems of size 5, 20
            newSysDict = util.generateRandDataDict(numImp=10, numOut=40, numSamples=40 * 20,
                                                   dataType='Tracked')
            totalEnts = len(newSysDict['importerNames']) + len(newSysDict['outletNames'])
            _ = util.GetVectorForms(newSysDict)
            newSysDict.update({'prior': methods.prior_normal(var=currVar)})
            outDict = methods.FormEstimates(newSysDict, retOptStatus=True, printUpdate=False)
            currHessDiags = np.diag(outDict['hess'])
            negDiags = [i for i in currHessDiags if i < 0]
            resultsMat_10_40[currVarInd, randSysInd] = len(negDiags) / totalEnts
            avgdevsMat_10_40[currVarInd, randSysInd] = np.average(negDiags)
            percOutMat_10_40[currVarInd, randSysInd] = len([i for i in currHessDiags[10:] if i < 0]) / len(negDiags)
            percNegEigMat_10_40[currVarInd, randSysInd] = len([i for i in np.linalg.eigvals(outDict['hess']) if i < 0]) / totalEnts

    resultsMat_15_60 = np.zeros((len(priorVarList), numSystems))
    avgdevsMat_15_60 = np.zeros((len(priorVarList), numSystems))
    percOutMat_15_60 = np.zeros((len(priorVarList), numSystems))
    percNegEigMat_15_60 = np.zeros((len(priorVarList), numSystems))
    for currVarInd, currVar in enumerate(priorVarList):
        print('Working on variance of ' + str(currVar) + '...')
        for randSysInd in range(numSystems):  # Systems of size 5, 20
            newSysDict = util.generateRandDataDict(numImp=15, numOut=60, numSamples=60 * 20,
                                                   dataType='Tracked')
            totalEnts = len(newSysDict['importerNames']) + len(newSysDict['outletNames'])
            _ = util.GetVectorForms(newSysDict)
            newSysDict.update({'prior': methods.prior_normal(var=currVar)})
            outDict = methods.FormEstimates(newSysDict, retOptStatus=True, printUpdate=False)
            currHessDiags = np.diag(outDict['hess'])
            negDiags = [i for i in currHessDiags if i < 0]
            resultsMat_15_60[currVarInd, randSysInd] = len(negDiags) / totalEnts
            avgdevsMat_15_60[currVarInd, randSysInd] = np.average(negDiags)
            percOutMat_15_60[currVarInd, randSysInd] = len([i for i in currHessDiags[15:] if i < 0]) / len(negDiags)
            percNegEigMat_15_60[currVarInd, randSysInd] = len([i for i in np.linalg.eigvals(outDict['hess']) if i < 0]) / totalEnts

    resultsMat_15_100 = np.zeros((len(priorVarList), numSystems))
    avgdevsMat_15_100 = np.zeros((len(priorVarList), numSystems))
    percOutMat_15_100 = np.zeros((len(priorVarList), numSystems))
    percNegEigMat_15_100 = np.zeros((len(priorVarList), numSystems))
    for currVarInd, currVar in enumerate(priorVarList):
        print('Working on variance of ' + str(currVar) + '...')
        for randSysInd in range(numSystems):  # Systems of size 5, 20
            newSysDict = util.generateRandDataDict(numImp=15, numOut=100, numSamples=100 * 20,
                                                   dataType='Tracked')
            totalEnts = len(newSysDict['importerNames']) + len(newSysDict['outletNames'])
            _ = util.GetVectorForms(newSysDict)
            newSysDict.update({'prior': methods.prior_normal(var=currVar)})
            outDict = methods.FormEstimates(newSysDict, retOptStatus=True, printUpdate=False)
            currHessDiags = np.diag(outDict['hess'])
            negDiags = [i for i in currHessDiags if i < 0]
            resultsMat_15_100[currVarInd, randSysInd] = len(negDiags) / totalEnts
            avgdevsMat_15_100[currVarInd, randSysInd] = np.average(negDiags)
            percOutMat_15_100[currVarInd, randSysInd] = len([i for i in currHessDiags[15:] if i < 0]) / len(negDiags)
            percNegEigMat_15_100[currVarInd, randSysInd] = len([i for i in np.linalg.eigvals(outDict['hess']) if i < 0]) / totalEnts

    resultsSummaryMat = np.zeros((len(priorVarList), 8))
    for currVarInd, currVar in enumerate(priorVarList):
        resultsSummaryMat[currVarInd, 0] = np.quantile(resultsMat_5_20[currVarInd, :], 0.05)
        resultsSummaryMat[currVarInd, 1] = np.quantile(resultsMat_5_20[currVarInd, :], 0.95)
        resultsSummaryMat[currVarInd, 2] = np.quantile(resultsMat_10_40[currVarInd, :], 0.05)
        resultsSummaryMat[currVarInd, 3] = np.quantile(resultsMat_10_40[currVarInd, :], 0.95)
        resultsSummaryMat[currVarInd, 4] = np.quantile(resultsMat_15_60[currVarInd, :], 0.05)
        resultsSummaryMat[currVarInd, 5] = np.quantile(resultsMat_15_60[currVarInd, :], 0.95)
        resultsSummaryMat[currVarInd, 6] = np.quantile(resultsMat_15_100[currVarInd, :], 0.05)
        resultsSummaryMat[currVarInd, 7] = np.quantile(resultsMat_15_100[currVarInd, :], 0.95)
    avgdevsSummaryMat = np.zeros((len(priorVarList), 8))
    for currVarInd, currVar in enumerate(priorVarList):
        avgdevsSummaryMat[currVarInd, 0] = np.quantile(avgdevsMat_5_20[currVarInd, :], 0.05)
        avgdevsSummaryMat[currVarInd, 1] = np.quantile(avgdevsMat_5_20[currVarInd, :], 0.95)
        avgdevsSummaryMat[currVarInd, 2] = np.quantile(avgdevsMat_10_40[currVarInd, :], 0.05)
        avgdevsSummaryMat[currVarInd, 3] = np.quantile(avgdevsMat_10_40[currVarInd, :], 0.95)
        avgdevsSummaryMat[currVarInd, 4] = np.quantile(avgdevsMat_15_60[currVarInd, :], 0.05)
        avgdevsSummaryMat[currVarInd, 5] = np.quantile(avgdevsMat_15_60[currVarInd, :], 0.95)
        avgdevsSummaryMat[currVarInd, 6] = np.quantile(avgdevsMat_15_100[currVarInd, :], 0.05)
        avgdevsSummaryMat[currVarInd, 7] = np.quantile(avgdevsMat_15_100[currVarInd, :], 0.95)
    percOutSummaryMat = np.zeros((len(priorVarList), 8))
    for currVarInd, currVar in enumerate(priorVarList):
        percOutSummaryMat[currVarInd, 0] = np.quantile(percOutMat_5_20[currVarInd, :], 0.05)
        percOutSummaryMat[currVarInd, 1] = np.quantile(percOutMat_5_20[currVarInd, :], 0.95)
        percOutSummaryMat[currVarInd, 2] = np.quantile(percOutMat_10_40[currVarInd, :], 0.05)
        percOutSummaryMat[currVarInd, 3] = np.quantile(percOutMat_10_40[currVarInd, :], 0.95)
        percOutSummaryMat[currVarInd, 4] = np.quantile(percOutMat_15_60[currVarInd, :], 0.05)
        percOutSummaryMat[currVarInd, 5] = np.quantile(percOutMat_15_60[currVarInd, :], 0.95)
        percOutSummaryMat[currVarInd, 6] = np.quantile(percOutMat_15_100[currVarInd, :], 0.05)
        percOutSummaryMat[currVarInd, 7] = np.quantile(percOutMat_15_100[currVarInd, :], 0.95)
    percNegEigSummaryMat = np.zeros((len(priorVarList), 8))
    for currVarInd, currVar in enumerate(priorVarList):
        percNegEigSummaryMat[currVarInd, 0] = np.quantile(percNegEigMat_5_20[currVarInd, :], 0.05)
        percNegEigSummaryMat[currVarInd, 1] = np.quantile(percNegEigMat_5_20[currVarInd, :], 0.95)
        percNegEigSummaryMat[currVarInd, 2] = np.quantile(percNegEigMat_10_40[currVarInd, :], 0.05)
        percNegEigSummaryMat[currVarInd, 3] = np.quantile(percNegEigMat_10_40[currVarInd, :], 0.95)
        percNegEigSummaryMat[currVarInd, 4] = np.quantile(percNegEigMat_15_60[currVarInd, :], 0.05)
        percNegEigSummaryMat[currVarInd, 5] = np.quantile(percNegEigMat_15_60[currVarInd, :], 0.95)
        percNegEigSummaryMat[currVarInd, 6] = np.quantile(percNegEigMat_15_100[currVarInd, :], 0.05)
        percNegEigSummaryMat[currVarInd, 7] = np.quantile(percNegEigMat_15_100[currVarInd, :], 0.95)


    import matplotlib.pyplot as plt
    #from matplotlib.lines import Line2D

    #zippedList1 = zip(resultsSummaryMat[:, 0], resultsSummaryMat[:, 1], priorVarList)
    #zippedList2 = zip(resultsSummaryMat[:, 2], resultsSummaryMat[:, 3], priorVarList)
    #zippedList3 = zip(resultsSummaryMat[:, 4], resultsSummaryMat[:, 5], priorVarList)

    #custom_lines = [Line2D([0], [0], color='orange', lw=4),
    #                Line2D([0], [0], color='red', lw=4),
    #                Line2D([0], [0], color='purple', lw=4)]
    fig, ax = plt.subplots(figsize=(8, 10), ncols=1)
    #for lower, upper, name in zippedList1:
    #    plt.plot((name, name), (lower, upper), 'o-', color='orange')
    #for lower, upper, name in zippedList2:
    #    plt.plot((name, name), (lower, upper), 'o-', color='red')
    #for lower, upper, name in zippedList3:
    #    plt.plot((name, name), (lower, upper), 'o-', color='purple')
    plt.plot(priorVarList, resultsSummaryMat[:, 0], 'o--', color='orange', label='5imp_20out - lower 90%')
    plt.plot(priorVarList, resultsSummaryMat[:, 1], 'o-', color='orange', label='5imp_20out - upper 90%')
    plt.plot(priorVarList, resultsSummaryMat[:, 2], 'o--', color='red', label='10imp_40out - lower 90%')
    plt.plot(priorVarList, resultsSummaryMat[:, 3], 'o-', color='red', label='10imp_40out - upper 90%')
    plt.plot(priorVarList, resultsSummaryMat[:, 4], 'o--', color='purple', label='15imp_60out - lower 90%')
    plt.plot(priorVarList, resultsSummaryMat[:, 5], 'o-', color='purple', label='15imp_60out - upper 90%')
    plt.plot(priorVarList, resultsSummaryMat[:, 6], 'o--', color='blue', label='15imp_100out - lower 90%')
    plt.plot(priorVarList, resultsSummaryMat[:, 7], 'o-', color='blue', label='15imp_100out - upper 90%')
    plt.ylim([0, 1])
    plt.title('90% Intervals on PERCENTAGE of Neg. Hessian Diagonal Values\nvs. Prior Variance, for 4 different system sizes',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Prior variance', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    #ax.legend(custom_lines, ['5 importers, 20 outlets', '10 importers, 40 outlets', '15 importers, 60 outlets'])
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close()

    # Size of deviations below 0
    fig, ax = plt.subplots(figsize=(8, 10), ncols=1)
    plt.plot(priorVarList, avgdevsSummaryMat[:, 0], 'o--', color='orange', label='5imp_20out - lower 90%')
    plt.plot(priorVarList, avgdevsSummaryMat[:, 1], 'o-', color='orange', label='5imp_20out - upper 90%')
    plt.plot(priorVarList, avgdevsSummaryMat[:, 2], 'o--', color='red', label='10imp_40out - lower 90%')
    plt.plot(priorVarList, avgdevsSummaryMat[:, 3], 'o-', color='red', label='10imp_40out - upper 90%')
    plt.plot(priorVarList, avgdevsSummaryMat[:, 4], 'o--', color='purple', label='15imp_60out - lower 90%')
    plt.plot(priorVarList, avgdevsSummaryMat[:, 5], 'o-', color='purple', label='15imp_60out - upper 90%')
    plt.plot(priorVarList, avgdevsSummaryMat[:, 6], 'o--', color='blue', label='15imp_100out - lower 90%')
    plt.plot(priorVarList, avgdevsSummaryMat[:, 7], 'o-', color='blue', label='15imp_100out - upper 90%')
    plt.ylim([-3, 0])
    plt.title('90% Intervals on SIZE of Neg. Hessian Diagonal Values\nvs. Prior Variance, for 4 different system sizes',
        fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Prior variance', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close()

    # Percentage of negative diagonals that are outlets
    fig, ax = plt.subplots(figsize=(8, 10), ncols=1)
    plt.plot(priorVarList, percOutSummaryMat[:, 0], 'o--', color='orange', label='5imp_20out - lower 90%')
    plt.plot(priorVarList, percOutSummaryMat[:, 1], 'o-', color='orange', label='5imp_20out - upper 90%')
    plt.plot(priorVarList, percOutSummaryMat[:, 2], 'o--', color='red', label='10imp_40out - lower 90%')
    plt.plot(priorVarList, percOutSummaryMat[:, 3], 'o-', color='red', label='10imp_40out - upper 90%')
    plt.plot(priorVarList, percOutSummaryMat[:, 4], 'o--', color='purple', label='15imp_60out - lower 90%')
    plt.plot(priorVarList, percOutSummaryMat[:, 5], 'o-', color='purple', label='15imp_60out - upper 90%')
    plt.plot(priorVarList, percOutSummaryMat[:, 6], 'o--', color='blue', label='15imp_100out - lower 90%')
    plt.plot(priorVarList, percOutSummaryMat[:, 7], 'o-', color='blue', label='15imp_100out - upper 90%')
    plt.ylim([0, 1])
    plt.title('90% Intervals on % THAT ARE OUTLETS of Neg. Hessian Diag. Vals.\nvs. Prior Variance, for 4 different system sizes',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Prior variance', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close()

    # Percentage of eigenvalues that are negative
    fig, ax = plt.subplots(figsize=(8, 10), ncols=1)
    plt.plot(priorVarList, percNegEigSummaryMat[:, 0], 'o--', color='orange', label='5imp_20out - lower 90%')
    plt.plot(priorVarList, percNegEigSummaryMat[:, 1], 'o-', color='orange', label='5imp_20out - upper 90%')
    plt.plot(priorVarList, percNegEigSummaryMat[:, 2], 'o--', color='red', label='10imp_40out - lower 90%')
    plt.plot(priorVarList, percNegEigSummaryMat[:, 3], 'o-', color='red', label='10imp_40out - upper 90%')
    plt.plot(priorVarList, percNegEigSummaryMat[:, 4], 'o--', color='purple', label='15imp_60out - lower 90%')
    plt.plot(priorVarList, percNegEigSummaryMat[:, 5], 'o-', color='purple', label='15imp_60out - upper 90%')
    plt.plot(priorVarList, percNegEigSummaryMat[:, 6], 'o--', color='blue', label='15imp_100out - lower 90%')
    plt.plot(priorVarList, percNegEigSummaryMat[:, 7], 'o-', color='blue', label='15imp_100out - upper 90%')
    plt.ylim([0, 1])
    plt.title(
        '90% Intervals on % NEG. EIGENVALUES of Hessian\nvs. Prior Variance, for 4 different system sizes',
        fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Prior variance', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close()


    return

def testtoberemovedlaterforhessanalysis():
    import numpy as np
    for randSys in range(10):
        newSysDict = util.generateRandDataDict(numImp=10, numOut=100, numSamples=100 * 20,
                                               dataType='Tracked')
        _ = util.GetVectorForms(newSysDict)
        newSysDict.update({'prior': methods.prior_normal()})
        outDict = methods.FormEstimates(newSysDict, retOptStatus=True)
        print(np.sum(outDict['optStatus']))
    # Ran for 100 systems of size 10/100; no instance of a non-successful optimizer exit

    # Check the generated Hessian diagonals WRT the prior variance; try for 3 different system sizes
    priorVarList = [0.1,1,3,5,7]
    numSystems = 100

    resultsMat_5_20 = np.zeros((len(priorVarList), numSystems))  # for proportion of Hessian diagonals that are negative
    avgdevsMat_5_20 = np.zeros((len(priorVarList), numSystems))  # for SIZE of negative diagonals
    avgPosdevsMat_5_20 = np.zeros((len(priorVarList), numSystems))  # for SIZE of positive diagonals
    #percOutMat_5_20 = np.zeros((len(priorVarList), numSystems))  # for proportion of negative diagonals that are outlets
    percNegEigMat_5_20 = np.zeros((len(priorVarList), numSystems))  # for proportion negative eigenvalues
    avgEigMat_5_20 = np.zeros((len(priorVarList), numSystems, 5+20))  # for average eigenvalue size
    for currVarInd, currVar in enumerate(priorVarList):
        print('Working on variance of ' + str(currVar) + '...')
        for randSysInd in range(numSystems):  # Systems of size 5, 20
            newSysDict = util.generateRandDataDict(numImp=5, numOut=20, numSamples=20 * 20,
                                                   dataType='Tracked')
            totalEnts = len(newSysDict['importerNames']) + len(newSysDict['outletNames'])
            _ = util.GetVectorForms(newSysDict)
            newSysDict.update({'prior': methods.prior_normal(var=currVar)})
            outDict = methods.FormEstimates(newSysDict, retOptStatus=True, printUpdate=False)
            currHessDiags = np.diag(outDict['hess'])
            negDiags = [i for i in currHessDiags if i < 0]
            posDiags = [i for i in currHessDiags if i > 0]
            resultsMat_5_20[currVarInd, randSysInd] = len(negDiags) / totalEnts
            if len(negDiags) > 0:
                avgdevsMat_5_20[currVarInd, randSysInd] = np.average(negDiags)
            else:
                avgdevsMat_5_20[currVarInd, randSysInd] = 0
            if len(posDiags) > 0:
                avgPosdevsMat_5_20[currVarInd, randSysInd] = np.average(posDiags)
            else:
                avgPosdevsMat_5_20[currVarInd, randSysInd] = 0
            #percOutMat_5_20[currVarInd, randSysInd] = len([i for i in currHessDiags[5:] if i < 0]) / len(negDiags)
            percNegEigMat_5_20[currVarInd, randSysInd] = len(
                [i for i in np.linalg.eigvals(outDict['hess']) if i < 0]) / totalEnts
            avgEigMat_5_20[currVarInd, randSysInd, :] = np.linalg.eigvals(outDict['hess'])
    resultsMat_10_40 = np.zeros((len(priorVarList), numSystems))
    avgdevsMat_10_40 = np.zeros((len(priorVarList), numSystems))
    avgPosdevsMat_10_40 = np.zeros((len(priorVarList), numSystems))  # for SIZE of positive diagonals
    #percOutMat_10_40 = np.zeros((len(priorVarList), numSystems))
    percNegEigMat_10_40 = np.zeros((len(priorVarList), numSystems))
    avgEigMat_10_40 = np.zeros((len(priorVarList), numSystems, 10+40))  # for average eigenvalue size
    for currVarInd, currVar in enumerate(priorVarList):
        print('Working on variance of ' + str(currVar) + '...')
        for randSysInd in range(numSystems):  # Systems of size 5, 20
            newSysDict = util.generateRandDataDict(numImp=10, numOut=40, numSamples=40 * 20,
                                                   dataType='Tracked')
            totalEnts = len(newSysDict['importerNames']) + len(newSysDict['outletNames'])
            _ = util.GetVectorForms(newSysDict)
            newSysDict.update({'prior': methods.prior_normal(var=currVar)})
            outDict = methods.FormEstimates(newSysDict, retOptStatus=True, printUpdate=False)
            currHessDiags = np.diag(outDict['hess'])
            negDiags = [i for i in currHessDiags if i < 0]
            posDiags = [i for i in currHessDiags if i > 0]
            resultsMat_10_40[currVarInd, randSysInd] = len(negDiags) / totalEnts
            if len(negDiags) > 0:
                avgdevsMat_10_40[currVarInd, randSysInd] = np.average(negDiags)
            else:
                avgdevsMat_10_40[currVarInd, randSysInd] = 0
            if len(posDiags) > 0:
                avgPosdevsMat_10_40[currVarInd, randSysInd] = np.average(posDiags)
            else:
                avgPosdevsMat_10_40[currVarInd, randSysInd] = 0
            #percOutMat_10_40[currVarInd, randSysInd] = len([i for i in currHessDiags[10:] if i < 0]) / len(negDiags)
            percNegEigMat_10_40[currVarInd, randSysInd] = len(
                [i for i in np.linalg.eigvals(outDict['hess']) if i < 0]) / totalEnts
            avgEigMat_10_40[currVarInd, randSysInd, :] = np.linalg.eigvals(outDict['hess'])
    resultsMat_15_60 = np.zeros((len(priorVarList), numSystems))
    avgdevsMat_15_60 = np.zeros((len(priorVarList), numSystems))
    avgPosdevsMat_15_60 = np.zeros((len(priorVarList), numSystems))  # for SIZE of positive diagonals
    #percOutMat_15_60 = np.zeros((len(priorVarList), numSystems))
    percNegEigMat_15_60 = np.zeros((len(priorVarList), numSystems))
    avgEigMat_15_60 = np.zeros((len(priorVarList), numSystems, 15+60))  # for average eigenvalue size
    for currVarInd, currVar in enumerate(priorVarList):
        print('Working on variance of ' + str(currVar) + '...')
        for randSysInd in range(numSystems):  # Systems of size 5, 20
            newSysDict = util.generateRandDataDict(numImp=15, numOut=60, numSamples=60 * 20,
                                                   dataType='Tracked')
            totalEnts = len(newSysDict['importerNames']) + len(newSysDict['outletNames'])
            _ = util.GetVectorForms(newSysDict)
            newSysDict.update({'prior': methods.prior_normal(var=currVar)})
            outDict = methods.FormEstimates(newSysDict, retOptStatus=True, printUpdate=False)
            currHessDiags = np.diag(outDict['hess'])
            negDiags = [i for i in currHessDiags if i < 0]
            posDiags = [i for i in currHessDiags if i > 0]
            resultsMat_15_60[currVarInd, randSysInd] = len(negDiags) / totalEnts
            if len(negDiags) > 0:
                avgdevsMat_15_60[currVarInd, randSysInd] = np.average(negDiags)
            else:
                avgdevsMat_15_60[currVarInd, randSysInd] = 0
            if len(posDiags) > 0:
                avgPosdevsMat_15_60[currVarInd, randSysInd] = np.average(posDiags)
            else:
                avgPosdevsMat_15_60[currVarInd, randSysInd] = 0
            #percOutMat_15_60[currVarInd, randSysInd] = len([i for i in currHessDiags[15:] if i < 0]) / len(negDiags)
            percNegEigMat_15_60[currVarInd, randSysInd] = len(
                [i for i in np.linalg.eigvals(outDict['hess']) if i < 0]) / totalEnts
            avgEigMat_15_60[currVarInd, randSysInd, :] = np.linalg.eigvals(outDict['hess'])
    resultsMat_15_100 = np.zeros((len(priorVarList), numSystems))
    avgdevsMat_15_100 = np.zeros((len(priorVarList), numSystems))
    avgPosdevsMat_15_100 = np.zeros((len(priorVarList), numSystems))  # for SIZE of positive diagonals
    #percOutMat_15_100 = np.zeros((len(priorVarList), numSystems))
    percNegEigMat_15_100 = np.zeros((len(priorVarList), numSystems))
    avgEigMat_15_100 = np.zeros((len(priorVarList), numSystems, 15+100))  # for average eigenvalue size
    for currVarInd, currVar in enumerate(priorVarList):
        print('Working on variance of ' + str(currVar) + '...')
        for randSysInd in range(numSystems):  # Systems of size 5, 20
            newSysDict = util.generateRandDataDict(numImp=15, numOut=100, numSamples=100 * 20,
                                                   dataType='Tracked')
            totalEnts = len(newSysDict['importerNames']) + len(newSysDict['outletNames'])
            _ = util.GetVectorForms(newSysDict)
            newSysDict.update({'prior': methods.prior_normal(var=currVar)})
            outDict = methods.FormEstimates(newSysDict, retOptStatus=True, printUpdate=False)
            currHessDiags = np.diag(outDict['hess'])
            negDiags = [i for i in currHessDiags if i < 0]
            posDiags = [i for i in currHessDiags if i > 0]
            resultsMat_15_100[currVarInd, randSysInd] = len(negDiags) / totalEnts
            if len(negDiags) > 0:
                avgdevsMat_15_100[currVarInd, randSysInd] = np.average(negDiags)
            else:
                avgdevsMat_15_100[currVarInd, randSysInd] = 0
            if len(posDiags) > 0:
                avgPosdevsMat_15_100[currVarInd, randSysInd] = np.average(posDiags)
            else:
                avgPosdevsMat_15_100[currVarInd, randSysInd] = 0
            #percOutMat_15_100[currVarInd, randSysInd] = len([i for i in currHessDiags[15:] if i < 0]) / len(negDiags)
            percNegEigMat_15_100[currVarInd, randSysInd] = len(
                [i for i in np.linalg.eigvals(outDict['hess']) if i < 0]) / totalEnts
            avgEigMat_15_100[currVarInd, randSysInd, :] = np.linalg.eigvals(outDict['hess'])
    resultsSummaryMat = np.zeros((len(priorVarList), 8))
    for currVarInd, currVar in enumerate(priorVarList):
        resultsSummaryMat[currVarInd, 0] = np.quantile(resultsMat_5_20[currVarInd, :], 0.05)
        resultsSummaryMat[currVarInd, 1] = np.quantile(resultsMat_5_20[currVarInd, :], 0.95)
        resultsSummaryMat[currVarInd, 2] = np.quantile(resultsMat_10_40[currVarInd, :], 0.05)
        resultsSummaryMat[currVarInd, 3] = np.quantile(resultsMat_10_40[currVarInd, :], 0.95)
        resultsSummaryMat[currVarInd, 4] = np.quantile(resultsMat_15_60[currVarInd, :], 0.05)
        resultsSummaryMat[currVarInd, 5] = np.quantile(resultsMat_15_60[currVarInd, :], 0.95)
        resultsSummaryMat[currVarInd, 6] = np.quantile(resultsMat_15_100[currVarInd, :], 0.05)
        resultsSummaryMat[currVarInd, 7] = np.quantile(resultsMat_15_100[currVarInd, :], 0.95)
    avgdevsSummaryMat = np.zeros((len(priorVarList), 8))
    for currVarInd, currVar in enumerate(priorVarList):
        avgdevsSummaryMat[currVarInd, 0] = np.quantile(avgdevsMat_5_20[currVarInd, :], 0.05)
        avgdevsSummaryMat[currVarInd, 1] = np.quantile(avgdevsMat_5_20[currVarInd, :], 0.95)
        avgdevsSummaryMat[currVarInd, 2] = np.quantile(avgdevsMat_10_40[currVarInd, :], 0.05)
        avgdevsSummaryMat[currVarInd, 3] = np.quantile(avgdevsMat_10_40[currVarInd, :], 0.95)
        avgdevsSummaryMat[currVarInd, 4] = np.quantile(avgdevsMat_15_60[currVarInd, :], 0.05)
        avgdevsSummaryMat[currVarInd, 5] = np.quantile(avgdevsMat_15_60[currVarInd, :], 0.95)
        avgdevsSummaryMat[currVarInd, 6] = np.quantile(avgdevsMat_15_100[currVarInd, :], 0.05)
        avgdevsSummaryMat[currVarInd, 7] = np.quantile(avgdevsMat_15_100[currVarInd, :], 0.95)
    avgPosdevsSummaryMat = np.zeros((len(priorVarList), 8))
    for currVarInd, currVar in enumerate(priorVarList):
        avgPosdevsSummaryMat[currVarInd, 0] = np.quantile(avgPosdevsMat_5_20[currVarInd, :], 0.05)
        avgPosdevsSummaryMat[currVarInd, 1] = np.quantile(avgPosdevsMat_5_20[currVarInd, :], 0.95)
        avgPosdevsSummaryMat[currVarInd, 2] = np.quantile(avgPosdevsMat_10_40[currVarInd, :], 0.05)
        avgPosdevsSummaryMat[currVarInd, 3] = np.quantile(avgPosdevsMat_10_40[currVarInd, :], 0.95)
        avgPosdevsSummaryMat[currVarInd, 4] = np.quantile(avgPosdevsMat_15_60[currVarInd, :], 0.05)
        avgPosdevsSummaryMat[currVarInd, 5] = np.quantile(avgPosdevsMat_15_60[currVarInd, :], 0.95)
        avgPosdevsSummaryMat[currVarInd, 6] = np.quantile(avgPosdevsMat_15_100[currVarInd, :], 0.05)
        avgPosdevsSummaryMat[currVarInd, 7] = np.quantile(avgPosdevsMat_15_100[currVarInd, :], 0.95)
    '''
    percOutSummaryMat = np.zeros((len(priorVarList), 8))
    for currVarInd, currVar in enumerate(priorVarList):
        percOutSummaryMat[currVarInd, 0] = np.quantile(percOutMat_5_20[currVarInd, :], 0.05)
        percOutSummaryMat[currVarInd, 1] = np.quantile(percOutMat_5_20[currVarInd, :], 0.95)
        percOutSummaryMat[currVarInd, 2] = np.quantile(percOutMat_10_40[currVarInd, :], 0.05)
        percOutSummaryMat[currVarInd, 3] = np.quantile(percOutMat_10_40[currVarInd, :], 0.95)
        percOutSummaryMat[currVarInd, 4] = np.quantile(percOutMat_15_60[currVarInd, :], 0.05)
        percOutSummaryMat[currVarInd, 5] = np.quantile(percOutMat_15_60[currVarInd, :], 0.95)
        percOutSummaryMat[currVarInd, 6] = np.quantile(percOutMat_15_100[currVarInd, :], 0.05)
        percOutSummaryMat[currVarInd, 7] = np.quantile(percOutMat_15_100[currVarInd, :], 0.95)
    '''
    percNegEigSummaryMat = np.zeros((len(priorVarList), 8))
    for currVarInd, currVar in enumerate(priorVarList):
        percNegEigSummaryMat[currVarInd, 0] = np.quantile(percNegEigMat_5_20[currVarInd, :], 0.05)
        percNegEigSummaryMat[currVarInd, 1] = np.quantile(percNegEigMat_5_20[currVarInd, :], 0.95)
        percNegEigSummaryMat[currVarInd, 2] = np.quantile(percNegEigMat_10_40[currVarInd, :], 0.05)
        percNegEigSummaryMat[currVarInd, 3] = np.quantile(percNegEigMat_10_40[currVarInd, :], 0.95)
        percNegEigSummaryMat[currVarInd, 4] = np.quantile(percNegEigMat_15_60[currVarInd, :], 0.05)
        percNegEigSummaryMat[currVarInd, 5] = np.quantile(percNegEigMat_15_60[currVarInd, :], 0.95)
        percNegEigSummaryMat[currVarInd, 6] = np.quantile(percNegEigMat_15_100[currVarInd, :], 0.05)
        percNegEigSummaryMat[currVarInd, 7] = np.quantile(percNegEigMat_15_100[currVarInd, :], 0.95)

    avgEigSummaryMat = np.zeros((len(priorVarList), 8))
    for currVarInd, currVar in enumerate(priorVarList):
        avgEigSummaryMat[currVarInd, 0] = np.quantile(avgEigMat_5_20[currVarInd, :, :], 0.05)
        avgEigSummaryMat[currVarInd, 1] = np.quantile(avgEigMat_5_20[currVarInd, :, :], 0.95)
        avgEigSummaryMat[currVarInd, 2] = np.quantile(avgEigMat_10_40[currVarInd, :, :], 0.05)
        avgEigSummaryMat[currVarInd, 3] = np.quantile(avgEigMat_10_40[currVarInd, :, :], 0.95)
        avgEigSummaryMat[currVarInd, 4] = np.quantile(avgEigMat_15_60[currVarInd, :, :], 0.05)
        avgEigSummaryMat[currVarInd, 5] = np.quantile(avgEigMat_15_60[currVarInd, :, :], 0.95)
        avgEigSummaryMat[currVarInd, 6] = np.quantile(avgEigMat_15_100[currVarInd, :, :], 0.05)
        avgEigSummaryMat[currVarInd, 7] = np.quantile(avgEigMat_15_100[currVarInd, :, :], 0.95)
    import matplotlib.pyplot as plt
    # from matplotlib.lines import Line2D

    # zippedList1 = zip(resultsSummaryMat[:, 0], resultsSummaryMat[:, 1], priorVarList)
    # zippedList2 = zip(resultsSummaryMat[:, 2], resultsSummaryMat[:, 3], priorVarList)
    # zippedList3 = zip(resultsSummaryMat[:, 4], resultsSummaryMat[:, 5], priorVarList)

    # custom_lines = [Line2D([0], [0], color='orange', lw=4),
    #                Line2D([0], [0], color='red', lw=4),
    #                Line2D([0], [0], color='purple', lw=4)]
    fig, ax = plt.subplots(figsize=(8, 10), ncols=1)
    # for lower, upper, name in zippedList1:
    #    plt.plot((name, name), (lower, upper), 'o-', color='orange')
    # for lower, upper, name in zippedList2:
    #    plt.plot((name, name), (lower, upper), 'o-', color='red')
    # for lower, upper, name in zippedList3:
    #    plt.plot((name, name), (lower, upper), 'o-', color='purple')
    plt.plot(priorVarList, resultsSummaryMat[:, 0], 'o--', color='orange', label='5imp_20out - lower 90%')
    plt.plot(priorVarList, resultsSummaryMat[:, 1], 'o-', color='orange', label='5imp_20out - upper 90%')
    plt.plot(priorVarList, resultsSummaryMat[:, 2], 'o--', color='red', label='10imp_40out - lower 90%')
    plt.plot(priorVarList, resultsSummaryMat[:, 3], 'o-', color='red', label='10imp_40out - upper 90%')
    plt.plot(priorVarList, resultsSummaryMat[:, 4], 'o--', color='purple', label='15imp_60out - lower 90%')
    plt.plot(priorVarList, resultsSummaryMat[:, 5], 'o-', color='purple', label='15imp_60out - upper 90%')
    plt.plot(priorVarList, resultsSummaryMat[:, 6], 'o--', color='blue', label='15imp_100out - lower 90%')
    plt.plot(priorVarList, resultsSummaryMat[:, 7], 'o-', color='blue', label='15imp_100out - upper 90%')
    plt.ylim([0, 1])
    plt.title(
        '90% Intervals on PERCENTAGE of Neg. Hessian Diagonal Values\nvs. Prior Variance, for 4 different system sizes',
        fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Prior variance', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    # ax.legend(custom_lines, ['5 importers, 20 outlets', '10 importers, 40 outlets', '15 importers, 60 outlets'])
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close()

    # Size of deviations below 0
    fig, ax = plt.subplots(figsize=(8, 10), ncols=1)
    plt.plot(priorVarList, avgdevsSummaryMat[:, 0], 'o--', color='orange', label='5imp_20out - lower 90%')
    plt.plot(priorVarList, avgdevsSummaryMat[:, 1], 'o-', color='orange', label='5imp_20out - upper 90%')
    plt.plot(priorVarList, avgdevsSummaryMat[:, 2], 'o--', color='red', label='10imp_40out - lower 90%')
    plt.plot(priorVarList, avgdevsSummaryMat[:, 3], 'o-', color='red', label='10imp_40out - upper 90%')
    plt.plot(priorVarList, avgdevsSummaryMat[:, 4], 'o--', color='purple', label='15imp_60out - lower 90%')
    plt.plot(priorVarList, avgdevsSummaryMat[:, 5], 'o-', color='purple', label='15imp_60out - upper 90%')
    plt.plot(priorVarList, avgdevsSummaryMat[:, 6], 'o--', color='blue', label='15imp_100out - lower 90%')
    plt.plot(priorVarList, avgdevsSummaryMat[:, 7], 'o-', color='blue', label='15imp_100out - upper 90%')
    plt.ylim([-3, 0])
    plt.title('90% Intervals on SIZE of Neg. Hessian Diagonal Values\nvs. Prior Variance, for 4 different system sizes',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Prior variance', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close()

    # Size of deviations above 0
    fig, ax = plt.subplots(figsize=(8, 10), ncols=1)
    plt.plot(priorVarList, avgPosdevsSummaryMat[:, 0], 'o--', color='orange', label='5imp_20out - lower 90%')
    plt.plot(priorVarList, avgPosdevsSummaryMat[:, 1], 'o-', color='orange', label='5imp_20out - upper 90%')
    plt.plot(priorVarList, avgPosdevsSummaryMat[:, 2], 'o--', color='red', label='10imp_40out - lower 90%')
    plt.plot(priorVarList, avgPosdevsSummaryMat[:, 3], 'o-', color='red', label='10imp_40out - upper 90%')
    plt.plot(priorVarList, avgPosdevsSummaryMat[:, 4], 'o--', color='purple', label='15imp_60out - lower 90%')
    plt.plot(priorVarList, avgPosdevsSummaryMat[:, 5], 'o-', color='purple', label='15imp_60out - upper 90%')
    plt.plot(priorVarList, avgPosdevsSummaryMat[:, 6], 'o--', color='blue', label='15imp_100out - lower 90%')
    plt.plot(priorVarList, avgPosdevsSummaryMat[:, 7], 'o-', color='blue', label='15imp_100out - upper 90%')
    plt.ylim([0, 13])
    plt.title('90% Intervals on SIZE of Pos. Hessian Diagonal Values\nvs. Prior Variance, for 4 different system sizes',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Prior variance', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close()

    # Percentage of negative diagonals that are outlets
    '''
    fig, ax = plt.subplots(figsize=(8, 10), ncols=1)
    plt.plot(priorVarList, percOutSummaryMat[:, 0], 'o--', color='orange', label='5imp_20out - lower 90%')
    plt.plot(priorVarList, percOutSummaryMat[:, 1], 'o-', color='orange', label='5imp_20out - upper 90%')
    plt.plot(priorVarList, percOutSummaryMat[:, 2], 'o--', color='red', label='10imp_40out - lower 90%')
    plt.plot(priorVarList, percOutSummaryMat[:, 3], 'o-', color='red', label='10imp_40out - upper 90%')
    plt.plot(priorVarList, percOutSummaryMat[:, 4], 'o--', color='purple', label='15imp_60out - lower 90%')
    plt.plot(priorVarList, percOutSummaryMat[:, 5], 'o-', color='purple', label='15imp_60out - upper 90%')
    plt.plot(priorVarList, percOutSummaryMat[:, 6], 'o--', color='blue', label='15imp_100out - lower 90%')
    plt.plot(priorVarList, percOutSummaryMat[:, 7], 'o-', color='blue', label='15imp_100out - upper 90%')
    plt.ylim([0, 1])
    plt.title(
        '90% Intervals on % THAT ARE OUTLETS of Neg. Hessian Diag. Vals.\nvs. Prior Variance, for 4 different system sizes',
        fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Prior variance', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close()
    '''

    # Percentage of eigenvalues that are negative
    fig, ax = plt.subplots(figsize=(8, 10), ncols=1)
    plt.plot(priorVarList, percNegEigSummaryMat[:, 0], 'o--', color='orange', label='5imp_20out - lower 90%')
    plt.plot(priorVarList, percNegEigSummaryMat[:, 1], 'o-', color='orange', label='5imp_20out - upper 90%')
    plt.plot(priorVarList, percNegEigSummaryMat[:, 2], 'o--', color='red', label='10imp_40out - lower 90%')
    plt.plot(priorVarList, percNegEigSummaryMat[:, 3], 'o-', color='red', label='10imp_40out - upper 90%')
    plt.plot(priorVarList, percNegEigSummaryMat[:, 4], 'o--', color='purple', label='15imp_60out - lower 90%')
    plt.plot(priorVarList, percNegEigSummaryMat[:, 5], 'o-', color='purple', label='15imp_60out - upper 90%')
    plt.plot(priorVarList, percNegEigSummaryMat[:, 6], 'o--', color='blue', label='15imp_100out - lower 90%')
    plt.plot(priorVarList, percNegEigSummaryMat[:, 7], 'o-', color='blue', label='15imp_100out - upper 90%')
    plt.ylim([0, 1])
    plt.title(
        '90% Intervals on % NEG. EIGENVALUES of Hessian\nvs. Prior Variance, for 4 different system sizes',
        fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Prior variance', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close()

    # Distribution of size of eigenvalues
    fig, ax = plt.subplots(figsize=(8, 10), ncols=1)
    plt.plot(priorVarList, avgEigSummaryMat[:, 0], 'o--', color='orange', label='5imp_20out - lower 90%')
    plt.plot(priorVarList, avgEigSummaryMat[:, 1], 'o-', color='orange', label='5imp_20out - upper 90%')
    plt.plot(priorVarList, avgEigSummaryMat[:, 2], 'o--', color='red', label='10imp_40out - lower 90%')
    plt.plot(priorVarList, avgEigSummaryMat[:, 3], 'o-', color='red', label='10imp_40out - upper 90%')
    plt.plot(priorVarList, avgEigSummaryMat[:, 4], 'o--', color='purple', label='15imp_60out - lower 90%')
    plt.plot(priorVarList, avgEigSummaryMat[:, 5], 'o-', color='purple', label='15imp_60out - upper 90%')
    plt.plot(priorVarList, avgEigSummaryMat[:, 6], 'o--', color='blue', label='15imp_100out - lower 90%')
    plt.plot(priorVarList, avgEigSummaryMat[:, 7], 'o-', color='blue', label='15imp_100out - upper 90%')
    plt.ylim([0, 1])
    plt.title(
        '90% Intervals on SIZE OF EIGENVALUES of Hessian\nvs. Prior Variance, for 4 different system sizes',
        fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Prior variance', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close()

    return

def testing():
    # For testing the Laplace and MCMC methods under non-diffuse sourcing matrices
    testing = util.generateRandDataDict(numImp=5, numOut=3, diagSens=0.90,
                         diagSpec=0.99, numSamples=5 * 50,
                         dataType='Tracked', transMatLambda=0.05,
                         randSeed=3,trueRates=[])
    _ = util.GetVectorForms(testing)
    print(testing['transMat'])
    import numpy as np
    print(testing['N'])
    print(testing['Y'])
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    testing.update({'diagSens': 0.90,
                        'diagSpec': 0.99,
                        'numPostSamples': 500,
                        'prior': methods.prior_normal(),
                        'MCMCdict': MCMCdict})
    logistigateDict = runlogistigate(testing)
    util.plotPostSamples(logistigateDict)

    return

def testtoberemovedlaterforhessanalysis():
    import numpy as np
    for randSys in range(10):
        newSysDict = util.generateRandDataDict(numImp=10, numOut=100, numSamples=100 * 20,
                                               dataType='Tracked')
        _ = util.GetVectorForms(newSysDict)
        newSysDict.update({'prior': methods.prior_normal()})
        outDict = methods.FormEstimates(newSysDict, retOptStatus=True)
        print(np.sum(outDict['optStatus']))
    # Ran for 100 systems of size 10/100; no instance of a non-successful optimizer exit

    # Check the generated Hessian diagonals WRT the prior variance; try for 3 different system sizes
    priorVarList = [0.1,1,3,5,7]
    numSystems = 100

    resultsMat_5_20 = np.zeros((len(priorVarList), numSystems))  # for proportion of Hessian diagonals that are negative
    avgdevsMat_5_20 = np.zeros((len(priorVarList), numSystems))  # for SIZE of negative diagonals
    avgPosdevsMat_5_20 = np.zeros((len(priorVarList), numSystems))  # for SIZE of positive diagonals
    #percOutMat_5_20 = np.zeros((len(priorVarList), numSystems))  # for proportion of negative diagonals that are outlets
    percNegEigMat_5_20 = np.zeros((len(priorVarList), numSystems))  # for proportion negative eigenvalues
    avgEigMat_5_20 = np.zeros((len(priorVarList), numSystems, 5+20))  # for average eigenvalue size
    for currVarInd, currVar in enumerate(priorVarList):
        print('Working on variance of ' + str(currVar) + '...')
        for randSysInd in range(numSystems):  # Systems of size 5, 20
            newSysDict = util.generateRandDataDict(numImp=5, numOut=20, numSamples=20 * 20,
                                                   dataType='Tracked')
            totalEnts = len(newSysDict['importerNames']) + len(newSysDict['outletNames'])
            _ = util.GetVectorForms(newSysDict)
            newSysDict.update({'prior': methods.prior_normal(var=currVar)})
            outDict = methods.FormEstimates(newSysDict, retOptStatus=True, printUpdate=False)
            currHessDiags = np.diag(outDict['hess'])
            negDiags = [i for i in currHessDiags if i < 0]
            posDiags = [i for i in currHessDiags if i > 0]
            resultsMat_5_20[currVarInd, randSysInd] = len(negDiags) / totalEnts
            if len(negDiags) > 0:
                avgdevsMat_5_20[currVarInd, randSysInd] = np.average(negDiags)
            else:
                avgdevsMat_5_20[currVarInd, randSysInd] = 0
            if len(posDiags) > 0:
                avgPosdevsMat_5_20[currVarInd, randSysInd] = np.average(posDiags)
            else:
                avgPosdevsMat_5_20[currVarInd, randSysInd] = 0
            #percOutMat_5_20[currVarInd, randSysInd] = len([i for i in currHessDiags[5:] if i < 0]) / len(negDiags)
            percNegEigMat_5_20[currVarInd, randSysInd] = len(
                [i for i in np.linalg.eigvals(outDict['hess']) if i < 0]) / totalEnts
            avgEigMat_5_20[currVarInd, randSysInd, :] = np.linalg.eigvals(outDict['hess'])
    resultsMat_10_40 = np.zeros((len(priorVarList), numSystems))
    avgdevsMat_10_40 = np.zeros((len(priorVarList), numSystems))
    avgPosdevsMat_10_40 = np.zeros((len(priorVarList), numSystems))  # for SIZE of positive diagonals
    #percOutMat_10_40 = np.zeros((len(priorVarList), numSystems))
    percNegEigMat_10_40 = np.zeros((len(priorVarList), numSystems))
    avgEigMat_10_40 = np.zeros((len(priorVarList), numSystems, 10+40))  # for average eigenvalue size
    for currVarInd, currVar in enumerate(priorVarList):
        print('Working on variance of ' + str(currVar) + '...')
        for randSysInd in range(numSystems):  # Systems of size 5, 20
            newSysDict = util.generateRandDataDict(numImp=10, numOut=40, numSamples=40 * 20,
                                                   dataType='Tracked')
            totalEnts = len(newSysDict['importerNames']) + len(newSysDict['outletNames'])
            _ = util.GetVectorForms(newSysDict)
            newSysDict.update({'prior': methods.prior_normal(var=currVar)})
            outDict = methods.FormEstimates(newSysDict, retOptStatus=True, printUpdate=False)
            currHessDiags = np.diag(outDict['hess'])
            negDiags = [i for i in currHessDiags if i < 0]
            posDiags = [i for i in currHessDiags if i > 0]
            resultsMat_10_40[currVarInd, randSysInd] = len(negDiags) / totalEnts
            if len(negDiags) > 0:
                avgdevsMat_10_40[currVarInd, randSysInd] = np.average(negDiags)
            else:
                avgdevsMat_10_40[currVarInd, randSysInd] = 0
            if len(posDiags) > 0:
                avgPosdevsMat_10_40[currVarInd, randSysInd] = np.average(posDiags)
            else:
                avgPosdevsMat_10_40[currVarInd, randSysInd] = 0
            #percOutMat_10_40[currVarInd, randSysInd] = len([i for i in currHessDiags[10:] if i < 0]) / len(negDiags)
            percNegEigMat_10_40[currVarInd, randSysInd] = len(
                [i for i in np.linalg.eigvals(outDict['hess']) if i < 0]) / totalEnts
            avgEigMat_10_40[currVarInd, randSysInd, :] = np.linalg.eigvals(outDict['hess'])
    resultsMat_15_60 = np.zeros((len(priorVarList), numSystems))
    avgdevsMat_15_60 = np.zeros((len(priorVarList), numSystems))
    avgPosdevsMat_15_60 = np.zeros((len(priorVarList), numSystems))  # for SIZE of positive diagonals
    #percOutMat_15_60 = np.zeros((len(priorVarList), numSystems))
    percNegEigMat_15_60 = np.zeros((len(priorVarList), numSystems))
    avgEigMat_15_60 = np.zeros((len(priorVarList), numSystems, 15+60))  # for average eigenvalue size
    for currVarInd, currVar in enumerate(priorVarList):
        print('Working on variance of ' + str(currVar) + '...')
        for randSysInd in range(numSystems):  # Systems of size 5, 20
            newSysDict = util.generateRandDataDict(numImp=15, numOut=60, numSamples=60 * 20,
                                                   dataType='Tracked')
            totalEnts = len(newSysDict['importerNames']) + len(newSysDict['outletNames'])
            _ = util.GetVectorForms(newSysDict)
            newSysDict.update({'prior': methods.prior_normal(var=currVar)})
            outDict = methods.FormEstimates(newSysDict, retOptStatus=True, printUpdate=False)
            currHessDiags = np.diag(outDict['hess'])
            negDiags = [i for i in currHessDiags if i < 0]
            posDiags = [i for i in currHessDiags if i > 0]
            resultsMat_15_60[currVarInd, randSysInd] = len(negDiags) / totalEnts
            if len(negDiags) > 0:
                avgdevsMat_15_60[currVarInd, randSysInd] = np.average(negDiags)
            else:
                avgdevsMat_15_60[currVarInd, randSysInd] = 0
            if len(posDiags) > 0:
                avgPosdevsMat_15_60[currVarInd, randSysInd] = np.average(posDiags)
            else:
                avgPosdevsMat_15_60[currVarInd, randSysInd] = 0
            #percOutMat_15_60[currVarInd, randSysInd] = len([i for i in currHessDiags[15:] if i < 0]) / len(negDiags)
            percNegEigMat_15_60[currVarInd, randSysInd] = len(
                [i for i in np.linalg.eigvals(outDict['hess']) if i < 0]) / totalEnts
            avgEigMat_15_60[currVarInd, randSysInd, :] = np.linalg.eigvals(outDict['hess'])
    resultsMat_15_100 = np.zeros((len(priorVarList), numSystems))
    avgdevsMat_15_100 = np.zeros((len(priorVarList), numSystems))
    avgPosdevsMat_15_100 = np.zeros((len(priorVarList), numSystems))  # for SIZE of positive diagonals
    #percOutMat_15_100 = np.zeros((len(priorVarList), numSystems))
    percNegEigMat_15_100 = np.zeros((len(priorVarList), numSystems))
    avgEigMat_15_100 = np.zeros((len(priorVarList), numSystems, 15+100))  # for average eigenvalue size
    for currVarInd, currVar in enumerate(priorVarList):
        print('Working on variance of ' + str(currVar) + '...')
        for randSysInd in range(numSystems):  # Systems of size 5, 20
            newSysDict = util.generateRandDataDict(numImp=15, numOut=100, numSamples=100 * 20,
                                                   dataType='Tracked')
            totalEnts = len(newSysDict['importerNames']) + len(newSysDict['outletNames'])
            _ = util.GetVectorForms(newSysDict)
            newSysDict.update({'prior': methods.prior_normal(var=currVar)})
            outDict = methods.FormEstimates(newSysDict, retOptStatus=True, printUpdate=False)
            currHessDiags = np.diag(outDict['hess'])
            negDiags = [i for i in currHessDiags if i < 0]
            posDiags = [i for i in currHessDiags if i > 0]
            resultsMat_15_100[currVarInd, randSysInd] = len(negDiags) / totalEnts
            if len(negDiags) > 0:
                avgdevsMat_15_100[currVarInd, randSysInd] = np.average(negDiags)
            else:
                avgdevsMat_15_100[currVarInd, randSysInd] = 0
            if len(posDiags) > 0:
                avgPosdevsMat_15_100[currVarInd, randSysInd] = np.average(posDiags)
            else:
                avgPosdevsMat_15_100[currVarInd, randSysInd] = 0
            #percOutMat_15_100[currVarInd, randSysInd] = len([i for i in currHessDiags[15:] if i < 0]) / len(negDiags)
            percNegEigMat_15_100[currVarInd, randSysInd] = len(
                [i for i in np.linalg.eigvals(outDict['hess']) if i < 0]) / totalEnts
            avgEigMat_15_100[currVarInd, randSysInd, :] = np.linalg.eigvals(outDict['hess'])
    resultsSummaryMat = np.zeros((len(priorVarList), 8))
    for currVarInd, currVar in enumerate(priorVarList):
        resultsSummaryMat[currVarInd, 0] = np.quantile(resultsMat_5_20[currVarInd, :], 0.05)
        resultsSummaryMat[currVarInd, 1] = np.quantile(resultsMat_5_20[currVarInd, :], 0.95)
        resultsSummaryMat[currVarInd, 2] = np.quantile(resultsMat_10_40[currVarInd, :], 0.05)
        resultsSummaryMat[currVarInd, 3] = np.quantile(resultsMat_10_40[currVarInd, :], 0.95)
        resultsSummaryMat[currVarInd, 4] = np.quantile(resultsMat_15_60[currVarInd, :], 0.05)
        resultsSummaryMat[currVarInd, 5] = np.quantile(resultsMat_15_60[currVarInd, :], 0.95)
        resultsSummaryMat[currVarInd, 6] = np.quantile(resultsMat_15_100[currVarInd, :], 0.05)
        resultsSummaryMat[currVarInd, 7] = np.quantile(resultsMat_15_100[currVarInd, :], 0.95)
    avgdevsSummaryMat = np.zeros((len(priorVarList), 8))
    for currVarInd, currVar in enumerate(priorVarList):
        avgdevsSummaryMat[currVarInd, 0] = np.quantile(avgdevsMat_5_20[currVarInd, :], 0.05)
        avgdevsSummaryMat[currVarInd, 1] = np.quantile(avgdevsMat_5_20[currVarInd, :], 0.95)
        avgdevsSummaryMat[currVarInd, 2] = np.quantile(avgdevsMat_10_40[currVarInd, :], 0.05)
        avgdevsSummaryMat[currVarInd, 3] = np.quantile(avgdevsMat_10_40[currVarInd, :], 0.95)
        avgdevsSummaryMat[currVarInd, 4] = np.quantile(avgdevsMat_15_60[currVarInd, :], 0.05)
        avgdevsSummaryMat[currVarInd, 5] = np.quantile(avgdevsMat_15_60[currVarInd, :], 0.95)
        avgdevsSummaryMat[currVarInd, 6] = np.quantile(avgdevsMat_15_100[currVarInd, :], 0.05)
        avgdevsSummaryMat[currVarInd, 7] = np.quantile(avgdevsMat_15_100[currVarInd, :], 0.95)
    avgPosdevsSummaryMat = np.zeros((len(priorVarList), 8))
    for currVarInd, currVar in enumerate(priorVarList):
        avgPosdevsSummaryMat[currVarInd, 0] = np.quantile(avgPosdevsMat_5_20[currVarInd, :], 0.05)
        avgPosdevsSummaryMat[currVarInd, 1] = np.quantile(avgPosdevsMat_5_20[currVarInd, :], 0.95)
        avgPosdevsSummaryMat[currVarInd, 2] = np.quantile(avgPosdevsMat_10_40[currVarInd, :], 0.05)
        avgPosdevsSummaryMat[currVarInd, 3] = np.quantile(avgPosdevsMat_10_40[currVarInd, :], 0.95)
        avgPosdevsSummaryMat[currVarInd, 4] = np.quantile(avgPosdevsMat_15_60[currVarInd, :], 0.05)
        avgPosdevsSummaryMat[currVarInd, 5] = np.quantile(avgPosdevsMat_15_60[currVarInd, :], 0.95)
        avgPosdevsSummaryMat[currVarInd, 6] = np.quantile(avgPosdevsMat_15_100[currVarInd, :], 0.05)
        avgPosdevsSummaryMat[currVarInd, 7] = np.quantile(avgPosdevsMat_15_100[currVarInd, :], 0.95)
    '''
    percOutSummaryMat = np.zeros((len(priorVarList), 8))
    for currVarInd, currVar in enumerate(priorVarList):
        percOutSummaryMat[currVarInd, 0] = np.quantile(percOutMat_5_20[currVarInd, :], 0.05)
        percOutSummaryMat[currVarInd, 1] = np.quantile(percOutMat_5_20[currVarInd, :], 0.95)
        percOutSummaryMat[currVarInd, 2] = np.quantile(percOutMat_10_40[currVarInd, :], 0.05)
        percOutSummaryMat[currVarInd, 3] = np.quantile(percOutMat_10_40[currVarInd, :], 0.95)
        percOutSummaryMat[currVarInd, 4] = np.quantile(percOutMat_15_60[currVarInd, :], 0.05)
        percOutSummaryMat[currVarInd, 5] = np.quantile(percOutMat_15_60[currVarInd, :], 0.95)
        percOutSummaryMat[currVarInd, 6] = np.quantile(percOutMat_15_100[currVarInd, :], 0.05)
        percOutSummaryMat[currVarInd, 7] = np.quantile(percOutMat_15_100[currVarInd, :], 0.95)
    '''
    percNegEigSummaryMat = np.zeros((len(priorVarList), 8))
    for currVarInd, currVar in enumerate(priorVarList):
        percNegEigSummaryMat[currVarInd, 0] = np.quantile(percNegEigMat_5_20[currVarInd, :], 0.05)
        percNegEigSummaryMat[currVarInd, 1] = np.quantile(percNegEigMat_5_20[currVarInd, :], 0.95)
        percNegEigSummaryMat[currVarInd, 2] = np.quantile(percNegEigMat_10_40[currVarInd, :], 0.05)
        percNegEigSummaryMat[currVarInd, 3] = np.quantile(percNegEigMat_10_40[currVarInd, :], 0.95)
        percNegEigSummaryMat[currVarInd, 4] = np.quantile(percNegEigMat_15_60[currVarInd, :], 0.05)
        percNegEigSummaryMat[currVarInd, 5] = np.quantile(percNegEigMat_15_60[currVarInd, :], 0.95)
        percNegEigSummaryMat[currVarInd, 6] = np.quantile(percNegEigMat_15_100[currVarInd, :], 0.05)
        percNegEigSummaryMat[currVarInd, 7] = np.quantile(percNegEigMat_15_100[currVarInd, :], 0.95)

    avgEigSummaryMat = np.zeros((len(priorVarList), 8))
    for currVarInd, currVar in enumerate(priorVarList):
        avgEigSummaryMat[currVarInd, 0] = np.quantile(avgEigMat_5_20[currVarInd, :, :], 0.05)
        avgEigSummaryMat[currVarInd, 1] = np.quantile(avgEigMat_5_20[currVarInd, :, :], 0.95)
        avgEigSummaryMat[currVarInd, 2] = np.quantile(avgEigMat_10_40[currVarInd, :, :], 0.05)
        avgEigSummaryMat[currVarInd, 3] = np.quantile(avgEigMat_10_40[currVarInd, :, :], 0.95)
        avgEigSummaryMat[currVarInd, 4] = np.quantile(avgEigMat_15_60[currVarInd, :, :], 0.05)
        avgEigSummaryMat[currVarInd, 5] = np.quantile(avgEigMat_15_60[currVarInd, :, :], 0.95)
        avgEigSummaryMat[currVarInd, 6] = np.quantile(avgEigMat_15_100[currVarInd, :, :], 0.05)
        avgEigSummaryMat[currVarInd, 7] = np.quantile(avgEigMat_15_100[currVarInd, :, :], 0.95)
    import matplotlib.pyplot as plt
    # from matplotlib.lines import Line2D

    # zippedList1 = zip(resultsSummaryMat[:, 0], resultsSummaryMat[:, 1], priorVarList)
    # zippedList2 = zip(resultsSummaryMat[:, 2], resultsSummaryMat[:, 3], priorVarList)
    # zippedList3 = zip(resultsSummaryMat[:, 4], resultsSummaryMat[:, 5], priorVarList)

    # custom_lines = [Line2D([0], [0], color='orange', lw=4),
    #                Line2D([0], [0], color='red', lw=4),
    #                Line2D([0], [0], color='purple', lw=4)]
    fig, ax = plt.subplots(figsize=(8, 10), ncols=1)
    # for lower, upper, name in zippedList1:
    #    plt.plot((name, name), (lower, upper), 'o-', color='orange')
    # for lower, upper, name in zippedList2:
    #    plt.plot((name, name), (lower, upper), 'o-', color='red')
    # for lower, upper, name in zippedList3:
    #    plt.plot((name, name), (lower, upper), 'o-', color='purple')
    plt.plot(priorVarList, resultsSummaryMat[:, 0], 'o--', color='orange', label='5imp_20out - lower 90%')
    plt.plot(priorVarList, resultsSummaryMat[:, 1], 'o-', color='orange', label='5imp_20out - upper 90%')
    plt.plot(priorVarList, resultsSummaryMat[:, 2], 'o--', color='red', label='10imp_40out - lower 90%')
    plt.plot(priorVarList, resultsSummaryMat[:, 3], 'o-', color='red', label='10imp_40out - upper 90%')
    plt.plot(priorVarList, resultsSummaryMat[:, 4], 'o--', color='purple', label='15imp_60out - lower 90%')
    plt.plot(priorVarList, resultsSummaryMat[:, 5], 'o-', color='purple', label='15imp_60out - upper 90%')
    plt.plot(priorVarList, resultsSummaryMat[:, 6], 'o--', color='blue', label='15imp_100out - lower 90%')
    plt.plot(priorVarList, resultsSummaryMat[:, 7], 'o-', color='blue', label='15imp_100out - upper 90%')
    plt.ylim([0, 1])
    plt.title(
        '90% Intervals on PERCENTAGE of Neg. Hessian Diagonal Values\nvs. Prior Variance, for 4 different system sizes',
        fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Prior variance', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    # ax.legend(custom_lines, ['5 importers, 20 outlets', '10 importers, 40 outlets', '15 importers, 60 outlets'])
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close()

    # Size of deviations below 0
    fig, ax = plt.subplots(figsize=(8, 10), ncols=1)
    plt.plot(priorVarList, avgdevsSummaryMat[:, 0], 'o--', color='orange', label='5imp_20out - lower 90%')
    plt.plot(priorVarList, avgdevsSummaryMat[:, 1], 'o-', color='orange', label='5imp_20out - upper 90%')
    plt.plot(priorVarList, avgdevsSummaryMat[:, 2], 'o--', color='red', label='10imp_40out - lower 90%')
    plt.plot(priorVarList, avgdevsSummaryMat[:, 3], 'o-', color='red', label='10imp_40out - upper 90%')
    plt.plot(priorVarList, avgdevsSummaryMat[:, 4], 'o--', color='purple', label='15imp_60out - lower 90%')
    plt.plot(priorVarList, avgdevsSummaryMat[:, 5], 'o-', color='purple', label='15imp_60out - upper 90%')
    plt.plot(priorVarList, avgdevsSummaryMat[:, 6], 'o--', color='blue', label='15imp_100out - lower 90%')
    plt.plot(priorVarList, avgdevsSummaryMat[:, 7], 'o-', color='blue', label='15imp_100out - upper 90%')
    plt.ylim([-3, 0])
    plt.title('90% Intervals on SIZE of Neg. Hessian Diagonal Values\nvs. Prior Variance, for 4 different system sizes',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Prior variance', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close()

    # Size of deviations above 0
    fig, ax = plt.subplots(figsize=(8, 10), ncols=1)
    plt.plot(priorVarList, avgPosdevsSummaryMat[:, 0], 'o--', color='orange', label='5imp_20out - lower 90%')
    plt.plot(priorVarList, avgPosdevsSummaryMat[:, 1], 'o-', color='orange', label='5imp_20out - upper 90%')
    plt.plot(priorVarList, avgPosdevsSummaryMat[:, 2], 'o--', color='red', label='10imp_40out - lower 90%')
    plt.plot(priorVarList, avgPosdevsSummaryMat[:, 3], 'o-', color='red', label='10imp_40out - upper 90%')
    plt.plot(priorVarList, avgPosdevsSummaryMat[:, 4], 'o--', color='purple', label='15imp_60out - lower 90%')
    plt.plot(priorVarList, avgPosdevsSummaryMat[:, 5], 'o-', color='purple', label='15imp_60out - upper 90%')
    plt.plot(priorVarList, avgPosdevsSummaryMat[:, 6], 'o--', color='blue', label='15imp_100out - lower 90%')
    plt.plot(priorVarList, avgPosdevsSummaryMat[:, 7], 'o-', color='blue', label='15imp_100out - upper 90%')
    plt.ylim([0, 13])
    plt.title('90% Intervals on SIZE of Pos. Hessian Diagonal Values\nvs. Prior Variance, for 4 different system sizes',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Prior variance', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close()

    # Percentage of negative diagonals that are outlets
    '''
    fig, ax = plt.subplots(figsize=(8, 10), ncols=1)
    plt.plot(priorVarList, percOutSummaryMat[:, 0], 'o--', color='orange', label='5imp_20out - lower 90%')
    plt.plot(priorVarList, percOutSummaryMat[:, 1], 'o-', color='orange', label='5imp_20out - upper 90%')
    plt.plot(priorVarList, percOutSummaryMat[:, 2], 'o--', color='red', label='10imp_40out - lower 90%')
    plt.plot(priorVarList, percOutSummaryMat[:, 3], 'o-', color='red', label='10imp_40out - upper 90%')
    plt.plot(priorVarList, percOutSummaryMat[:, 4], 'o--', color='purple', label='15imp_60out - lower 90%')
    plt.plot(priorVarList, percOutSummaryMat[:, 5], 'o-', color='purple', label='15imp_60out - upper 90%')
    plt.plot(priorVarList, percOutSummaryMat[:, 6], 'o--', color='blue', label='15imp_100out - lower 90%')
    plt.plot(priorVarList, percOutSummaryMat[:, 7], 'o-', color='blue', label='15imp_100out - upper 90%')
    plt.ylim([0, 1])
    plt.title(
        '90% Intervals on % THAT ARE OUTLETS of Neg. Hessian Diag. Vals.\nvs. Prior Variance, for 4 different system sizes',
        fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Prior variance', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close()
    '''

    # Percentage of eigenvalues that are negative
    fig, ax = plt.subplots(figsize=(8, 10), ncols=1)
    plt.plot(priorVarList, percNegEigSummaryMat[:, 0], 'o--', color='orange', label='5imp_20out - lower 90%')
    plt.plot(priorVarList, percNegEigSummaryMat[:, 1], 'o-', color='orange', label='5imp_20out - upper 90%')
    plt.plot(priorVarList, percNegEigSummaryMat[:, 2], 'o--', color='red', label='10imp_40out - lower 90%')
    plt.plot(priorVarList, percNegEigSummaryMat[:, 3], 'o-', color='red', label='10imp_40out - upper 90%')
    plt.plot(priorVarList, percNegEigSummaryMat[:, 4], 'o--', color='purple', label='15imp_60out - lower 90%')
    plt.plot(priorVarList, percNegEigSummaryMat[:, 5], 'o-', color='purple', label='15imp_60out - upper 90%')
    plt.plot(priorVarList, percNegEigSummaryMat[:, 6], 'o--', color='blue', label='15imp_100out - lower 90%')
    plt.plot(priorVarList, percNegEigSummaryMat[:, 7], 'o-', color='blue', label='15imp_100out - upper 90%')
    plt.ylim([0, 1])
    plt.title(
        '90% Intervals on % NEG. EIGENVALUES of Hessian\nvs. Prior Variance, for 4 different system sizes',
        fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Prior variance', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close()

    # Distribution of size of eigenvalues
    fig, ax = plt.subplots(figsize=(8, 10), ncols=1)
    plt.plot(priorVarList, avgEigSummaryMat[:, 0], 'o--', color='orange', label='5imp_20out - lower 90%')
    plt.plot(priorVarList, avgEigSummaryMat[:, 1], 'o-', color='orange', label='5imp_20out - upper 90%')
    plt.plot(priorVarList, avgEigSummaryMat[:, 2], 'o--', color='red', label='10imp_40out - lower 90%')
    plt.plot(priorVarList, avgEigSummaryMat[:, 3], 'o-', color='red', label='10imp_40out - upper 90%')
    plt.plot(priorVarList, avgEigSummaryMat[:, 4], 'o--', color='purple', label='15imp_60out - lower 90%')
    plt.plot(priorVarList, avgEigSummaryMat[:, 5], 'o-', color='purple', label='15imp_60out - upper 90%')
    plt.plot(priorVarList, avgEigSummaryMat[:, 6], 'o--', color='blue', label='15imp_100out - lower 90%')
    plt.plot(priorVarList, avgEigSummaryMat[:, 7], 'o-', color='blue', label='15imp_100out - upper 90%')
    plt.ylim([0, 17])
    plt.title(
        '90% Intervals on SIZE OF EIGENVALUES of Hessian\nvs. Prior Variance, for 4 different system sizes',
        fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Prior variance', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close()

    return





if __name__ == '__main__':
    examiningLaplaceApprox()

