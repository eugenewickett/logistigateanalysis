import numpy as np
import scipy.optimize as spo
import scipy.stats as spstat
import scipy.special as sps
import logistigate.logistigate.utilities as util # Pull from the submodule "develop" branch

# Workaround for the 'methods' file not being able to locate the 'mcmcsamplers' folder for importing
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, 'logistigate','logistigate')))

import logistigate.logistigate.methods as methods # Pull from the submodule "develop" branch

def update(): # CURRENTLY NOT UPDATING AS EXPECTED
    import git
    repo = git.Repo('C:/Users/eugen/OneDrive/Documents/EAGER Project/Simulator/logistigate-analysis')
    for submodule in repo.submodules:
        submodule.update(init=True)
    return

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
    newSysDict.update({'prior': methods.prior_normal(var=5)}) # Set prior variance here
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
    soln_hess = methods.Tracked_NegLogPost_Hess(soln_trans, newSysDict['N'], newSysDict['Y'],
                                            newSysDict['diagSens'], newSysDict['diagSpec'],
                                            prior=newSysDict['prior'])
    print(soln_jac) # Gradient seems within tolerance of 0
    print(np.diag(soln_hess))

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

    return


def FormEstimates(dataTblDict, retOptStatus=False, printUpdate=True):
    '''
    Takes a data input dictionary and returns an estimate dictionary using Laplace approximation.
    The L-BFGS-B method of the SciPy Optimizer is used to maximize the posterior log-likelihood,
    warm-started using random points via the prior.

    INPUTS
    ------
    dataTblDict should be a dictionary with the following keys:
        type: string
            'Tracked' or 'Untracked'
        N, Y: Numpy array
            If Tracked, it should be a matrix of size (outletNum, importerNum).
            If Untracked, it should a vector of size (outletNum).
            N is for the number of total tests conducted, Y is for the number of
            positive tests.
        transMat: Numpy 2-D array
            Matrix rows/columns should signify outlets/importers; values should
            be between 0 and 1, and rows must sum to 1. Required for Untracked.
        outletNames, importerNames: list of strings
            Should correspond to the order of the transition matrix
        diagSens, diagSpec: float
            Diagnostic characteristics for the data compiled in dataTbl
        prior: prior Class object
            Prior object for use with the posterior likelihood, as well as for warm-starting
    OUTPUTS
    -------
    Returns an estimate dictionary containing the following keys:
        impEst:    Maximizers of posterior likelihood for importer echelon
        outEst:    Maximizers of posterior likelihood for outlet echelon
        90upper_imp, 90lower_imp, 95upper_imp, 95lower_imp,
        99upper_imp, 99lower_imp, 90upper_out, 90lower_out,
        95upper_out, 95lower_out, 99upper_out, 99lower_out:
                   Upper and lower values for the 90%, 95%, and 99%
                   intervals on importer and outlet aberration rates
        hess:      Hessian matrix at the maximum
    '''
    # CHECK THAT ALL NECESSARY KEYS ARE IN THE INPUT DICTIONARY
    if not all(key in dataTblDict for key in ['type', 'N', 'Y', 'outletNames', 'importerNames',
                                              'diagSens', 'diagSpec', 'prior']):
        print('The input dictionary does not contain all required information for the Laplace approximation.' +
              ' Please check and try again.')
        return {}
    if printUpdate:
        print('Generating estimates and confidence intervals...')

    outDict = {}
    N, Y = dataTblDict['N'], dataTblDict['Y']
    Sens, Spec = dataTblDict['diagSens'], dataTblDict['diagSpec']
    prior = dataTblDict['prior']
    if dataTblDict['type'] == 'Tracked':
        (numOut, numImp) = N.shape
    elif dataTblDict['type'] == 'Untracked':
        transMat = dataTblDict['transMat']
        (numOut, numImp) = transMat.shape

    beta0_List = []
    for sampNum in range(10):  # Choose 10 random samples from the prior
        beta0_List.append(prior.rand(numImp + numOut))

    # Loop through each possible initial point and store the optimal solution likelihood values
    likelihoodsList = []
    solsList = []
    if retOptStatus:
        OptStatusList = []
    bds = spo.Bounds(np.zeros(numImp + numOut) - 8, np.zeros(numImp + numOut) + 8)
    if dataTblDict['type'] == 'Tracked':
        for curr_beta0 in beta0_List:
            opVal = spo.minimize(methods.Tracked_NegLogPost, curr_beta0,
                                 args=(N, Y, Sens, Spec, prior), method='L-BFGS-B',
                                 jac=methods.Tracked_NegLogPost_Grad,
                                 options={'disp': False}, bounds=bds)
            likelihoodsList.append(opVal.fun)
            solsList.append(opVal.x)
            if retOptStatus:
                OptStatusList.append(opVal.status)
        best_x = solsList[np.argmin(likelihoodsList)]
        hess = methods.Tracked_NegLogPost_Hess(best_x, N, Y, Sens, Spec, prior)
    elif dataTblDict['type'] == 'Untracked':
        for curr_beta0 in beta0_List:
            opVal = spo.minimize(methods.Untracked_NegLogPost, curr_beta0,
                                 args=(N, Y, Sens, Spec, transMat, prior),
                                 method='L-BFGS-B', jac=methods.Untracked_NegLogPost_Grad,
                                 options={'disp': False}, bounds=bds)
            likelihoodsList.append(opVal.fun)
            solsList.append(opVal.x)
            if retOptStatus:
                OptStatusList.append(opVal.status)
        best_x = solsList[np.argmin(likelihoodsList)]
        hess = methods.Untracked_NegLogPost_Hess(best_x, N, Y, Sens, Spec, transMat, prior)
    # Generate confidence intervals
    impEst = sps.expit(best_x[:numImp])
    outEst = sps.expit(best_x[numImp:])
    hessinv = np.linalg.pinv(hess)  # Pseudo-inverse of the Hessian
    hInvs = [i if i >= 0 else i * -1 for i in np.diag(hessinv)]
    z90, z95, z99 = spstat.norm.ppf(0.95), spstat.norm.ppf(0.975), spstat.norm.ppf(0.995)
    imp_Int90, imp_Int95, imp_Int99 = z90 * np.sqrt(hInvs[:numImp]), z95 * np.sqrt(hInvs[:numImp]), z99 * np.sqrt(
        hInvs[:numImp])
    out_Int90, out_Int95, out_Int99 = z90 * np.sqrt(hInvs[numImp:]), z95 * np.sqrt(hInvs[numImp:]), z99 * np.sqrt(
        hInvs[numImp:])
    outDict['90upper_imp'] = sps.expit(best_x[:numImp] + imp_Int90)
    outDict['90lower_imp'] = sps.expit(best_x[:numImp] - imp_Int90)
    outDict['95upper_imp'] = sps.expit(best_x[:numImp] + imp_Int95)
    outDict['95lower_imp'] = sps.expit(best_x[:numImp] - imp_Int95)
    outDict['99upper_imp'] = sps.expit(best_x[:numImp] + imp_Int99)
    outDict['99lower_imp'] = sps.expit(best_x[:numImp] - imp_Int99)
    outDict['90upper_out'] = sps.expit(best_x[numImp:] + out_Int90)
    outDict['90lower_out'] = sps.expit(best_x[numImp:] - out_Int90)
    outDict['95upper_out'] = sps.expit(best_x[numImp:] + out_Int95)
    outDict['95lower_out'] = sps.expit(best_x[numImp:] - out_Int95)
    outDict['99upper_out'] = sps.expit(best_x[numImp:] + out_Int99)
    outDict['99lower_out'] = sps.expit(best_x[numImp:] - out_Int99)
    outDict['impEst'], outDict['outEst'] = impEst, outEst
    outDict['hess'], outDict['hessinv'] = hess, hessinv
    if retOptStatus:
        outDict['optStatus'] = OptStatusList
    if printUpdate:
        print('Estimates and confidence intervals generated')

    return outDict

if __name__ == '__main__':
    examiningLaplaceApprox()


