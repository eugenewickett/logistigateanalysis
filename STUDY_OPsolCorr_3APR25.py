
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from logistigate.logistigate import lossfunctions as lf

from orienteering.senegalsetup import *
from logistigate.logistigate import orienteering as opf

import scipy.optimize as spo
from scipy.optimize import milp
import scipy.special as sps

plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.family"] = "serif"

# Pull data from newly constructed CSV files
dept_df, regcost_mat, regNames, deptNames, manufNames, numReg, testdatadict = GetSenegalCSVData()
(numTN, numSN) = testdatadict['N'].shape  # For later use

PrintDataSummary(testdatadict)  # Cursory data check

# Set up logistigate dictionary
lgdict = util.initDataDict(testdatadict['N'], testdatadict['Y'])
lgdict.update({'TNnames': deptNames, 'SNnames': manufNames})

# Set up priors for SFP rates at nodes
SetupSenegalPriors(lgdict)

# Set up MCMC
lgdict['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 1000, 'delta': 0.4}
mcmcfilestr = os.path.join('orienteering', 'numpy_objects', 'draws')

RetrieveMCMCBatches(lgdict, 60, mcmcfilestr, maxbatchnum=99, rand=True, randseed=1122)

# Add boostrap-sampled sourcing vectors for non-tested test nodes; 20 is the avg number of tests per visited dept
AddBootstrapQ(lgdict, numboot=int(np.sum(lgdict['N'])/np.count_nonzero(np.sum(lgdict['Q'], axis=1))), randseed=44)
Q = lgdict['Q']

# Parameter dictionary for loss
paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN))

# Set up utility estimation parameter dictionary with desired truth and data draws
SetupUtilEstParamDict(lgdict, paramdict, 100000, 300, randseed=56)
util.print_param_checks(paramdict)  # Parameter check

# Set these parameters per the program described in the paper
batchcost, B, ctest = 0, 1400, 2
batchsize, bigM = B, B*ctest

dept_df_sort = dept_df.sort_values('Department')

FTEcostperday = 200
f_dept = np.array(dept_df_sort['DeptFixedCostDays'].tolist())*FTEcostperday  # Fixed district costs
f_reg = np.array(regcost_mat)*FTEcostperday  # Fixed region-to-region arc costs

# Set up optimization
optparamdict = {'batchcost':batchcost, 'budget':B, 'pertestcost':ctest, 'Mconstant':bigM, 'batchsize':batchsize,
                'deptfixedcostvec':f_dept, 'arcfixedcostmat': f_reg, 'reghqname':'Dakar', 'reghqind':0,
                'deptnames':deptNames, 'regnames':regNames, 'dept_df':dept_df_sort}

# What are the upper bounds for our allocation variables?
deptallocbds = opf.GetUpperBounds(optparamdict)
# Lower upper bounds to maximum of observed prior tests at any district
maxpriortests = int(np.max(np.sum(lgdict['N'],axis=1)))
deptallocbds = np.array([min(deptallocbds[i], maxpriortests) for i in range(deptallocbds.shape[0])])

print(deptNames[np.argmin(deptallocbds)], min(deptallocbds))
print(deptNames[np.argmax(deptallocbds)], max(deptallocbds))

interp_df = pd.read_csv(os.path.join('orienteering', 'csv_utility', 'interp_df_BASE.csv'))

paths_df = pd.read_csv(os.path.join('orienteering', 'csv_paths', 'paths_BASE_1400.csv'))

def GetPathSequenceAndCost(paths_df):
    """Retrieves optimization-ready lists pertaining to path sequences and costs"""
    seqlist = paths_df['Sequence'].copy()
    seqcostlist = paths_df['Cost'].copy()
    bindistaccessvectors = np.array(paths_df['DistAccessBinaryVec'].tolist())
    bindistaccessvectors = np.array([eval(x) for x in bindistaccessvectors])
    seqlist = seqlist.reset_index()
    seqlist = seqlist.drop(columns='index')
    seqcostlist = seqcostlist.reset_index()
    seqcostlist = seqcostlist.drop(columns='index')

    return seqlist, seqcostlist, bindistaccessvectors

seqlist_trim, seqcostlist_trim, bindistaccessvectors_trim = GetPathSequenceAndCost(paths_df)

def GetInterpVectors(interp_df):
    """Build needed interpolation vectors for use with relaxed program"""
    lvec, juncvec, m1vec, m2vec, bds, lovals, hivals = [], [], [], [], [], [], []
    for ind in range(interp_df.shape[0]):
        row = interp_df.iloc[ind]
        currBound, loval, hival = row['Bounds'], row['Util_lo'], row['Util_hi']
        # Get interpolation values
        _, _, l, k, m1, m2 = opf.GetTriangleInterpolation([0, 1, currBound], [0, loval, hival])
        lvec.append(l)
        juncvec.append(k)
        m1vec.append(m1)
        m2vec.append(m2)
        bds.append(currBound)
        lovals.append(loval)
        hivals.append(hival)

    return lvec, juncvec, m1vec, m2vec, bds, lovals, hivals

lvec, juncvec, m1vec, m2vec, bds, lovals, hivals = GetInterpVectors(interp_df)

numPath = paths_df.shape[0]

##################
# TODO: NEW STUFF STARTS HERE: HAVE TO REDESIGN THE LINEAR OPTIMIZATION PROGRAM TO INCLUDE VIRTUAL VARS
##################
# Weight matrix here
omegaMat = np.identity(numTN)

# Number of paths
numPath = paths_df.shape[0]

# todo: Variable vectors are in form (z, n, x, z_hat, n_hat)
#                                   [districts, allocations, paths, virtual districts, virtual allocations]

def GetVarBds(numTN, numPath, juncvec, interp_df):
    lbounds = np.concatenate((np.zeros(numTN * 3), np.zeros(numPath), np.zeros(numTN * 3)))
    ubounds = np.concatenate((np.ones(numTN),
                      np.array([juncvec[i] - 1 for i in range(numTN)]),
                      np.array(interp_df['Bounds'].tolist()) - np.array([juncvec[i] - 1 for i in range(numTN)]),
                      np.ones(numPath),
                      np.ones(numTN),
                      np.array([juncvec[i] - 1 for i in range(numTN)]),
                      np.array([bigM for i in range(numTN)])))
    return spo.Bounds(lbounds, ubounds)

optbounds = GetVarBds(numTN, numPath, juncvec, interp_df)

# Objective vector
def GetObjective(lvec, m1vec, m2vec, numPath):
    """Negative-ed as milp requires minimization"""
    return -np.concatenate((np.zeros((numTN * 3) + numPath), np.array(lvec), np.array(m1vec), np.array(m2vec)))

optobjvec = GetObjective(lvec, m1vec, m2vec, numPath)

# Constraints
def GetConstraints(optparamdict, juncvec, seqcostlist, bindistaccessvectors, omegamat):
    numTN, B, ctest = len(optparamdict['deptnames']), optparamdict['budget'], optparamdict['pertestcost']
    f_dept, bigM = optparamdict['deptfixedcostvec'], optparamdict['Mconstant']
    # Build lower and upper inequality values
    optconstrlower = np.concatenate((np.ones(numTN*4+1) * -np.inf, np.array([1]),
                                     np.ones(numTN*3)*-np.inf))
    optconstrupper = np.concatenate((np.array([B]), np.zeros(numTN*2), np.array(juncvec), np.zeros(numTN),
                                     np.array([1]),
                                     np.zeros(numTN), np.array(juncvec), np.zeros(numTN)))
    # Build A matrix, from left to right
    # Build z district binaries first
    optconstraintmat1 = np.vstack((f_dept, -bigM * np.identity(numTN), np.identity(numTN), 0 * np.identity(numTN),
                                   np.identity(numTN), np.zeros(numTN),
                                   0 * np.identity(numTN), 0 * np.identity(numTN), 0 * np.identity(numTN)))
    # n^' matrices
    optconstraintmat2 = np.vstack((ctest * np.ones(numTN), np.identity(numTN), -np.identity(numTN),
                                   np.identity(numTN), 0 * np.identity(numTN), np.zeros(numTN),
                                   -omegamat, 0 * np.identity(numTN), 0 * np.identity(numTN)))
    # n^'' matrices
    optconstraintmat3 = np.vstack((ctest * np.ones(numTN), np.identity(numTN), -np.identity(numTN),
                                   0 * np.identity(numTN), 0 * np.identity(numTN), np.zeros(numTN),
                                   -omegamat, 0 * np.identity(numTN), 0 * np.identity(numTN)))
    # path matrices
    optconstraintmat4 = np.vstack((np.array(seqcostlist).T, np.zeros((numTN * 3, numPath)),
                                   (-bindistaccessvectors).T, np.ones(numPath),
                                    np.zeros((numTN * 3, numPath))))
    # z_hat matrices
    optconstraintmat5 = np.vstack((np.zeros(((numTN * 6) + 2, numTN)), np.identity(numTN)))
    # n_hat^' matrices
    optconstraintmat6 = np.vstack((np.zeros(((numTN*4)+2, numTN)),
                                   np.identity(numTN), np.identity(numTN), -np.identity(numTN)))
    # n_hat^'' matrices
    optconstraintmat7 = np.vstack((np.zeros(((numTN*4)+2, numTN)),
                                   np.identity(numTN), 0 * np.identity(numTN), -np.identity(numTN)))

    optconstraintmat = np.hstack((optconstraintmat1, optconstraintmat2, optconstraintmat3, optconstraintmat4,
                                  optconstraintmat5, optconstraintmat6, optconstraintmat7))
    return spo.LinearConstraint(optconstraintmat, optconstrlower, optconstrupper)

optconstraints = GetConstraints(optparamdict, juncvec, seqcostlist_trim, bindistaccessvectors_trim, omegaMat)

def GetIntegrality(optobjvec):
    return np.ones_like(optobjvec)

# Define integrality for all variables
optintegrality = GetIntegrality(optobjvec)

# Solve
spoOutput = milp(c=optobjvec, constraints=optconstraints, integrality=optintegrality, bounds=optbounds)
initsoln, initsoln_obj = spoOutput.x, spoOutput.fun*-1

def scipytoallocation(spo_x, distNames, regNames, seqlist_trim_df, eliminateZeros=False):
    """function for turning scipy solution into something interpretable"""
    tnnum = len(distNames)
    z = np.round(spo_x[:tnnum])
    n1 = np.round(spo_x[tnnum:tnnum * 2])
    n2 = np.round(spo_x[tnnum * 2:tnnum * 3])
    x = np.round(spo_x[tnnum * 3:]) # Solver sometimes gives non-integer solutions
    path = eval(seqlist_trim_df.iloc[np.where(x == 1)[0][0],0])
    # Print district name with key solution elements
    for distind, distname in enumerate(distNames):
        if not eliminateZeros:
            print(str(distname)+':', str(int(z[distind])), str(int(n1[distind])), str(int(n2[distind])))
        else:  # Remove zeros
            if int(z[distind])==1:
                print(str(distname)+ ':', str(int(z[distind])), str(int(n1[distind])), str(int(n2[distind])))
    pathstr = ''
    for regind in path:
        pathstr = pathstr + str(regNames[regind]) + ' '
    print('Path: '+ pathstr)
    return

scipytoallocation(initsoln, deptNames, regNames, seqlist_trim, eliminateZeros=True)

'''SOLUTION UNDER ORIGINAL IP-RP; 
Bambey: 1 0 11
Birkilane: 1 0 8
Dakar: 1 17 25
Fatick: 1 0 14
Foundiougne: 1 0 10
Gossas: 1 0 9
Guinguineo: 1 0 11
Kaffrine: 1 0 11
Kaolack: 1 3 39
Keur Massar: 1 0 31
Koungheul: 1 9 36
Malem Hoddar: 1 0 10
Mbacke: 1 0 8
Nioro du Rip: 1 0 15
Pikine: 1 0 9
Path: Dakar Fatick Kaolack Kaffrine Diourbel

util: 5.636652804231508
util_CI: (5.289598860601801, 5.9837067478612145)
'''
def scipysoltoallocvec(spo_x, tnnum):
    return spo_x[tnnum:tnnum*2] + spo_x[tnnum*2:tnnum*3]

n = scipysoltoallocvec(initsoln, numTN)
util, util_CI = sampf.getUtilityEstimate_parallel(n, lgdict, paramdict, zlevel=0.95)
print(util)
print(util_CI)

def sampling_plan_loss_list_importance(design, numtests, priordatadict, paramdict, numimportdraws,
                                       numdatadrawsforimportance=1000, extremadelta=0.01,
                                       preservevar=True, preservevarzlevel=0.95):
    """
    Produces a list of sampling plan losses, a la sampling_plan_loss_list(). This method uses the importance
    sampling approach, using numdatadrawsforimportance draws to produce an 'average' data set. An MCMC set of
    numimportdraws is produced assuming this average data set; this MCMC set should be closer to the important region
    of SFP rates for this design. The importance weights can produce extrema that increase loss variance and bias;
    parameter extremadelta indicates the weight quantile for which the corresponding MCMC draws are removed
    from loss calculations, introducing some bias in the draws used for the estimate in order to eliminate the estimate
    bias stemming from very large importance weights. Removing extrema artificially reduces the utility estimate
    variance; option preservevar, when True, returns preserve_CI, the interval when using all draws. The width of this
    interval can then be transferred onto the estimate obtained when removing the extrema.

    design: sampling probability vector along all test nodes/traces
    numtests: test budget
    priordatadict: logistigate data dictionary capturing known data
    paramdict: parameter dictionary containing a loss matrix, truth and data MCMC draws, and an optional method for
        rounding the design to an integer allocation
    numimportdraws: number of MCMC draws to generate in the importance zone
    numdatadrawsforimportance: number of data sets to simulate for establishing the importance data set average
    extremadelta: proportion of importance weights to drop from consideration, starting with the largest
    """
    if 'roundalg' in paramdict:  # Set default rounding algorithm for plan
        roundalg = paramdict['roundalg'].copy()
    else:
        roundalg = 'lo'
    # Initialize samples to be drawn from traces, per the design, using a rounding algorithm
    sampMat = util.generate_sampling_array(design, numtests, roundalg)
    (numTN, numSN), Q, s, r = priordatadict['N'].shape, priordatadict['Q'], priordatadict['diagSens'], priordatadict['diagSpec']
    # Identify an 'average' data set that will help establish the important region for importance sampling
    importancedatadrawinds = np.random.choice(np.arange(paramdict['datadraws'].shape[0]),
                                          size = numdatadrawsforimportance, # Oversample if needed
                                          replace = paramdict['datadraws'].shape[0] < numdatadrawsforimportance)
    importancedatadraws = paramdict['datadraws'][importancedatadrawinds]
    zMatData = util.zProbTrVec(numSN, importancedatadraws, sens=s, spec=r)  # Probs. using data draws
    NMat = np.moveaxis(np.array([np.random.multinomial(sampMat[tnInd], Q[tnInd], size=numdatadrawsforimportance)
                                 for tnInd in range(numTN)]), 1, 0).astype(int)
    YMat = np.random.binomial(NMat, zMatData)
    # Get average rounded data set from these few draws
    NMatAvg, YMatAvg = np.round(np.average(NMat, axis=0)).astype(int), np.round(np.average(YMat, axis=0)).astype(int)
    # Add these data to a new data dictionary and generate a new set of MCMC draws
    impdict = priordatadict.copy()
    impdict['N'], impdict['Y'] = priordatadict['N'] + NMatAvg, priordatadict['Y'] + YMatAvg
    # Generate a new MCMC importance set
    impdict['numPostSamples'] = numimportdraws
    impdict = methods.GeneratePostSamples(impdict, maxTime=5000)

    # Get simulated data likelihoods - don't normalize
    numdatadraws =  paramdict['datadraws'].shape[0]
    zMatTruth = util.zProbTrVec(numSN, impdict['postSamples'], sens=s, spec=r)  # Matrix of SFP probabilities, as a function of SFP rate draws
    zMatData = util.zProbTrVec(numSN, paramdict['datadraws'], sens=s, spec=r)  # Probs. using data draws
    NMat = np.moveaxis(np.array([np.random.multinomial(sampMat[tnInd], Q[tnInd], size=numdatadraws)
                                 for tnInd in range(numTN)]), 1, 0).astype(int)
    YMat = np.random.binomial(NMat, zMatData)
    tempW = np.zeros(shape=(numimportdraws, numdatadraws))
    for snInd in range(numSN):  # Loop through each SN and TN combination; DON'T vectorize as resulting matrix can be too big
        for tnInd in range(numTN):
            if sampMat[tnInd] > 0 and Q[tnInd, snInd] > 0:  # Save processing by only looking at feasible traces
                # Get zProbs corresponding to current trace
                bigZtemp = np.transpose(
                    np.reshape(np.tile(zMatTruth[:, tnInd, snInd], numdatadraws), (numdatadraws, numimportdraws)))
                bigNtemp = np.reshape(np.tile(NMat[:, tnInd, snInd], numimportdraws), (numimportdraws, numdatadraws))
                bigYtemp = np.reshape(np.tile(YMat[:, tnInd, snInd], numimportdraws), (numimportdraws, numdatadraws))
                combNYtemp = np.reshape(np.tile(sps.comb(NMat[:, tnInd, snInd], YMat[:, tnInd, snInd]), numimportdraws),
                                        (numimportdraws, numdatadraws))
                tempW += (bigYtemp * np.log(bigZtemp)) + ((bigNtemp - bigYtemp) * np.log(1 - bigZtemp)) + np.log(
                    combNYtemp)
    Wimport = np.exp(tempW)

    # Get risk matrix
    Rimport = lf.risk_check_array(impdict['postSamples'], paramdict['riskdict'])
    # Get critical ratio
    q = paramdict['scoredict']['underestweight'] / (1 + paramdict['scoredict']['underestweight'])

    # Get likelihood weights WRT original data set: p(gamma|d_0)
    zMatImport = util.zProbTrVec(numSN, impdict['postSamples'], sens=s, spec=r)  # Matrix of SFP probabilities along each trace
    NMatPrior, YMatPrior = priordatadict['N'], priordatadict['Y']
    Vimport = np.zeros(shape = numimportdraws)
    for snInd in range(numSN):  # Loop through each SN and TN combination; DON'T vectorize as resulting matrix can be too big
        for tnInd in range(numTN):
            if NMatPrior[tnInd, snInd] > 0:
                bigZtemp = np.transpose(
                    np.reshape(np.tile(zMatImport[:, tnInd, snInd], 1), (1, numimportdraws)))
                bigNtemp = np.reshape(np.tile(NMatPrior[tnInd, snInd], numimportdraws), (numimportdraws, 1))
                bigYtemp = np.reshape(np.tile(YMatPrior[tnInd, snInd], numimportdraws), (numimportdraws, 1))
                combNYtemp = np.reshape(np.tile(sps.comb(NMatPrior[tnInd, snInd], YMatPrior[tnInd, snInd]),
                                                numimportdraws), (numimportdraws, 1))
                Vimport += np.squeeze( (bigYtemp * np.log(bigZtemp)) + ((bigNtemp - bigYtemp) * np.log(1 - bigZtemp)) + np.log(
                    combNYtemp))
    Vimport = np.exp(Vimport)

    # Get likelihood weights WRT average data set: p(gamma|d_0, d_imp)
    NMatPrior, YMatPrior = impdict['N'].copy(), impdict['Y'].copy()
    Uimport = np.zeros(shape=numimportdraws)
    for snInd in range(
            numSN):  # Loop through each SN and TN combination; DON'T vectorize as resulting matrix can be too big
        for tnInd in range(numTN):
            if NMatPrior[tnInd, snInd] > 0:
                bigZtemp = np.transpose(
                    np.reshape(np.tile(zMatImport[:, tnInd, snInd], 1), (1, numimportdraws)))
                bigNtemp = np.reshape(np.tile(NMatPrior[tnInd, snInd], numimportdraws), (numimportdraws, 1))
                bigYtemp = np.reshape(np.tile(YMatPrior[tnInd, snInd], numimportdraws), (numimportdraws, 1))
                combNYtemp = np.reshape(np.tile(sps.comb(NMatPrior[tnInd, snInd], YMatPrior[tnInd, snInd]),
                                                numimportdraws), (numimportdraws, 1))
                Uimport += np.squeeze(
                    (bigYtemp * np.log(bigZtemp)) + ((bigNtemp - bigYtemp) * np.log(1 - bigZtemp)) + np.log(
                        combNYtemp))
    Uimport = np.exp(Uimport)

    # Importance likelihood ratio for importance draws
    VoverU = (Vimport / Uimport)

    # Compile list of optima
    # Use minslist WITHOUT extrema removed if preserving variance
    if preservevar==True:
        print('Getting preserved variance...')
        minslist = []
        for j in range(Wimport.shape[1]):
            tempwtarray = Wimport[:, j] * VoverU * numimportdraws / np.sum(Wimport[:, j] * VoverU)
            # Don't remove any extrema
            tempremoveinds = np.where(tempwtarray > np.quantile(tempwtarray, 1))
            tempwtarray = np.delete(tempwtarray, tempremoveinds)
            tempwtarray = tempwtarray / np.sum(tempwtarray)  # Normalize
            tempimportancedraws = np.delete(impdict['postSamples'], tempremoveinds, axis=0)
            tempRimport = np.delete(Rimport, tempremoveinds, axis=0)
            est = sampf.bayesest_critratio(tempimportancedraws, tempwtarray, q)
            minslist.append(sampf.cand_obj_val(est, tempimportancedraws, tempwtarray, paramdict, tempRimport))
        # Get original variance
        _, preserve_CI = sampf.process_loss_list(minslist, zlevel=preservevarzlevel)
    else:
        preserve_CI = np.empty(0)

    if extremadelta > 0:  # Only regenerate minslist if extremadelta exceeds zero
        print('Getting estimate with extrema removed...')
        minslist = []
        for j in range(Wimport.shape[1]):
            tempwtarray = Wimport[:, j] * VoverU * numimportdraws / np.sum(Wimport[:, j] * VoverU)
            # Remove inds for top extremadelta of weights
            tempremoveinds = np.where(tempwtarray>np.quantile(tempwtarray, 1-extremadelta))
            tempwtarray = np.delete(tempwtarray, tempremoveinds)
            tempwtarray = tempwtarray/np.sum(tempwtarray)
            tempimportancedraws = np.delete(impdict['postSamples'], tempremoveinds, axis=0)
            tempRimport = np.delete(Rimport, tempremoveinds, axis=0)
            est = sampf.bayesest_critratio(tempimportancedraws, tempwtarray, q)
            minslist.append(sampf.cand_obj_val(est, tempimportancedraws, tempwtarray, paramdict, tempRimport))
    elif extremadelta == -1:  # Identify extrema removal that minimizes the resulting loss estimate
        print('Getting estimate with extrema removed, while fitting to lowest possible estimate...')
        estincr = True  # Boolean tracking if the current estimate is decreasing
        currextremadelta, currminsavg = 0.0, 0.0
        stepint = max(0.0005, 5/numimportdraws)  # Step interval for trying new extrema deltas
        while estincr:
            currextremadelta += stepint
            print('Current extrema delta: ' + str(currextremadelta))
            currminslist = []
            for j in range(Wimport.shape[1]):
                tempwtarray = Wimport[:, j] * VoverU * numimportdraws / np.sum(Wimport[:, j] * VoverU)
                # Remove inds for top extremadelta of weights
                tempremoveinds = np.where(tempwtarray > np.quantile(tempwtarray, 1 - currextremadelta))
                tempwtarray = np.delete(tempwtarray, tempremoveinds)
                tempwtarray = tempwtarray / np.sum(tempwtarray)
                tempimportancedraws = np.delete(impdict['postSamples'], tempremoveinds, axis=0)
                tempRimport = np.delete(Rimport, tempremoveinds, axis=0)
                est = sampf.bayesest_critratio(tempimportancedraws, tempwtarray, q)
                currminslist.append(sampf.cand_obj_val(est, tempimportancedraws, tempwtarray, paramdict, tempRimport))
            print('Current loss: ' + str(np.average(currminslist)))
            if np.average(currminslist) < currminsavg:
                minslist = currminslist.copy()
                estincr = False
            else:
                currminsavg = np.average(currminslist)

    return minslist, preserve_CI

util, util_CI = sampf.getImportanceUtilityEstimate(n, lgdict, paramdict, 20000, extremadelta=-1, zlevel=0.95)

########
# Now update omegaMat and see what happens
########
# What is the ratio of importance of learning more about SNs?
maxratio = 1/2

# Build Jaccard index for each pair of districts
Qbin = np.ceil(Q)  # binary matrix identifying *any* sourcing between TN and SN
J = np.zeros((numTN, numTN))
for i in range(numTN):
    for j in range(numTN):
        s1 = np.nonzero(Qbin[i])
        s2 = np.nonzero(Qbin[j])
        J[i, j] = np.intersect1d(s1, s2).shape[0] / np.union1d(s1, s2).shape[0]

Jadj = J * maxratio  # adjust for maximum learning importance
Jadj = Jadj + np.identity(numTN)*(1-maxratio)  # adjust for diagonals to be 1
omegaMat = Jadj

optconstraints = GetConstraints(optparamdict, juncvec, seqcostlist_trim, bindistaccessvectors_trim, omegaMat)
spoOutput = milp(c=optobjvec, constraints=optconstraints, integrality=optintegrality, bounds=optbounds)
initsoln, initsoln_obj = spoOutput.x, spoOutput.fun*-1
scipytoallocation(initsoln, deptNames, regNames, seqlist_trim, eliminateZeros=True)

''' SOLUTION WITH INITIAL JACCARD INDEX; TOO MUCH VALUE FROM SIMPLY INCREASING TESTS AT REACHABLE LOCATIONS
Dakar: 1 26 25
Fatick: 1 14 67
Foundiougne: 1 10 71
Gossas: 1 9 72
Guediawaye: 1 51 30
Keur Massar: 1 31 50
Pikine: 1 9 72
Path: Dakar Fatick
util: 6.521669243524179
util_CI: (6.158360526804159, 6.884977960244199)
'''
n = scipysoltoallocvec(initsoln, numTN)
util, util_CI = sampf.getUtilityEstimate_parallel(n, lgdict, paramdict, zlevel=0.95)
print(util)
print(util_CI)

### TRY Q NORM
QnormMat = np.zeros((numTN, numTN))
for i in range(numTN):
    for j in range(numTN):
        QnormMat[i, j] = maxratio * (1 - (np.linalg.norm(Q[i]-Q[j])/np.sqrt(2)))

QnormMat = QnormMat + np.identity(numTN)*(1-maxratio)  # adjust for diagonals to be 1

omegaMat = QnormMat

optconstraints = GetConstraints(optparamdict, juncvec, seqcostlist_trim, bindistaccessvectors_trim, omegaMat)
spoOutput = milp(c=optobjvec, constraints=optconstraints, integrality=optintegrality, bounds=optbounds)
initsoln, initsoln_obj = spoOutput.x, spoOutput.fun*-1
scipytoallocation(initsoln, deptNames, regNames, seqlist_trim, eliminateZeros=True)

'''
Bambey: 1 9 70
Dakar: 1 55 25
Diourbel: 1 77 4
Guediawaye: 1 50 30
Keur Massar: 1 11 50
Mbacke: 1 8 73
Pikine: 1 8 72
Path: Dakar Diourbel
 
'''
n = scipysoltoallocvec(initsoln, numTN)
util, util_CI = sampf.getUtilityEstimate_parallel(n, lgdict, paramdict, zlevel=0.95)

# What if correlation value is too high? Try Jaccard index with a smaller maxratio
maxratio = 1/100
Jadj = J * maxratio  # adjust for maximum learning importance
Jadj = Jadj + np.identity(numTN)*(1-maxratio)  # adjust for diagonals to be 1
omegaMat = Jadj

optconstraints = GetConstraints(optparamdict, juncvec, seqcostlist_trim, bindistaccessvectors_trim, omegaMat)
spoOutput = milp(c=optobjvec, constraints=optconstraints, integrality=optintegrality, bounds=optbounds)
initsoln, initsoln_obj = spoOutput.x, spoOutput.fun*-1
scipytoallocation(initsoln, deptNames, regNames, seqlist_trim, eliminateZeros=True)

n = scipysoltoallocvec(initsoln, numTN)
util, util_CI = sampf.getUtilityEstimate_parallel(n, lgdict, paramdict, zlevel=0.95)

