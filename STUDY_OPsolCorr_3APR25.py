
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from logistigate.logistigate import lossfunctions as lf
from logistigate.logistigate import utilities as util

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
SetupUtilEstParamDict(lgdict, paramdict, 50000, 500, randseed=56)
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

util: 2.494192171959302
util_CI: (2.483419894433508, 2.5049644494850956)
'''
def scipysoltoallocvec(spo_x, tnnum):
    return spo_x[tnnum:tnnum*2] + spo_x[tnnum*2:tnnum*3]

def MakeAllocationHeatMap(n, optparamdict, plotTitle='', cmapstr='gray',
                          vlist='NA', sortby='districtcost'):
    """Generate an allocation heat map"""
    distNames = optparamdict['deptnames']
    # Sort regions by distance to HQ, taken to be row 0
    reg_sortinds = np.argsort(optparamdict['arcfixedcostmat'][0])
    regNames_sort = [optparamdict['regnames'][x] for x in reg_sortinds]
    # District list for each region of regNames_sort
    dist_df = optparamdict['dept_df']
    distinreglist = []
    for currReg in regNames_sort:
        currDists = opf.GetDeptChildren(currReg, dist_df)
        # todo: CAN SORT BY OTHER THINGS HERE
        currDistFixedCosts = [dist_df.loc[dist_df['Department'] == x]['DeptFixedCostDays'].to_numpy()[0] for x in currDists]
        distinreglist.append([currDists[x] for x in np.argsort(currDistFixedCosts)])
    listlengths = [len(x) for x in distinreglist]
    maxdistnum = max(listlengths)
    # Initialize storage matrix
    dispmat = np.zeros((len(regNames_sort), maxdistnum))

    for distind, curralloc in enumerate(n):
        currDistName = distNames[distind]
        currRegName = opf.GetRegion(currDistName, dist_df)
        regmatind = regNames_sort.index(currRegName)
        distmatind = distinreglist[regmatind].index(currDistName)
        dispmat[regmatind, distmatind] = curralloc
    if vlist != 'NA':
        fig, ax = plt.subplots()
        img = plt.imshow(dispmat, cmap=cmapstr, interpolation='nearest',
                        vmin=vlist[0], vmax=vlist[1])
        plt.colorbar(img)
    else:
        plt.imshow(dispmat, cmap=cmapstr, interpolation='nearest')
    plt.ylabel('Ranked distance from HQ region')
    plt.xlabel('Ranked distance from regional capital')
    plt.title(plotTitle)
    plt.tight_layout()
    plt.show()
    return

n = scipysoltoallocvec(initsoln, numTN)
MakeAllocationHeatMap(n, optparamdict, plotTitle='Base IP-RP solution', cmapstr='Blues')

util, util_CI = sampf.getImportanceUtilityEstimate(n, lgdict, paramdict, numimportdraws=20000,
                                                   extremadelta=-1, preservevar=False)
print(util)
print(util_CI)


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
util: 1.717267016379056
util_CI: (1.7069057761018946, 1.7276282566562173)
'''
n = scipysoltoallocvec(initsoln, numTN)
util, util_CI = sampf.getImportanceUtilityEstimate(n, lgdict, paramdict, numimportdraws=20000,
                                                   extremadelta=-1, preservevar=False)
print(util)
print(util_CI)

###############
### TRY Q NORM
###############
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
util: 1.5969955540687888
util_CI: (1.5872917331923038, 1.6066993749452738)
'''
n = scipysoltoallocvec(initsoln, numTN)
util, util_CI = sampf.getImportanceUtilityEstimate(n, lgdict, paramdict, numimportdraws=20000,
                                                   extremadelta=-1, preservevar=False)
print(util)
print(util_CI)

# What if correlation value is too high? Try Jaccard index with a smaller maxratio
maxratio = 1/50
Jadj = J * maxratio  # adjust for maximum learning importance
Jadj = Jadj + np.identity(numTN)*(1-maxratio)  # adjust for diagonals to be 1
omegaMat = Jadj

optconstraints = GetConstraints(optparamdict, juncvec, seqcostlist_trim, bindistaccessvectors_trim, omegaMat)
spoOutput = milp(c=optobjvec, constraints=optconstraints, integrality=optintegrality, bounds=optbounds)
initsoln, initsoln_obj = spoOutput.x, spoOutput.fun*-1
scipytoallocation(initsoln, deptNames, regNames, seqlist_trim, eliminateZeros=True)

n = scipysoltoallocvec(initsoln, numTN)
util, util_CI = sampf.getUtilityEstimate_parallel(n, lgdict, paramdict, zlevel=0.95)

###############
### TRY NEW MEASURE FOR J(): MIN(Q[i,b],Q[j,b], for all b)
###############
maxratio = 1/2

# Build Jaccard index for each pair of districts
J = np.zeros((numTN, numTN))
for i in range(numTN):
    for j in range(numTN):
        J[i, j] = np.sum(np.min(np.vstack((Q[i], Q[j])), axis=0))

Jadj = J * maxratio  # adjust for maximum learning importance
Jadj = Jadj + np.identity(numTN)*(1-maxratio)  # adjust for diagonals to be 1
omegaMat = Jadj

optconstraints = GetConstraints(optparamdict, juncvec, seqcostlist_trim, bindistaccessvectors_trim, omegaMat)
spoOutput = milp(c=optobjvec, constraints=optconstraints, integrality=optintegrality, bounds=optbounds)
initsoln, initsoln_obj = spoOutput.x, spoOutput.fun*-1
scipytoallocation(initsoln, deptNames, regNames, seqlist_trim, eliminateZeros=True)

''' SOLUTION WITH 2ND JACCARD INDEX; STRONGLY RESEMBLES QNORM SOLUTION
Bambey: 1 0 67
Dakar: 1 51 25
Diourbel: 1 77 4
Guediawaye: 1 51 30
Keur Massar: 1 25 50
Mbacke: 1 8 73
Pikine: 1 9 72
Path: Dakar Diourbel
util: 1.50774338106328
util_CI: (1.4984872911490843, 1.516999470977476)
'''
n = scipysoltoallocvec(initsoln, numTN)
util, util_CI = sampf.getImportanceUtilityEstimate(n, lgdict, paramdict, numimportdraws=20000,
                                                   extremadelta=-1, preservevar=False)
print(util)
print(util_CI)

###############
### USE NEW J() MEASURE; ADD IN NEW STRUCTURE FOR Z CONSTRAINTS, WHERE Z VALUE IS GAINED ONLY IF ACTUALLY VISITED
###############

def GetObjective_nozhat(lvec, m1vec, m2vec, numPath):
    """Negative-ed as milp requires minimization"""
    return -np.concatenate((np.array(lvec), np.zeros((numTN * 3) + numPath),  np.array(m1vec), np.array(m2vec)))

optobjvec = GetObjective_nozhat(lvec, m1vec, m2vec, numPath)
optconstraints = GetConstraints(optparamdict, juncvec, seqcostlist_trim, bindistaccessvectors_trim, omegaMat)
spoOutput = milp(c=optobjvec, constraints=optconstraints, integrality=optintegrality, bounds=optbounds)
initsoln, initsoln_obj = spoOutput.x, spoOutput.fun*-1
scipytoallocation(initsoln, deptNames, regNames, seqlist_trim, eliminateZeros=True)
''' SOLUTION WITH UPDATED Z FORMULATION; PUSHED IT A LITTLE FARTHER OUT, BUT STILL FOCUSED ON INCREASING TESTS
Dakar: 1 56 25
Fatick: 1 13 67
Foundiougne: 1 10 71
Gossas: 1 4 72
Guediawaye: 1 50 30
Keur Massar: 1 8 50
Pikine: 1 9 72
Path: Dakar Fatick
util: 1.7862346173516759
util_CI: (1.7768626906763334, 1.7956065440270184)
'''
n = scipysoltoallocvec(initsoln, numTN)
util, util_CI = sampf.getImportanceUtilityEstimate(n, lgdict, paramdict, numimportdraws=20000,
                                                   extremadelta=-1, preservevar=False)
print(util)
print(util_CI)

###############
### USE NEW J() MEASURE AND NEW Z STRUCTURE;
# ADD: ONLY GETTING INDICATOR VARS IF PREVIOUSLY VISITED *OR* VISITED NOW
###############
# binary indicating prior visit in original data
TNtests = np.sum(lgdict['N'], axis=1)
c_visited = np.array([1 if TNtests[x] > 0 else 0 for x in range(numTN)])

def GetConstraints_new(optparamdict, juncvec, seqcostlist, bindistaccessvectors, omegamat):
    numTN, B, ctest = len(optparamdict['deptnames']), optparamdict['budget'], optparamdict['pertestcost']
    f_dept, bigM = optparamdict['deptfixedcostvec'], optparamdict['Mconstant']
    # Build lower and upper inequality values
    optconstrlower = np.concatenate((np.ones(numTN*4+1) * -np.inf, np.array([1]),
                                     np.ones(numTN*3)*-np.inf,
                                    np.ones(numTN)*-np.inf))
    optconstrupper = np.concatenate((np.array([B]), np.zeros(numTN*2), np.array(juncvec), np.zeros(numTN),
                                     np.array([1]),
                                     np.zeros(numTN), np.array(juncvec), np.zeros(numTN),
                                     bigM*c_visited))
    # Build A matrix, from left to right
    # Build z district binaries first
    optconstraintmat1 = np.vstack((f_dept, -bigM * np.identity(numTN), np.identity(numTN), 0 * np.identity(numTN),
                                   np.identity(numTN), np.zeros(numTN),
                                   0 * np.identity(numTN), 0 * np.identity(numTN), 0 * np.identity(numTN),
                                   0 * np.identity(numTN)))
    # n^' matrices
    optconstraintmat2 = np.vstack((ctest * np.ones(numTN), np.identity(numTN), -np.identity(numTN),
                                   np.identity(numTN), 0 * np.identity(numTN), np.zeros(numTN),
                                   -omegamat, 0 * np.identity(numTN), 0 * np.identity(numTN),
                                   -bigM*np.identity(numTN)))
    # n^'' matrices
    optconstraintmat3 = np.vstack((ctest * np.ones(numTN), np.identity(numTN), -np.identity(numTN),
                                   0 * np.identity(numTN), 0 * np.identity(numTN), np.zeros(numTN),
                                   -omegamat, 0 * np.identity(numTN), 0 * np.identity(numTN),
                                   -bigM*np.identity(numTN)))
    # path matrices
    optconstraintmat4 = np.vstack((np.array(seqcostlist).T, np.zeros((numTN * 3, numPath)),
                                   (-bindistaccessvectors).T, np.ones(numPath),
                                    np.zeros((numTN * 4, numPath))))
    # z_hat matrices
    optconstraintmat5 = np.vstack((np.zeros(((numTN * 6) + 2, numTN)), np.identity(numTN),
                                   0 * np.identity(numTN)))
    # n_hat^' matrices
    optconstraintmat6 = np.vstack((np.zeros(((numTN*4)+2, numTN)),
                                   np.identity(numTN), np.identity(numTN), -np.identity(numTN),
                                   np.identity(numTN)))
    # n_hat^'' matrices
    optconstraintmat7 = np.vstack((np.zeros(((numTN*4)+2, numTN)),
                                   np.identity(numTN), 0 * np.identity(numTN), -np.identity(numTN),
                                   np.identity(numTN)))

    optconstraintmat = np.hstack((optconstraintmat1, optconstraintmat2, optconstraintmat3, optconstraintmat4,
                                  optconstraintmat5, optconstraintmat6, optconstraintmat7))
    return spo.LinearConstraint(optconstraintmat, optconstrlower, optconstrupper)

optobjvec = GetObjective_nozhat(lvec, m1vec, m2vec, numPath)
optconstraints = GetConstraints_new(optparamdict, juncvec, seqcostlist_trim, bindistaccessvectors_trim, omegaMat)
spoOutput = milp(c=optobjvec, constraints=optconstraints, integrality=optintegrality, bounds=optbounds)
initsoln, initsoln_obj = spoOutput.x, spoOutput.fun*-1
scipytoallocation(initsoln, deptNames, regNames, seqlist_trim, eliminateZeros=True)
scipytoallocation(initsoln, deptNames, regNames, seqlist_trim, eliminateZeros=False)
''' SOLUTION WITH NEW INDICATOR VAR LOGIC;
*MUCH* DIFFERENT ALLOCATION; NO PRIOR TESTED TNS WERE VISITED (EVEN THE CAPITAL DAKAR);
MODERATELY EXPLORATIVE PATH CHOSEN, WITH 4 PROVINCES
Bambey: 1 0 1
Fatick: 1 13 67
Foundiougne: 1 7 71
Gossas: 1 0 1
Guinguineo: 1 0 6
Keur Massar: 1 0 2
Mbacke: 1 8 73
Nioro du Rip: 1 14 66
Pikine: 1 6 72
Path: Dakar Diourbel Kaolack Fatick 
util: 1.6785062463012945
util_CI: (1.6690669055971963, 1.6879455870053928)
'''
n = scipysoltoallocvec(initsoln, numTN)
util, util_CI = sampf.getImportanceUtilityEstimate(n, lgdict, paramdict, numimportdraws=20000,
                                                   extremadelta=-1, preservevar=False)
print(util)
print(util_CI)

#0 Bakel: 0 0 0
#0 Bambey: 1 0 1
#0 Bignona: 0 0 0
#0 Birkilane: 0 0 0
#0 Bounkiling: 0 0 0

#1 Dagana: 0 0 0
#1 Dakar: 0 0 0
#1 Diourbel: 0 0 0
#0 Fatick: 1 13 67
#0 Foundiougne: 1 7 71

#0 Gossas: 1 0 1
#0 Goudiry: 0 0 0
#0 Goudoump: 0 0 0
#1 Guediawaye: 0 0 0
#0 Guinguineo: 1 0 6

#1 Kaffrine: 0 0 0
#1 Kanel: 0 0 0
#1 Kaolack: 0 0 0
#1 Kebemer: 0 0 0
#1 Kedougou: 0 0 0

#0 Keur Massar: 1 0 2
#1 Kolda: 0 0 0
#1 Koumpentoum: 0 0 0
#0 Koungheul: 0 0 0
#0 Linguere: 0 0 0

#0 Louga: 0 0 0
#0 Malem Hoddar: 0 0 0
#1 Matam: 0 0 0
#0 Mbacke: 1 8 73
#1 Mbour: 0 0 0

#0 Medina Yoro Foulah: 0 0 0
#0 Nioro du Rip: 1 14 66
#0 Oussouye: 0 0 0
#0 Pikine: 1 6 72
#1 Podor: 0 0 0

#0 Ranerou Ferlo: 0 0 0
#1 Rufisque: 0 0 0
#1 Saint-Louis: 0 0 0
#0 Salemata: 0 0 0
#0 Saraya: 0 0 0

#0 Sedhiou: 0 0 0
#1 Tambacounda: 0 0 0
#1 Thies: 0 0 0
#1 Tivaoune: 0 0 0
#1 Velingara: 0 0 0
#0 Ziguinchor: 0 0 0

###############
### USE NEW J() MEASURE, Z STRUCTURE, AND INDICATOR VAR LOGIC;
# ADD: BOUND ON n + n_hat
# ADD: MINIMUM on n
###############
maxratio = 1/4
J = np.zeros((numTN, numTN))
for i in range(numTN):
    for j in range(numTN):
        J[i, j] = np.sum(np.min(np.vstack((Q[i], Q[j])), axis=0))

Jadj = J * maxratio  # adjust for maximum learning importance
Jadj = Jadj + np.identity(numTN)*(1-maxratio)  # adjust for diagonals to be 1
omegaMat = Jadj
bdsarr = np.array(bds)
c_testmin = 5
def GetConstraints_new_two(optparamdict, juncvec, seqcostlist, bindistaccessvectors, omegamat):
    numTN, B, ctest = len(optparamdict['deptnames']), optparamdict['budget'], optparamdict['pertestcost']
    f_dept, bigM = optparamdict['deptfixedcostvec'], optparamdict['Mconstant']
    # Build lower and upper inequality values
    optconstrlower = np.concatenate((np.ones(numTN*4+1) * -np.inf, np.array([1]),
                                     np.ones(numTN*3)*-np.inf,
                                    np.ones(numTN*2)*-np.inf))
    optconstrupper = np.concatenate((np.array([B]), np.zeros(numTN*2), np.array(juncvec), np.zeros(numTN),
                                     np.array([1]),
                                     np.zeros(numTN), np.array(juncvec), np.zeros(numTN),
                                     bigM*c_visited, bdsarr))
    # Build A matrix, from left to right
    # Build z district binaries first
    optconstraintmat1 = np.vstack((f_dept, -bigM * np.identity(numTN), c_testmin*np.identity(numTN), 0 * np.identity(numTN),
                                   np.identity(numTN), np.zeros(numTN),
                                   0 * np.identity(numTN), 0 * np.identity(numTN), 0 * np.identity(numTN),
                                   0 * np.identity(numTN), 0 * np.identity(numTN)))
    # n^' matrices
    optconstraintmat2 = np.vstack((ctest * np.ones(numTN), np.identity(numTN), -np.identity(numTN),
                                   np.identity(numTN), 0 * np.identity(numTN), np.zeros(numTN),
                                   -omegamat, 0 * np.identity(numTN), 0 * np.identity(numTN),
                                   -bigM*np.identity(numTN), np.identity(numTN)))
    # n^'' matrices
    optconstraintmat3 = np.vstack((ctest * np.ones(numTN), np.identity(numTN), -np.identity(numTN),
                                   0 * np.identity(numTN), 0 * np.identity(numTN), np.zeros(numTN),
                                   -omegamat, 0 * np.identity(numTN), 0 * np.identity(numTN),
                                   -bigM*np.identity(numTN), np.identity(numTN)))
    # path matrices
    optconstraintmat4 = np.vstack((np.array(seqcostlist).T, np.zeros((numTN * 3, numPath)),
                                   (-bindistaccessvectors).T, np.ones(numPath),
                                    np.zeros((numTN * 5, numPath))))
    # z_hat matrices
    optconstraintmat5 = np.vstack((np.zeros(((numTN * 6) + 2, numTN)), np.identity(numTN),
                                   0 * np.identity(numTN), 0 * np.identity(numTN)))
    # n_hat^' matrices
    optconstraintmat6 = np.vstack((np.zeros(((numTN*4)+2, numTN)),
                                   np.identity(numTN), np.identity(numTN), -np.identity(numTN),
                                   np.identity(numTN), np.identity(numTN)))
    # n_hat^'' matrices
    optconstraintmat7 = np.vstack((np.zeros(((numTN*4)+2, numTN)),
                                   np.identity(numTN), 0 * np.identity(numTN), -np.identity(numTN),
                                   np.identity(numTN), np.identity(numTN)))

    optconstraintmat = np.hstack((optconstraintmat1, optconstraintmat2, optconstraintmat3, optconstraintmat4,
                                  optconstraintmat5, optconstraintmat6, optconstraintmat7))
    return spo.LinearConstraint(optconstraintmat, optconstrlower, optconstrupper)

optconstraints = GetConstraints_new_two(optparamdict, juncvec, seqcostlist_trim,
                                        bindistaccessvectors_trim, omegaMat)
spoOutput = milp(c=optobjvec, constraints=optconstraints, integrality=optintegrality, bounds=optbounds)
initsoln, initsoln_obj = spoOutput.x, spoOutput.fun*-1
scipytoallocation(initsoln, deptNames, regNames, seqlist_trim, eliminateZeros=True)
# for i in range(numTN):
#     print(str(initsoln[i]) + ' ' + str(c_visited[i]))
'''SOLUTION WITH INDICATOR VAR BOUND; WITH maxratio=1/2
Bambey: 1 0 5
Birkilane: 1 0 16
Dakar: 1 0 10
Fatick: 1 0 5
Foundiougne: 1 0 39
Gossas: 1 0 5
Guinguineo: 1 0 70
Kaffrine: 1 0 7
Kaolack: 1 0 26
Keur Massar: 1 0 5
Koungheul: 1 0 5
Malem Hoddar: 1 0 14
Mbacke: 1 0 59
Nioro du Rip: 1 0 5
Pikine: 1 0 5
Path: Dakar Fatick Kaolack Kaffrine Diourbel
util: 1.6216637459049839 [pre-min]
2.4380140438457936
(2.4285033135552805, 2.4475247741363066)
'''

n = scipysoltoallocvec(initsoln, numTN)
MakeAllocationHeatMap(n, optparamdict, plotTitle='Coverage solution', cmapstr='Blues')

util, util_CI = sampf.getImportanceUtilityEstimate(n, lgdict, paramdict, numimportdraws=20000,
                                                   extremadelta=-1, preservevar=False)
print(util)
print(util_CI)

''' WITH maxratio=1/4
Bambey: 1 0 23
Birkilane: 1 0 21
Dakar: 1 0 22
Diourbel: 1 16 4
Fatick: 1 0 20
Foundiougne: 1 0 20
Gossas: 1 0 22
Guinguineo: 1 0 21
Keur Massar: 1 0 23
Koungheul: 1 0 24
Malem Hoddar: 1 0 5
Mbacke: 1 0 19
Nioro du Rip: 1 0 21
Pikine: 1 0 20
Path: Dakar Fatick Kaolack Kaffrine Diourbel
util: XX
'''

n = scipysoltoallocvec(initsoln, numTN)
MakeAllocationHeatMap(n, optparamdict, plotTitle='Coverage solution', cmapstr='Blues')

util, util_CI = sampf.getImportanceUtilityEstimate(n, lgdict, paramdict, numimportdraws=20000,
                                                   extremadelta=-1, preservevar=False)
print(util)
print(util_CI)