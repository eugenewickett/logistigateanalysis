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

'''
def findingAnExample():
    dataDict = util.generateRandDataDict(numImp=2, numOut=3, diagSens=0.90,
                                    diagSpec=0.99, numSamples=90,
                                    dataType='Tracked', transMatLambda=1.1,
                                    randSeed=-1,
                                    trueRates=[0.1,0.3,0.3,0.2,0.1])
    MCMCdict_NUTS = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    dataDict.update({'numPostSamples': 500,
                            'prior': methods.prior_normal(),
                            'MCMCdict': MCMCdict_NUTS})

    lgDict = lg.runlogistigate(dataDict)
    util.plotPostSamples(lgDict)
    util.printEstimates(lgDict)
    print(lgDict['transMat'])

    return
'''


def decision1ModelSimulation(n=100, n1=50, t=0.20, delta=0.1, eps1=0.1, eps2=0.1, blameOrder=['Out1', 'Imp1', 'Out2'],
                             confInt=0.95, reps=1000):
    '''
    Function for simulating different parameters in a decision model regarding assigning blame in a 1-importer, 2-outlet
    system
    '''
    import numpy as np
    import scipy.stats as sps
    # Use blameOrder list to define the underlying SFP rates; 1st entry has SFP rate of t+delta, 2nd has t-eps1,
    #   3rd has t-eps2
    SFPrates = [t + delta, t - eps1, t - eps2]
    # Assign SFP rates for importer and outlets
    imp1Rate = SFPrates[blameOrder.index('Imp1')]
    out1Rate = SFPrates[blameOrder.index('Out1')]
    out2Rate = SFPrates[blameOrder.index('Out2')]
    # Generate data using n, n1, and assuming perfect diagnostic accuracy
    n2 = n - n1
    # Run for number of replications
    repsList = []
    for r in range(reps):
        n1pos = np.random.binomial(n1, p=out1Rate + (1 - out1Rate) * imp1Rate)
        n2pos = np.random.binomial(n2, p=out2Rate + (1 - out2Rate) * imp1Rate)
        # Form confidence intervals
        n1sampMean = n1pos / n1
        n2sampMean = n2pos / n2
        zscore = sps.norm.ppf(confInt + (1 - confInt) / 2)
        n1radius = zscore * np.sqrt(n1sampMean * (1 - n1sampMean) / n1)
        n2radius = zscore * np.sqrt(n2sampMean * (1 - n2sampMean) / n2)
        n1interval = [max(0, n1sampMean - n1radius), min(1, n1sampMean + n1radius)]
        n2interval = [max(0, n2sampMean - n2radius), min(1, n2sampMean + n2radius)]
        # Make a decision
        if n2interval[0] > n1interval[1]:
            repsList.append('Out2')
        elif n2interval[1] < n1interval[0]:
            repsList.append('Out1')
        else:
            repsList.append('Imp1')

    return repsList


def runDecisionSimsScratch():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    import winsound
    duration = 1000  # milliseconds
    freq = 440  # Hz

    numReps = 1000
    numSamps = 200
    currResultsList = decision1ModelSimulation(n=numSamps, n1=numSamps / 2, t=0.20, delta=0.1, eps1=0.1, eps2=0.1,
                                               blameOrder=['Out1', 'Imp1', 'Out2'], confInt=0.95, reps=numReps)
    percCorrect = currResultsList.count('Out1') / numReps

    numReps = 1000
    # Look at number of samples vs the threshold, importer 1 as cuplrit
    curr_blameOrder = ['Imp1', 'Out1', 'Out2']
    nVec = np.arange(50, 1050, 50)
    tVec = np.arange(0.15, 0.75, 0.05)
    nVSt_mat = np.zeros(shape=[len(nVec), len(tVec)])
    for nInd, curr_n in enumerate(nVec):
        for tInd, curr_t in enumerate(tVec):
            currResultsList = decision1ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=0.1, eps1=0.1, eps2=0.1,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            nVSt_mat[nInd, tInd] = currResultsList.count('Imp1') / numReps

    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(tVec, nVec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, nVSt_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nTotal sample size n, Threshold t\nUnder Importer 1 as culprit')
    plt.xlabel('t', size=16)
    plt.ylabel('n', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at number of samples vs the threshold, outlet 1 as cuplrit
    curr_blameOrder = ['Out1', 'Imp1', 'Out2']
    nVec = np.arange(50, 1050, 50)
    tVec = np.arange(0.15, 0.75, 0.05)
    nVSt_mat = np.zeros(shape=[len(nVec), len(tVec)])
    for nInd, curr_n in enumerate(nVec):
        for tInd, curr_t in enumerate(tVec):
            currResultsList = decision1ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=0.1, eps1=0.1, eps2=0.1,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            nVSt_mat[nInd, tInd] = currResultsList.count('Out1') / numReps

    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(tVec, nVec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, nVSt_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nTotal sample size n, Threshold t\nUnder Outlet 1 as culprit')
    plt.xlabel('t', size=16)
    plt.ylabel('n', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at delta vs. epsilon; set t=0.3, n=300; importer 1 is culprit
    curr_blameOrder = ['Imp1', 'Out1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    deltaVec = np.arange(0.01, 0.21, 0.01)
    epsVec = np.arange(0.01, 0.21, 0.01)
    deltaVSeps_mat = np.zeros(shape=[len(deltaVec), len(epsVec)])
    for dInd, curr_d in enumerate(deltaVec):
        for eInd, curr_e in enumerate(epsVec):
            currResultsList = decision1ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=curr_d,
                                                       eps1=curr_e, eps2=curr_e,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            deltaVSeps_mat[dInd, eInd] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(epsVec, deltaVec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, deltaVSeps_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nDistance delta, Distance eps\nUnder Importer 1 as culprit,t=0.3,n=300')
    plt.xlabel('eps', size=16)
    plt.ylabel('delta', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at delta vs. epsilon; set t=0.3, n=300; outlet 1 is culprit
    curr_blameOrder = ['Out1', 'Imp1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    deltaVec = np.arange(0.01, 0.21, 0.01)
    epsVec = np.arange(0.01, 0.21, 0.01)
    deltaVSeps_mat = np.zeros(shape=[len(deltaVec), len(epsVec)])
    for dInd, curr_d in enumerate(deltaVec):
        for eInd, curr_e in enumerate(epsVec):
            currResultsList = decision1ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=curr_d,
                                                       eps1=curr_e, eps2=curr_e,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            deltaVSeps_mat[dInd, eInd] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(epsVec, deltaVec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, deltaVSeps_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nDistance delta, Distance eps\nUnder Outlet 1 as culprit,t=0.3,n=300')
    plt.xlabel('eps', size=16)
    plt.ylabel('delta', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at confidence interval vs. ratio of samples from Outlet 1; set t=0.3, n=300; importer 1 is culprit
    curr_blameOrder = ['Imp1', 'Out1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    curr_d = 0.1
    curr_e = 0.1
    n1ratios = np.arange(0.1, 1.0, 0.1)
    confInts = np.arange(0.3, 1.0, 0.05)
    n1ratsVSconfs_mat = np.zeros(shape=[len(n1ratios), len(confInts)])
    for n1Ind, curr_n1 in enumerate(n1ratios):
        for cInd, curr_c in enumerate(confInts):
            currResultsList = decision1ModelSimulation(n=curr_n, n1=int(curr_n * curr_n1), t=curr_t, delta=curr_d,
                                                       eps1=curr_e, eps2=curr_e,
                                                       blameOrder=curr_blameOrder, confInt=curr_c, reps=numReps)
            n1ratsVSconfs_mat[n1Ind, cInd] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(confInts, n1ratios)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, n1ratsVSconfs_mat * 100, cmap=cm.coolwarm)
    plt.suptitle(
        'Classification accuracy vs.\nn1 ratio of n, CI level\nUnder Importer 1 as culprit,t=0.3,n=300,delta=eps=0.1')
    plt.xlabel('CI level', size=16)
    plt.ylabel('n1 ratio', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at confidence interval vs. ratio of samples from Outlet 1; set t=0.3, n=300; outlet 1 is culprit
    curr_blameOrder = ['Out1', 'Imp1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    curr_d = 0.1
    curr_e = 0.1
    n1ratios = np.arange(0.1, 1.0, 0.1)
    confInts = np.arange(0.3, 1.0, 0.05)
    n1ratsVSconfs_mat = np.zeros(shape=[len(n1ratios), len(confInts)])
    for n1Ind, curr_n1 in enumerate(n1ratios):
        for cInd, curr_c in enumerate(confInts):
            currResultsList = decision1ModelSimulation(n=curr_n, n1=int(curr_n * curr_n1), t=curr_t, delta=curr_d,
                                                       eps1=curr_e, eps2=curr_e,
                                                       blameOrder=curr_blameOrder, confInt=curr_c, reps=numReps)
            n1ratsVSconfs_mat[n1Ind, cInd] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(confInts, n1ratios)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, n1ratsVSconfs_mat * 100, cmap=cm.coolwarm)
    plt.suptitle(
        'Classification accuracy vs.\nn1 ratio of n, CI level\nUnder Outlet 1 as culprit,t=0.3,n=300,delta=eps=0.1')
    plt.xlabel('CI level', size=16)
    plt.ylabel('n1 ratio', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    return


def decision2ModelSimulation(n=100, n1=50, t=0.20, delta=0.1, eps1=0.1, eps2=0.1, blameOrder=['Out1', 'Imp1', 'Out2'],
                             confInt=0.95, reps=1000):
    '''
    Function for simulating different parameters in a decision model regarding assigning blame in a 1-importer, 2-outlet
    system; for d2, the outlet is blamed if the confidence interval is completely above the threshold t; otherwise,
    the importer is blamed
    '''
    import numpy as np
    import scipy.stats as sps
    # Use blameOrder list to define the underlying SFP rates; 1st entry has SFP rate of t+delta, 2nd has t-eps1,
    #   3rd has t-eps2
    SFPrates = [t + delta, t - eps1, t - eps2]
    # Assign SFP rates for importer and outlets
    imp1Rate = SFPrates[blameOrder.index('Imp1')]
    out1Rate = SFPrates[blameOrder.index('Out1')]
    out2Rate = SFPrates[blameOrder.index('Out2')]
    # Generate data using n, n1, and assuming perfect diagnostic accuracy
    n2 = n - n1
    # Run for number of replications
    repsList = []
    for r in range(reps):
        n1pos = np.random.binomial(n1, p=out1Rate + (1 - out1Rate) * imp1Rate)
        n2pos = np.random.binomial(n2, p=out2Rate + (1 - out2Rate) * imp1Rate)
        # Form confidence intervals
        n1sampMean = n1pos / n1
        n2sampMean = n2pos / n2
        zscore = sps.norm.ppf(confInt + (1 - confInt) / 2)
        n1radius = zscore * np.sqrt(n1sampMean * (1 - n1sampMean) / n1)
        n2radius = zscore * np.sqrt(n2sampMean * (1 - n2sampMean) / n2)
        n1interval = [max(0, n1sampMean - n1radius), min(1, n1sampMean + n1radius)]
        n2interval = [max(0, n2sampMean - n2radius), min(1, n2sampMean + n2radius)]
        # Make a decision, d2
        if n1interval[0] > t:
            if n1interval[0] >= n2interval[0]:
                repsList.append('Out1')
            else:  # Outlet 2 interval lower bound is above the lower bound for the interval for Outlet 1
                repsList.append('Out2')
        elif n1interval[0] > t:
            repsList.append('Out2')
        else:
            repsList.append('Imp1')

    return repsList


def runDecision2SimsScratch():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    import winsound
    duration = 1000  # milliseconds
    freq = 440  # Hz

    numReps = 1000
    numSamps = 200
    currResultsList = decision2ModelSimulation(n=numSamps, n1=numSamps / 2, t=0.20, delta=0.1, eps1=0.1, eps2=0.1,
                                               blameOrder=['Out1', 'Imp1', 'Out2'], confInt=0.95, reps=numReps)
    percCorrect = currResultsList.count('Out1') / numReps

    numReps = 1000
    # Look at number of samples vs the threshold, importer 1 as cuplrit
    curr_blameOrder = ['Imp1', 'Out1', 'Out2']
    nVec = np.arange(50, 1050, 50)
    tVec = np.arange(0.15, 0.75, 0.05)
    nVSt_mat = np.zeros(shape=[len(nVec), len(tVec)])
    for nInd, curr_n in enumerate(nVec):
        for tInd, curr_t in enumerate(tVec):
            currResultsList = decision2ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=0.1, eps1=0.1, eps2=0.1,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            nVSt_mat[nInd, tInd] = currResultsList.count('Imp1') / numReps

    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(tVec, nVec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, nVSt_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nTotal sample size n, Threshold t\nUnder Importer 1 as culprit')
    plt.xlabel('t', size=16)
    plt.ylabel('n', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at number of samples vs the threshold, outlet 1 as cuplrit
    curr_blameOrder = ['Out1', 'Imp1', 'Out2']
    nVec = np.arange(50, 1050, 50)
    tVec = np.arange(0.15, 0.75, 0.05)
    nVSt_mat = np.zeros(shape=[len(nVec), len(tVec)])
    for nInd, curr_n in enumerate(nVec):
        for tInd, curr_t in enumerate(tVec):
            currResultsList = decision2ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=0.1, eps1=0.1, eps2=0.1,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            nVSt_mat[nInd, tInd] = currResultsList.count('Out1') / numReps

    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(tVec, nVec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, nVSt_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nTotal sample size n, Threshold t\nUnder Outlet 1 as culprit')
    plt.xlabel('t', size=16)
    plt.ylabel('n', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at delta vs. epsilon; set t=0.3, n=300; importer 1 is culprit
    curr_blameOrder = ['Imp1', 'Out1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    deltaVec = np.arange(0.01, 0.21, 0.01)
    epsVec = np.arange(0.01, 0.21, 0.01)
    deltaVSeps_mat = np.zeros(shape=[len(deltaVec), len(epsVec)])
    for dInd, curr_d in enumerate(deltaVec):
        for eInd, curr_e in enumerate(epsVec):
            currResultsList = decision2ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=curr_d,
                                                       eps1=curr_e, eps2=curr_e,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            deltaVSeps_mat[dInd, eInd] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(epsVec, deltaVec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, deltaVSeps_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nDistance delta, Distance eps\nUnder Importer 1 as culprit,t=0.3,n=300')
    plt.xlabel('eps', size=16)
    plt.ylabel('delta', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at delta vs. epsilon; set t=0.3, n=300; outlet 1 is culprit
    curr_blameOrder = ['Out1', 'Imp1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    deltaVec = np.arange(0.01, 0.21, 0.01)
    epsVec = np.arange(0.01, 0.21, 0.01)
    deltaVSeps_mat = np.zeros(shape=[len(deltaVec), len(epsVec)])
    for dInd, curr_d in enumerate(deltaVec):
        for eInd, curr_e in enumerate(epsVec):
            currResultsList = decision2ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=curr_d,
                                                       eps1=curr_e, eps2=curr_e,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            deltaVSeps_mat[dInd, eInd] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(epsVec, deltaVec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, deltaVSeps_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nDistance delta, Distance eps\nUnder Outlet 1 as culprit,t=0.3,n=300')
    plt.xlabel('eps', size=16)
    plt.ylabel('delta', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at confidence interval vs. ratio of samples from Outlet 1; set t=0.3, n=300; importer 1 is culprit
    curr_blameOrder = ['Imp1', 'Out1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    curr_d = 0.1
    curr_e = 0.1
    n1ratios = np.arange(0.1, 1.0, 0.1)
    confInts = np.arange(0.3, 1.0, 0.05)
    n1ratsVSconfs_mat = np.zeros(shape=[len(n1ratios), len(confInts)])
    for n1Ind, curr_n1 in enumerate(n1ratios):
        for cInd, curr_c in enumerate(confInts):
            currResultsList = decision2ModelSimulation(n=curr_n, n1=int(curr_n * curr_n1), t=curr_t, delta=curr_d,
                                                       eps1=curr_e, eps2=curr_e,
                                                       blameOrder=curr_blameOrder, confInt=curr_c, reps=numReps)
            n1ratsVSconfs_mat[n1Ind, cInd] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(confInts, n1ratios)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, n1ratsVSconfs_mat * 100, cmap=cm.coolwarm)
    plt.suptitle(
        'Classification accuracy vs.\nn1 ratio of n, CI level\nUnder Importer 1 as culprit,t=0.3,n=300,delta=eps=0.1')
    plt.xlabel('CI level', size=16)
    plt.ylabel('n1 ratio', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at confidence interval vs. ratio of samples from Outlet 1; set t=0.3, n=300; outlet 1 is culprit
    curr_blameOrder = ['Out1', 'Imp1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    curr_d = 0.1
    curr_e = 0.1
    n1ratios = np.arange(0.1, 1.0, 0.1)
    confInts = np.arange(0.3, 1.0, 0.05)
    n1ratsVSconfs_mat = np.zeros(shape=[len(n1ratios), len(confInts)])
    for n1Ind, curr_n1 in enumerate(n1ratios):
        for cInd, curr_c in enumerate(confInts):
            currResultsList = decision2ModelSimulation(n=curr_n, n1=int(curr_n * curr_n1), t=curr_t, delta=curr_d,
                                                       eps1=curr_e, eps2=curr_e,
                                                       blameOrder=curr_blameOrder, confInt=curr_c, reps=numReps)
            n1ratsVSconfs_mat[n1Ind, cInd] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(confInts, n1ratios)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, n1ratsVSconfs_mat * 100, cmap=cm.coolwarm)
    plt.suptitle(
        'Classification accuracy vs.\nn1 ratio of n, CI level\nUnder Outlet 1 as culprit,t=0.3,n=300,delta=eps=0.1')
    plt.xlabel('CI level', size=16)
    plt.ylabel('n1 ratio', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at epsilon1 vs. epsilon2; set t=0.3, n=300; importer 1 is culprit
    curr_blameOrder = ['Imp1', 'Out1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    curr_d = 0.1
    eps1Vec = np.arange(0.01, 0.21, 0.01)
    eps2Vec = np.arange(0.01, 0.21, 0.01)
    eps1VSeps2_mat = np.zeros(shape=[len(eps1Vec), len(eps2Vec)])
    for e1Ind, curr_e1 in enumerate(eps1Vec):
        for e2Ind, curr_e2 in enumerate(eps2Vec):
            currResultsList = decision2ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=curr_d,
                                                       eps1=curr_e1, eps2=curr_e2,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            eps1VSeps2_mat[e1Ind, e2Ind] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(eps2Vec, eps1Vec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, eps1VSeps2_mat * 100, cmap=cm.coolwarm)
    plt.suptitle(
        'Classification accuracy vs.\nDistance eps1, Distance eps2\nUnder Importer 1 as culprit, Outlet1 as eps1,t=0.3,n=300,delta=0.1')
    plt.xlabel('eps2', size=16)
    plt.ylabel('eps1', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at epsilon1 vs. epsilon2; set t=0.3, n=300; outlet 1 is culprit
    curr_blameOrder = ['Out1', 'Imp1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    curr_d = 0.1
    eps1Vec = np.arange(0.01, 0.21, 0.01)
    eps2Vec = np.arange(0.01, 0.21, 0.01)
    eps1VSeps2_mat = np.zeros(shape=[len(eps1Vec), len(eps2Vec)])
    for e1Ind, curr_e1 in enumerate(eps1Vec):
        for e2Ind, curr_e2 in enumerate(eps2Vec):
            currResultsList = decision2ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=curr_d,
                                                       eps1=curr_e1, eps2=curr_e2,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            eps1VSeps2_mat[e1Ind, e2Ind] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(eps2Vec, eps1Vec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, eps1VSeps2_mat * 100, cmap=cm.coolwarm)
    plt.suptitle(
        'Classification accuracy vs.\nDistance eps1, Distance eps2\nUnder Outlet 1 as culprit, Importer 1 as eps1, t=0.3,n=300,delta=0.1')
    plt.xlabel('eps2', size=16)
    plt.ylabel('eps1', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at epsilon1 vs. epsilon2; set t=0.3, n=300; outlet 1 is culprit, outlet 2 is eps1
    curr_blameOrder = ['Out1', 'Out2', 'Imp1']
    curr_t = 0.3
    curr_n = 300
    curr_d = 0.1
    eps1Vec = np.arange(0.01, 0.21, 0.01)
    eps2Vec = np.arange(0.01, 0.21, 0.01)
    eps1VSeps2_mat = np.zeros(shape=[len(eps1Vec), len(eps2Vec)])
    for e1Ind, curr_e1 in enumerate(eps1Vec):
        for e2Ind, curr_e2 in enumerate(eps2Vec):
            currResultsList = decision2ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=curr_d,
                                                       eps1=curr_e1, eps2=curr_e2,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            eps1VSeps2_mat[e1Ind, e2Ind] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(eps2Vec, eps1Vec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, eps1VSeps2_mat * 100, cmap=cm.coolwarm)
    plt.suptitle(
        'Classification accuracy vs.\nDistance eps1, Distance eps2\nUnder Outlet 1 as culprit, Outlet 2 as eps1, t=0.3,n=300,delta=0.1')
    plt.xlabel('eps2', size=16)
    plt.ylabel('eps1', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    return


def decision3ModelSimulation(n=100, n1=50, t=0.20, delta=0.1, eps1=0.1, eps2=0.1, blameOrder=['Out1', 'Imp1', 'Out2'],
                             confInt=0.95, reps=1000):
    '''
    Function for simulating different parameters in a decision model regarding assigning blame in a 1-importer, 2-outlet
    system; for d3, the outlet is blamed if the confidence interval is completely above the threshold t; otherwise,
    the importer is blamed
    '''
    import numpy as np
    import scipy.stats as sps
    # Use blameOrder list to define the underlying SFP rates; 1st entry has SFP rate of t+delta, 2nd has t-eps1,
    #   3rd has t-eps2
    SFPrates = [t + delta, t - eps1, t - eps2]
    # Assign SFP rates for importer and outlets
    imp1Rate = SFPrates[blameOrder.index('Imp1')]
    out1Rate = SFPrates[blameOrder.index('Out1')]
    out2Rate = SFPrates[blameOrder.index('Out2')]
    # Generate data using n, n1, and assuming perfect diagnostic accuracy
    n2 = n - n1
    # Run for number of replications
    repsList = []
    for r in range(reps):
        n1pos = np.random.binomial(n1, p=out1Rate + (1 - out1Rate) * imp1Rate)
        n2pos = np.random.binomial(n2, p=out2Rate + (1 - out2Rate) * imp1Rate)
        # Form confidence intervals
        n1sampMean = n1pos / n1
        n2sampMean = n2pos / n2
        zscore = sps.norm.ppf(confInt + (1 - confInt) / 2)
        n1radius = zscore * np.sqrt(n1sampMean * (1 - n1sampMean) / n1)
        n2radius = zscore * np.sqrt(n2sampMean * (1 - n2sampMean) / n2)
        n1interval = [max(0, n1sampMean - n1radius), min(1, n1sampMean + n1radius)]
        n2interval = [max(0, n2sampMean - n2radius), min(1, n2sampMean + n2radius)]
        # Make a decision, d3
        if n1sampMean - n2interval[0] > t and n2sampMean - n1interval[0] < t:
            repsList.append('Out1')
        elif n2sampMean - n1interval[0] > t and n1sampMean - n2interval[0] < t:
            repsList.append('Out2')
        else:
            repsList.append('Imp1')

    return repsList


def runDecision3SimsScratch():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    import winsound
    duration = 1000  # milliseconds
    freq = 440  # Hz

    numReps = 1000
    numSamps = 200
    currResultsList = decision3ModelSimulation(n=numSamps, n1=numSamps / 2, t=0.20, delta=0.1, eps1=0.1, eps2=0.1,
                                               blameOrder=['Out1', 'Imp1', 'Out2'], confInt=0.95, reps=numReps)
    percCorrect = currResultsList.count('Out1') / numReps

    numReps = 1000
    # Look at number of samples vs the threshold, importer 1 as culprit
    curr_blameOrder = ['Imp1', 'Out1', 'Out2']
    nVec = np.arange(50, 1050, 50)
    tVec = np.arange(0.15, 0.75, 0.05)
    nVSt_mat = np.zeros(shape=[len(nVec), len(tVec)])
    for nInd, curr_n in enumerate(nVec):
        for tInd, curr_t in enumerate(tVec):
            currResultsList = decision3ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=0.1, eps1=0.1, eps2=0.1,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            nVSt_mat[nInd, tInd] = currResultsList.count('Imp1') / numReps

    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(tVec, nVec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, nVSt_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nTotal sample size n, Threshold t\nUnder Importer 1 as culprit')
    plt.xlabel('t', size=16)
    plt.ylabel('n', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at number of samples vs the threshold, outlet 1 as culprit
    curr_blameOrder = ['Out1', 'Imp1', 'Out2']
    nVec = np.arange(50, 1050, 50)
    tVec = np.arange(0.15, 0.75, 0.05)
    nVSt_mat = np.zeros(shape=[len(nVec), len(tVec)])
    for nInd, curr_n in enumerate(nVec):
        for tInd, curr_t in enumerate(tVec):
            currResultsList = decision3ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=0.1, eps1=0.1, eps2=0.1,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            nVSt_mat[nInd, tInd] = currResultsList.count('Out1') / numReps

    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(tVec, nVec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, nVSt_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nTotal sample size n, Threshold t\nUnder Outlet 1 as culprit')
    plt.xlabel('t', size=16)
    plt.ylabel('n', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at delta vs. epsilon; set t=0.3, n=300; importer 1 is culprit
    curr_blameOrder = ['Imp1', 'Out1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    deltaVec = np.arange(0.01, 0.21, 0.01)
    epsVec = np.arange(0.01, 0.21, 0.01)
    deltaVSeps_mat = np.zeros(shape=[len(deltaVec), len(epsVec)])
    for dInd, curr_d in enumerate(deltaVec):
        for eInd, curr_e in enumerate(epsVec):
            currResultsList = decision3ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=curr_d,
                                                       eps1=curr_e, eps2=curr_e,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            deltaVSeps_mat[dInd, eInd] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(epsVec, deltaVec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, deltaVSeps_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nDistance delta, Distance eps\nUnder Importer 1 as culprit,t=0.3,n=300')
    plt.xlabel('eps', size=16)
    plt.ylabel('delta', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at delta vs. epsilon; set t=0.3, n=300; outlet 1 is culprit
    curr_blameOrder = ['Out1', 'Imp1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    deltaVec = np.arange(0.01, 0.21, 0.01)
    epsVec = np.arange(0.01, 0.21, 0.01)
    deltaVSeps_mat = np.zeros(shape=[len(deltaVec), len(epsVec)])
    for dInd, curr_d in enumerate(deltaVec):
        for eInd, curr_e in enumerate(epsVec):
            currResultsList = decision3ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=curr_d,
                                                       eps1=curr_e, eps2=curr_e,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            deltaVSeps_mat[dInd, eInd] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(epsVec, deltaVec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, deltaVSeps_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nDistance delta, Distance eps\nUnder Outlet 1 as culprit,t=0.3,n=300')
    plt.xlabel('eps', size=16)
    plt.ylabel('delta', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at confidence interval vs. ratio of samples from Outlet 1; set t=0.3, n=300; importer 1 is culprit
    curr_blameOrder = ['Imp1', 'Out1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    curr_d = 0.1
    curr_e = 0.1
    n1ratios = np.arange(0.1, 1.0, 0.1)
    confInts = np.arange(0.3, 1.0, 0.05)
    n1ratsVSconfs_mat = np.zeros(shape=[len(n1ratios), len(confInts)])
    for n1Ind, curr_n1 in enumerate(n1ratios):
        for cInd, curr_c in enumerate(confInts):
            currResultsList = decision3ModelSimulation(n=curr_n, n1=int(curr_n * curr_n1), t=curr_t, delta=curr_d,
                                                       eps1=curr_e, eps2=curr_e,
                                                       blameOrder=curr_blameOrder, confInt=curr_c, reps=numReps)
            n1ratsVSconfs_mat[n1Ind, cInd] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(confInts, n1ratios)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, n1ratsVSconfs_mat * 100, cmap=cm.coolwarm)
    plt.suptitle(
        'Classification accuracy vs.\nn1 ratio of n, CI level\nUnder Importer 1 as culprit,t=0.3,n=300,delta=eps=0.1')
    plt.xlabel('CI level', size=16)
    plt.ylabel('n1 ratio', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at confidence interval vs. ratio of samples from Outlet 1; set t=0.3, n=300; outlet 1 is culprit
    curr_blameOrder = ['Out1', 'Imp1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    curr_d = 0.1
    curr_e = 0.1
    n1ratios = np.arange(0.1, 1.0, 0.1)
    confInts = np.arange(0.3, 1.0, 0.05)
    n1ratsVSconfs_mat = np.zeros(shape=[len(n1ratios), len(confInts)])
    for n1Ind, curr_n1 in enumerate(n1ratios):
        for cInd, curr_c in enumerate(confInts):
            currResultsList = decision3ModelSimulation(n=curr_n, n1=int(curr_n * curr_n1), t=curr_t, delta=curr_d,
                                                       eps1=curr_e, eps2=curr_e,
                                                       blameOrder=curr_blameOrder, confInt=curr_c, reps=numReps)
            n1ratsVSconfs_mat[n1Ind, cInd] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(confInts, n1ratios)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, n1ratsVSconfs_mat * 100, cmap=cm.coolwarm)
    plt.suptitle(
        'Classification accuracy vs.\nn1 ratio of n, CI level\nUnder Outlet 1 as culprit,t=0.3,n=300,delta=eps=0.1')
    plt.xlabel('CI level', size=16)
    plt.ylabel('n1 ratio', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at epsilon1 vs. epsilon2; set t=0.3, n=300; importer 1 is culprit
    curr_blameOrder = ['Imp1', 'Out1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    curr_d = 0.1
    eps1Vec = np.arange(0.01, 0.21, 0.01)
    eps2Vec = np.arange(0.01, 0.21, 0.01)
    eps1VSeps2_mat = np.zeros(shape=[len(eps1Vec), len(eps2Vec)])
    for e1Ind, curr_e1 in enumerate(eps1Vec):
        for e2Ind, curr_e2 in enumerate(eps2Vec):
            currResultsList = decision3ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=curr_d,
                                                       eps1=curr_e1, eps2=curr_e2,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            eps1VSeps2_mat[e1Ind, e2Ind] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(eps2Vec, eps1Vec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, eps1VSeps2_mat * 100, cmap=cm.coolwarm)
    plt.suptitle(
        'Classification accuracy vs.\nDistance eps1, Distance eps2\nUnder Importer 1 as culprit, Outlet1 as eps1,t=0.3,n=300,delta=0.1')
    plt.xlabel('eps2', size=16)
    plt.ylabel('eps1', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at epsilon1 vs. epsilon2; set t=0.3, n=300; outlet 1 is culprit
    curr_blameOrder = ['Out1', 'Imp1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    curr_d = 0.1
    eps1Vec = np.arange(0.01, 0.21, 0.01)
    eps2Vec = np.arange(0.01, 0.21, 0.01)
    eps1VSeps2_mat = np.zeros(shape=[len(eps1Vec), len(eps2Vec)])
    for e1Ind, curr_e1 in enumerate(eps1Vec):
        for e2Ind, curr_e2 in enumerate(eps2Vec):
            currResultsList = decision3ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=curr_d,
                                                       eps1=curr_e1, eps2=curr_e2,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            eps1VSeps2_mat[e1Ind, e2Ind] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(eps2Vec, eps1Vec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, eps1VSeps2_mat * 100, cmap=cm.coolwarm)
    plt.suptitle(
        'Classification accuracy vs.\nDistance eps1, Distance eps2\nUnder Outlet 1 as culprit, Importer 1 as eps1, t=0.3,n=300,delta=0.1')
    plt.xlabel('eps2', size=16)
    plt.ylabel('eps1', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at epsilon1 vs. epsilon2; set t=0.3, n=300; outlet 1 is culprit, outlet 2 is eps1
    curr_blameOrder = ['Out1', 'Out2', 'Imp1']
    curr_t = 0.3
    curr_n = 300
    curr_d = 0.1
    eps1Vec = np.arange(0.01, 0.21, 0.01)
    eps2Vec = np.arange(0.01, 0.21, 0.01)
    eps1VSeps2_mat = np.zeros(shape=[len(eps1Vec), len(eps2Vec)])
    for e1Ind, curr_e1 in enumerate(eps1Vec):
        for e2Ind, curr_e2 in enumerate(eps2Vec):
            currResultsList = decision3ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=curr_d,
                                                       eps1=curr_e1, eps2=curr_e2,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            eps1VSeps2_mat[e1Ind, e2Ind] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(eps2Vec, eps1Vec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, eps1VSeps2_mat * 100, cmap=cm.coolwarm)
    plt.suptitle(
        'Classification accuracy vs.\nDistance eps1, Distance eps2\nUnder Outlet 1 as culprit, Outlet 2 as eps1, t=0.3,n=300,delta=0.1')
    plt.xlabel('eps2', size=16)
    plt.ylabel('eps1', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    return