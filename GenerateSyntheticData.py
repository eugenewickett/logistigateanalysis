# -*- coding: utf-8 -*-
'''
Script that generates and analyzes a synthetic set of PMS data using 3 geographic levels. Each data point has

'''

from logistigate.logistigate import utilities as util # Pull from the submodule "develop" branch
from logistigate.logistigate import methods
from logistigate.logistigate import lg

def generateSyntheticData():
    '''
    Script for forming a synthetic data set. Significant SFP manifestation occurs at both test nodes and supply
    nodes.
    '''

    '''
    Use generated Q to produce 500 samples under specified random seeds
    '''
    import numpy as np
    import random

    Qrow = np.array([.01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01,
                     .02, .02, .02, .03, .03, .05, .05, .07, .07, .07, .10, .15, .20])
    random.seed(3)
    random.shuffle(Qrow)
    # Qrow: [0.01, 0.03, 0.1 , 0.02, 0.01, 0.01, 0.07, 0.01, 0.01, 0.02, 0.2, 0.02,
    #        0.01, 0.01, 0.07, 0.15, 0.01, 0.01, 0.03, 0.07, 0.01, 0.01, 0.05, 0.05, 0.01])


    # Overall SFP rate: ???
    # SN rates: 1% baseline; 20% node: 25%, 5% node: ~25/30%, 7% node: 10%, 2% node: 40%
    # TN rates: 1% baseline; 1 major node: 25%, 1 minor node: 30%; 3 minor nodes: 10%; 1 minor minor node: 50%

    numTN, numSN = 25, 25
    numSamples = 500
    s, r = 1.0, 1.0



    SNnames = ['Manufacturer ' + str(i + 1) for i in range(numSN)]
    TNnames = ['District ' + str(i + 1) for i in range(numTN)]

    trueRates = np.zeros(numSN + numTN)  # importers first, outlets second

    SNtrueRates = [.02 for i in range(numSN)]
    SN1ind = 3 # 40% SFP rate
    SN2ind = 10 # 25% SFP rate, major node
    SN3ind = 14 # 10% SFP rate, minor node
    SN4ind = 22 # 20% SFP rate, minor node
    SNtrueRates[SN1ind], SNtrueRates[SN2ind] = 0.35, 0.25
    SNtrueRates[SN3ind], SNtrueRates[SN4ind] = 0.1, 0.25

    trueRates[:numSN] = SNtrueRates # SN SFP rates

    TN1ind = 5 # 20% sampled node, 25% SFP rate
    TN2inds = [2, 11, 14, 22] # 10% sampled
    TN3inds = [3, 6, 8, 10, 16, 17, 24] # 3% sampled
    TN4inds = [0, 1, 9, 12, 18, 23] # 2% sampled
    TNsampProbs = [.01 for i in range(numTN)] # Update sampling probs
    TNsampProbs[TN1ind] = 0.20
    for j in TN2inds:
        TNsampProbs[j] = 0.10
    for j in TN3inds:
        TNsampProbs[j] = 0.03
    for j in TN4inds:
        TNsampProbs[j] = 0.02
    print(np.sum(TNsampProbs)) # sampling probability should add up to 1.0

    TNtrueRates = [.02 for i in range(numTN)] # Update SFP rates for TNs
    TNtrueRates[TN1ind] = 0.2
    TNtrueRates[TN2inds[1]] = 0.1
    TNtrueRates[TN2inds[2]] = 0.1
    TNtrueRates[TN3inds[1]] = 0.4
    trueRates[numSN:] = TNtrueRates # Put TN rates in main vector

    rseed = 56 # Change the seed here to get a different set of tests
    random.seed(rseed)
    np.random.seed(rseed+1)
    testingDataList = []
    for currSamp in range(numSamples):
        currTN = random.choices(TNnames, weights=TNsampProbs, k=1)[0]
        #if not currTN == 'District '
        currSN = random.choices(SNnames, weights=Qrow, k=1)[0] #[TNnames.index(currTN)] to index Q
        currTNrate = trueRates[numSN + TNnames.index(currTN)]
        currSNrate = trueRates[SNnames.index(currSN)]
        realRate = currTNrate + currSNrate - currTNrate * currSNrate
        realResult = np.random.binomial(1, p=realRate)
        if realResult == 1:
            result = np.random.binomial(1, p = s)
        if realResult == 0:
            result = np.random.binomial(1, p = 1. - r)
        testingDataList.append([currTN, currSN, result])

    # Inspect testing data; check: (1) overall SFP rate, (2) plots, (3) N, Y matrices align more or less with
    # statements from case-study section
    priorMean, priorScale = -2.5, 1.3
    numPostSamps = 1000
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}

    lowerQuant, upperQuant = 0.05, 0.95
    import scipy.special as spsp
    import scipy.stats as sps
    import matplotlib.pyplot as plt
    priorLower = spsp.expit(sps.laplace.ppf(lowerQuant, loc=priorMean, scale=priorScale))
    priorUpper = spsp.expit(sps.laplace.ppf(upperQuant, loc=priorMean, scale=priorScale))

    lgDict = util.testresultsfiletotable(testingDataList, csvName=False)
    print('size: '+str(lgDict['N'].shape)+', obsvns: '+str(lgDict['N'].sum())+', propor pos: '+str(lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=priorScale), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    numSN, numTN = lgDict['importerNum'], lgDict['outletNum']

    floorVal = 0.05 # Classification lines
    ceilVal = 0.25

    # Supply-node plot
    SNindsSubset = range(numSN)
    SNnames = [lgDict['importerNames'][i] for i in SNindsSubset]
    SNlowers = [np.quantile(lgDict['postSamples'][:, l], lowerQuant) for l in SNindsSubset]
    SNuppers = [np.quantile(lgDict['postSamples'][:, l], upperQuant) for l in SNindsSubset]
    # First group
    SNlowers1 = [i for i in SNlowers if i > floorVal]
    SNuppers1 = [SNuppers[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    SNnames1 = [SNnames[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    midpoints1 = [SNuppers1[i] - (SNuppers1[i] - SNlowers1[i]) / 2 for i in range(len(SNuppers1))]
    zippedList1 = zip(midpoints1, SNuppers1, SNlowers1, SNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    SNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    SNuppers2 = [i for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers2 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNnames2 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    midpoints2 = [SNuppers2[i] - (SNuppers2[i] - SNlowers2[i]) / 2 for i in range(len(SNuppers2))]
    zippedList2 = zip(midpoints2, SNuppers2, SNlowers2, SNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    SNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    SNuppers3 = [i for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers3 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNnames3 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    midpoints3 = [SNuppers3[i] - (SNuppers3[i] - SNlowers3[i]) / 2 for i in range(len(SNuppers3))]
    zippedList3 = zip(midpoints3, SNuppers3, SNlowers3, SNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    SNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    SNnamesSorted = SNnamesSorted1.copy()
    SNnamesSorted.append(' ')
    SNnamesSorted = SNnamesSorted + SNnamesSorted2
    SNnamesSorted.append(' ')
    SNnamesSorted = SNnamesSorted + SNnamesSorted3
    SNnamesSorted.append(' ')
    SNnamesSorted.append('(Prior)')

    fig, (ax) = plt.subplots(figsize=(10, 6), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o--', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o--', color='orange')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o:', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o:', color='green')
    plt.plot((SNnamesSorted[-1], SNnamesSorted[-1]), (priorLower, priorUpper), 'o-', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(SNnamesSorted)), SNnamesSorted, rotation=90)
    plt.title('Supply Node 90% Intervals\nManufacturer-District Analysis, Tracked Setting',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Supply Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    plt.axhline(y=floorVal, color='r', linestyle='-', alpha=0.1)  # line for 'l'
    plt.axhline(y=ceilVal, color='blue', linestyle='-', alpha=0.1)  # line for 'u'
    plt.text(26.3, ceilVal + .015, 'u=0.25', color='blue', alpha=0.5, size=9)
    plt.text(26.3, floorVal + .015, 'l=0.05', color='r', alpha=0.5, size=9)
    fig.tight_layout()
    plt.show()
    plt.close()

    # Test-node plot
    TNindsSubset = range(numTN)
    TNnames = [lgDict['outletNames'][i] for i in TNindsSubset]
    TNlowers = [np.quantile(lgDict['postSamples'][:, numSN + l], lowerQuant) for l in TNindsSubset]
    TNuppers = [np.quantile(lgDict['postSamples'][:, numSN + l], upperQuant) for l in TNindsSubset]
    # First group
    TNlowers1 = [i for i in TNlowers if i > floorVal]
    TNuppers1 = [TNuppers[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    TNnames1 = [TNnames[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    midpoints1 = [TNuppers1[i] - (TNuppers1[i] - TNlowers1[i]) / 2 for i in range(len(TNuppers1))]
    zippedList1 = zip(midpoints1, TNuppers1, TNlowers1, TNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    TNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    TNuppers2 = [i for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers2 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNnames2 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    midpoints2 = [TNuppers2[i] - (TNuppers2[i] - TNlowers2[i]) / 2 for i in range(len(TNuppers2))]
    zippedList2 = zip(midpoints2, TNuppers2, TNlowers2, TNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    TNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    TNuppers3 = [i for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers3 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNnames3 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    midpoints3 = [TNuppers3[i] - (TNuppers3[i] - TNlowers3[i]) / 2 for i in range(len(TNuppers3))]
    zippedList3 = zip(midpoints3, TNuppers3, TNlowers3, TNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    TNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    TNnamesSorted = TNnamesSorted1.copy()
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted2
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted3
    TNnamesSorted.append(' ')
    TNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(10, 6), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o--', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o--', color='orange')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o:', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o:', color='green')
    plt.plot((TNnamesSorted[-1], TNnamesSorted[-1]), (priorLower, priorUpper), 'o-', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(TNnamesSorted)), TNnamesSorted, rotation=90)
    plt.title('Test Node 90% Intervals\nManufacturer-District Analysis, Tracked Setting',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Test Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    plt.axhline(y=floorVal, color='r', linestyle='-', alpha=0.1)  # line for 'l'
    plt.axhline(y=ceilVal, color='blue', linestyle='-', alpha=0.1)  # line for 'u'
    plt.text(26.4, ceilVal + .015, 'u=0.25', color='blue', alpha=0.5, size=9)
    plt.text(26.4, floorVal + .015, 'l=0.05', color='r', alpha=0.5, size=9)
    fig.tight_layout()
    plt.show()
    plt.close()


    # How many observed arcs are there?
    np.count_nonzero(lgDict['N'])

    # Inspect raw data totals
    # Supply nodes
    for i in range(numSN): # sum across TNs to see totals for SNs
        currTotal = np.sum(lgDict['N'],axis=0)[i]
        currPos = np.sum(lgDict['Y'],axis=0)[i]
        print(lgDict['importerNames'][i]+': ' +str(currTotal)[:-2]+' samples, '
              + str(currPos)[:-2] + ' positives, ' + str(currPos/currTotal)[:5] + ' rate')
    # Test nodes
    for i in range(numTN): # sum across SNs to see totals for TNs
        currTotal = np.sum(lgDict['N'],axis=1)[i]
        currPos = np.sum(lgDict['Y'],axis=1)[i]
        print(lgDict['outletNames'][i]+': ' +str(currTotal)[:-2]+' samples, '
              + str(currPos)[:-2] + ' positives, ' + str(currPos/currTotal)[:5] + ' rate')

    # SNs, TNs with at least ten samples and 10% SFP rate
    for i in range(numSN): # sum across TNs to see totals for SNs
        currTotal = np.sum(lgDict['N'],axis=0)[i]
        currPos = np.sum(lgDict['Y'],axis=0)[i]
        if currPos/currTotal>0.1 and currTotal>10:
            print(lgDict['importerNames'][i]+': ' +str(currTotal)[:-2]+' samples, '
              + str(currPos)[:-2] + ' positives, ' + str(currPos/currTotal)[:5] + ' rate')
    # Test nodes
    for i in range(numTN): # sum across SNs to see totals for TNs
        currTotal = np.sum(lgDict['N'],axis=1)[i]
        currPos = np.sum(lgDict['Y'],axis=1)[i]
        if currPos / currTotal > 0.1 and currTotal > 10:
            print(lgDict['outletNames'][i]+': ' +str(currTotal)[:-2]+' samples, '
              + str(currPos)[:-2] + ' positives, ' + str(currPos/currTotal)[:5] + ' rate')

    # 90% intervals for SFP rates at SNs, TNs, using proportion CI
    for i in range(numSN):  # sum across TNs to see totals for SNs
        currTotal = np.sum(lgDict['N'], axis=0)[i]
        currPos = np.sum(lgDict['Y'], axis=0)[i]
        pHat = currPos/currTotal
        lowerBd = pHat-(1.645*np.sqrt(pHat*(1-pHat)/currTotal))
        upperBd = pHat+(1.645*np.sqrt(pHat*(1-pHat)/currTotal))
        print(lgDict['importerNames'][i]+': ('+str(lowerBd)[:5]+', '+str(upperBd)[:5]+')')
    # Test nodes
    for i in range(numTN):  # sum across SNs to see totals for TNs
        currTotal = np.sum(lgDict['N'], axis=1)[i]
        currPos = np.sum(lgDict['Y'], axis=1)[i]
        pHat = currPos / currTotal
        lowerBd = pHat - (1.645 * np.sqrt(pHat * (1 - pHat) / currTotal))
        upperBd = pHat + (1.645 * np.sqrt(pHat * (1 - pHat) / currTotal))
        print(lgDict['outletNames'][i] + ': (' + str(lowerBd)[:5] + ', ' + str(upperBd)[:5] + ')')


    # Print quantiles for analysis tables
    SNinds = lgDict['importerNames'].index('Manufacturer 4')
    print('Manufacturer 4: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    SNinds = lgDict['importerNames'].index('Manufacturer 11')
    print('Manufacturer 11: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    SNinds = lgDict['importerNames'].index('Manufacturer 23')
    print('Manufacturer 23: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    TNinds = lgDict['outletNames'].index('District 6')
    print('District 6: (' + str(np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.05))[
        :5] + ',' + str(np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.95))[:5] + ')')
    TNinds = lgDict['outletNames'].index('District 7')
    print('District 7: (' + str(np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.05))[
        :5] + ',' + str(np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.95))[:5] + ')')

    # Untracked
    lgDict = {}
    lgDict = util.testresultsfiletotable(testingDataList, csvName=False)
    Qest = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Qest[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked','diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=priorScale), 'MCMCdict': MCMCdict,
                   'transMat': Qest, 'importerNum': Qest.shape[1], 'outletNum': Qest.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    numSN, numTN = lgDict['importerNum'], lgDict['outletNum']

    SNindsSubset = range(numSN)
    SNnames = [lgDict['importerNames'][i] for i in SNindsSubset]
    SNlowers = [np.quantile(lgDict['postSamples'][:, l], lowerQuant) for l in SNindsSubset]
    SNuppers = [np.quantile(lgDict['postSamples'][:, l], upperQuant) for l in SNindsSubset]
    # First group
    SNlowers1 = [i for i in SNlowers if i > floorVal]
    SNuppers1 = [SNuppers[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    SNnames1 = [SNnames[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    midpoints1 = [SNuppers1[i] - (SNuppers1[i] - SNlowers1[i]) / 2 for i in range(len(SNuppers1))]
    zippedList1 = zip(midpoints1, SNuppers1, SNlowers1, SNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    SNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    SNuppers2 = [i for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers2 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNnames2 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    midpoints2 = [SNuppers2[i] - (SNuppers2[i] - SNlowers2[i]) / 2 for i in range(len(SNuppers2))]
    zippedList2 = zip(midpoints2, SNuppers2, SNlowers2, SNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    SNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    SNuppers3 = [i for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers3 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNnames3 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    midpoints3 = [SNuppers3[i] - (SNuppers3[i] - SNlowers3[i]) / 2 for i in range(len(SNuppers3))]
    zippedList3 = zip(midpoints3, SNuppers3, SNlowers3, SNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    SNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    SNnamesSorted = SNnamesSorted1.copy()
    SNnamesSorted.append(' ')
    SNnamesSorted = SNnamesSorted + SNnamesSorted2
    SNnamesSorted.append(' ')
    SNnamesSorted = SNnamesSorted + SNnamesSorted3
    SNnamesSorted.append(' ')
    SNnamesSorted.append('(Prior)')

    fig, (ax) = plt.subplots(figsize=(10, 6), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o--', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o--', color='orange')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o:', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o:', color='green')
    plt.plot((SNnamesSorted[-1], SNnamesSorted[-1]), (priorLower, priorUpper), 'o-', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(SNnamesSorted)), SNnamesSorted, rotation=90)
    plt.title('Supply Node 90% Intervals\nManufacturer-District Analysis, Untracked Setting',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Supply Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    plt.axhline(y=floorVal, color='r', linestyle='-', alpha=0.1)  # line for 'l'
    plt.axhline(y=ceilVal, color='blue', linestyle='-', alpha=0.1)  # line for 'u'
    plt.text(26.3, ceilVal + .015, 'u=0.25', color='blue', alpha=0.5, size=9)
    plt.text(26.3, floorVal + .015, 'l=0.05', color='r', alpha=0.5, size=9)
    fig.tight_layout()
    plt.show()
    plt.close()

    # Test-node plot
    TNindsSubset = range(numTN)
    TNnames = [lgDict['outletNames'][i] for i in TNindsSubset]
    TNlowers = [np.quantile(lgDict['postSamples'][:, numSN + l], lowerQuant) for l in TNindsSubset]
    TNuppers = [np.quantile(lgDict['postSamples'][:, numSN + l], upperQuant) for l in TNindsSubset]
    # First group
    TNlowers1 = [i for i in TNlowers if i > floorVal]
    TNuppers1 = [TNuppers[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    TNnames1 = [TNnames[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    midpoints1 = [TNuppers1[i] - (TNuppers1[i] - TNlowers1[i]) / 2 for i in range(len(TNuppers1))]
    zippedList1 = zip(midpoints1, TNuppers1, TNlowers1, TNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    TNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    TNuppers2 = [i for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers2 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNnames2 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    midpoints2 = [TNuppers2[i] - (TNuppers2[i] - TNlowers2[i]) / 2 for i in range(len(TNuppers2))]
    zippedList2 = zip(midpoints2, TNuppers2, TNlowers2, TNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    TNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    TNuppers3 = [i for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers3 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNnames3 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    midpoints3 = [TNuppers3[i] - (TNuppers3[i] - TNlowers3[i]) / 2 for i in range(len(TNuppers3))]
    zippedList3 = zip(midpoints3, TNuppers3, TNlowers3, TNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    TNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    TNnamesSorted = TNnamesSorted1.copy()
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted2
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted3
    TNnamesSorted.append(' ')
    TNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(10, 6), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o--', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o--', color='orange')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o:', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o:', color='green')
    plt.plot((TNnamesSorted[-1], TNnamesSorted[-1]), (priorLower, priorUpper), 'o-', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(TNnamesSorted)), TNnamesSorted, rotation=90)
    plt.title('Test Node 90% Intervals\nManufacturer-District Analysis, Untracked Setting',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Test Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    plt.axhline(y=floorVal, color='r', linestyle='-', alpha=0.1)  # line for 'l'
    plt.axhline(y=ceilVal, color='blue', linestyle='-', alpha=0.1)  # line for 'u'
    plt.text(26.4, ceilVal + .015, 'u=0.25', color='blue', alpha=0.5, size=9)
    plt.text(26.4, floorVal + .015, 'l=0.05', color='r', alpha=0.5, size=9)
    fig.tight_layout()
    plt.show()
    plt.close()

    # Generate data with a different Q, same underlying SFP rates, UNTRACKED
    Qrow = np.array([.01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01,
                     .02, .02, .02, .03, .03, .05, .05, .07, .07, .07, .10, .15, .20])

    Q = np.zeros(shape=(numTN,numSN))
    #base = 4
    for i in range(numTN):
        Qrow = np.array([.0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0,
                         .0, .0, .0, .0, .0, .0, .0, .0, .0, .1, .1, .2, .6])
        '''
        if i < 4:
            random.seed(4+base)
        elif i >= 4 and i < 8:
            random.seed(5+base)
        elif i >= 8 and i < 12:
            random.seed(6+base)
        elif i >= 12 and i < 16:
            random.seed(7+base)
        elif i >= 16:
            random.seed(8+base)
        '''
        random.seed(i+10)
        random.shuffle(Qrow)
        random.shuffle(Qrow)
        Q[i] = Qrow
    '''
    for i in range(numSN):
        if np.sum(Q[:,i]) == 0.0:
            print(i)
        print(np.sum(Q[:,i]))
    '''

    # Overall SFP rate: 10-20%
    # SN rates: 1% baseline; 20% node: 25%, 5% node: ~25/30%, 7% node: 10%, 2% node: 40%
    # TN rates: 1% baseline; 1 major node: 25%, 1 minor node: 30%; 3 minor nodes: 10%; 1 minor minor node: 50%

    numTN, numSN = 25, 25
    numSamples = 500
    s, r = 1.0, 1.0

    SNnames = ['Manufacturer ' + str(i + 1) for i in range(numSN)]
    TNnames = ['District ' + str(i + 1) for i in range(numTN)]

    trueRates = np.zeros(numSN + numTN)  # importers first, outlets second

    SNtrueRates = [.02 for i in range(numSN)]
    SN1ind = 3  # 40% SFP rate
    SN2ind = 10  # 25% SFP rate, major node
    SN3ind = 14  # 10% SFP rate, minor node
    SN4ind = 22  # 20% SFP rate, minor node
    SNtrueRates[SN1ind], SNtrueRates[SN2ind] = 0.35, 0.25
    SNtrueRates[SN3ind], SNtrueRates[SN4ind] = 0.1, 0.25

    trueRates[:numSN] = SNtrueRates  # SN SFP rates

    TN1ind = 5  # 20% sampled node, 25% SFP rate
    TN2inds = [2, 11, 14, 22]  # 10% sampled
    TN3inds = [3, 6, 8, 10, 16, 17, 24]  # 3% sampled
    TN4inds = [0, 1, 9, 12, 18, 23]  # 2% sampled
    TNsampProbs = [.01 for i in range(numTN)]  # Update sampling probs
    TNsampProbs[TN1ind] = 0.20
    for j in TN2inds:
        TNsampProbs[j] = 0.10
    for j in TN3inds:
        TNsampProbs[j] = 0.03
    for j in TN4inds:
        TNsampProbs[j] = 0.02
    print(np.sum(TNsampProbs))  # sampling probability should add up to 1.0

    TNtrueRates = [.01 for i in range(numTN)]  # Update SFP rates for TNs
    TNtrueRates[TN1ind] = 0.2
    TNtrueRates[TN2inds[1]] = 0.1
    TNtrueRates[TN2inds[2]] = 0.1
    TNtrueRates[TN3inds[1]] = 0.4
    trueRates[numSN:] = TNtrueRates  # Put TN rates in main vector

    rseed = 56  # Change the seed here to get a different set of tests
    random.seed(rseed)
    np.random.seed(rseed + 1)
    testingDataList = []
    for currSamp in range(numSamples):
        currTN = random.choices(TNnames, weights=TNsampProbs, k=1)[0]
        currTNind = TNnames.index(currTN)
        # if not currTN == 'District '
        currSN = random.choices(SNnames, weights=Q[currTNind], k=1)[0]  # [TNnames.index(currTN)] to index Q
        currTNrate = trueRates[numSN + TNnames.index(currTN)]
        currSNrate = trueRates[SNnames.index(currSN)]
        realRate = currTNrate + currSNrate - currTNrate * currSNrate
        realResult = np.random.binomial(1, p=realRate)
        if realResult == 1:
            result = np.random.binomial(1, p=s)
        if realResult == 0:
            result = np.random.binomial(1, p=1. - r)
        testingDataList.append([currTN, currSN, result])

    lgDict = {}
    lgDict = util.testresultsfiletotable(testingDataList, csvName=False)
    Qest = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Qest[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=priorScale), 'MCMCdict': MCMCdict,
                   'transMat': Qest, 'importerNum': Qest.shape[1], 'outletNum': Qest.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    numSN, numTN = lgDict['importerNum'], lgDict['outletNum']

    SNindsSubset = range(numSN)
    SNnames = [lgDict['importerNames'][i] for i in SNindsSubset]
    SNlowers = [np.quantile(lgDict['postSamples'][:, l], lowerQuant) for l in SNindsSubset]
    SNuppers = [np.quantile(lgDict['postSamples'][:, l], upperQuant) for l in SNindsSubset]
    # First group
    SNlowers1 = [i for i in SNlowers if i > floorVal]
    SNuppers1 = [SNuppers[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    SNnames1 = [SNnames[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    midpoints1 = [SNuppers1[i] - (SNuppers1[i] - SNlowers1[i]) / 2 for i in range(len(SNuppers1))]
    zippedList1 = zip(midpoints1, SNuppers1, SNlowers1, SNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    SNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    SNuppers2 = [i for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers2 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNnames2 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    midpoints2 = [SNuppers2[i] - (SNuppers2[i] - SNlowers2[i]) / 2 for i in range(len(SNuppers2))]
    zippedList2 = zip(midpoints2, SNuppers2, SNlowers2, SNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    SNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    SNuppers3 = [i for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers3 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNnames3 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    midpoints3 = [SNuppers3[i] - (SNuppers3[i] - SNlowers3[i]) / 2 for i in range(len(SNuppers3))]
    zippedList3 = zip(midpoints3, SNuppers3, SNlowers3, SNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    SNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    SNnamesSorted = SNnamesSorted1.copy()
    SNnamesSorted.append(' ')
    SNnamesSorted = SNnamesSorted + SNnamesSorted2
    SNnamesSorted.append(' ')
    SNnamesSorted = SNnamesSorted + SNnamesSorted3
    SNnamesSorted.append(' ')
    SNnamesSorted.append('(Prior)')

    fig, (ax) = plt.subplots(figsize=(10, 6), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o--', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o--', color='orange')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o:', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o:', color='green')
    plt.plot((SNnamesSorted[-1], SNnamesSorted[-1]), (priorLower, priorUpper), 'o-', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(SNnamesSorted)), SNnamesSorted, rotation=90)
    plt.title('Supply Node 90% Intervals\nManufacturer-District Analysis, Untracked Setting',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Supply Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    plt.axhline(y=floorVal, color='r', linestyle='-', alpha=0.1)  # line for 'l'
    plt.axhline(y=ceilVal, color='blue', linestyle='-', alpha=0.1)  # line for 'u'
    plt.text(26.3, ceilVal + .015, 'u=0.25', color='blue', alpha=0.5, size=9)
    plt.text(26.3, floorVal + .015, 'l=0.05', color='r', alpha=0.5, size=9)
    fig.tight_layout()
    plt.show()
    plt.close()

    # Test-node plot
    TNindsSubset = range(numTN)
    TNnames = [lgDict['outletNames'][i] for i in TNindsSubset]
    TNlowers = [np.quantile(lgDict['postSamples'][:, numSN + l], lowerQuant) for l in TNindsSubset]
    TNuppers = [np.quantile(lgDict['postSamples'][:, numSN + l], upperQuant) for l in TNindsSubset]
    # First group
    TNlowers1 = [i for i in TNlowers if i > floorVal]
    TNuppers1 = [TNuppers[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    TNnames1 = [TNnames[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    midpoints1 = [TNuppers1[i] - (TNuppers1[i] - TNlowers1[i]) / 2 for i in range(len(TNuppers1))]
    zippedList1 = zip(midpoints1, TNuppers1, TNlowers1, TNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    TNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    TNuppers2 = [i for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers2 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNnames2 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    midpoints2 = [TNuppers2[i] - (TNuppers2[i] - TNlowers2[i]) / 2 for i in range(len(TNuppers2))]
    zippedList2 = zip(midpoints2, TNuppers2, TNlowers2, TNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    TNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    TNuppers3 = [i for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers3 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNnames3 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    midpoints3 = [TNuppers3[i] - (TNuppers3[i] - TNlowers3[i]) / 2 for i in range(len(TNuppers3))]
    zippedList3 = zip(midpoints3, TNuppers3, TNlowers3, TNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    TNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    TNnamesSorted = TNnamesSorted1.copy()
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted2
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted3
    TNnamesSorted.append(' ')
    TNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(10, 6), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o--', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o--', color='orange')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o:', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o:', color='green')
    plt.plot((TNnamesSorted[-1], TNnamesSorted[-1]), (priorLower, priorUpper), 'o-', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(TNnamesSorted)), TNnamesSorted, rotation=90)
    plt.title('Test Node 90% Intervals\nManufacturer-District Analysis, Untracked Setting',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Test Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    plt.axhline(y=floorVal, color='r', linestyle='-', alpha=0.1)  # line for 'l'
    plt.axhline(y=ceilVal, color='blue', linestyle='-', alpha=0.1)  # line for 'u'
    plt.text(26.4, ceilVal + .015, 'u=0.25', color='blue', alpha=0.5, size=9)
    plt.text(26.4, floorVal + .015, 'l=0.05', color='r', alpha=0.5, size=9)
    fig.tight_layout()
    plt.show()
    plt.close()


    #np.savetxt("N_matrix.csv", lgDict['N'], delimiter=",")
    #np.savetxt("Y_matrix.csv", lgDict['Y'], delimiter=",")

    # NEED TO DO:
    #   GROUP DISTRICTS INTO PROVINCES, RE-RUN
    #   SEPARATE DISTRICTS INTO FACILITIES, RE-RUN
    #





    return
