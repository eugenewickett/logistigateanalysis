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

def generateSyntheticData():
    '''
    Script for forming a synthetic data set of 25 test nodes and 25 supply nodes.
    '''

    '''
    Use a generated sourcing-probability matrix to produce 500 samples under specified random seeds
    '''
    import random

    Qrow = np.array([.01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01,
                     .02, .02, .02, .03, .03, .05, .05, .07, .07, .07, .10, .15, .20])
    random.seed(3)
    random.shuffle(Qrow)
    # Qrow: [0.01, 0.03, 0.1 , 0.02, 0.01, 0.01, 0.07, 0.01, 0.01, 0.02, 0.2, 0.02,
    #        0.01, 0.01, 0.07, 0.15, 0.01, 0.01, 0.03, 0.07, 0.01, 0.01, 0.05, 0.05, 0.01])

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
    #print(np.sum(TNsampProbs)) # sampling probability should add up to 1.0

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
    #np.count_nonzero(lgDict['N'])

    '''
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
    '''

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

    # Run tracked again for completing sensitivity analyses
    priorMean, priorScale = -2.5, 1.3
    numPostSamps = 1000
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    lgDict = util.testresultsfiletotable(testingDataList, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=priorScale), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    Manu4list = [np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 4')], 0.95) -
                  np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 4')], 0.05)]
    Manu11list = [np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 11')], 0.95) -
                  np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 11')], 0.05)]
    Manu23list = [np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 23')], 0.95) -
                  np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 23')], 0.05)]
    Dist6list = [np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 6')], 0.95) -
                 np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 6')], 0.05)]
    Dist7list = [np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 7')], 0.95) -
                 np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 7')], 0.05)]
    Manu4list_prior, Manu11list_prior, Manu23list_prior = Manu4list.copy(), Manu11list.copy(), Manu23list.copy()
    Dist6list_prior, Dist7list_prior = Dist6list.copy(), Dist7list.copy()

    # Sensitivity analysis for Table 2; calculate interval widths for Manufacturers 4, 11, 23, and Districts 6, 7.
    # s=0.8,r=1.0
    lgDict = util.testresultsfiletotable(testingDataList, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 0.8, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=priorScale), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    Manu4list.append(np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 4')], 0.95) -
                     np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 4')], 0.05))
    Manu11list.append(np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 11')], 0.95) -
                      np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 11')], 0.05))
    Manu23list.append(np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 23')], 0.95) -
                      np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 23')], 0.05))
    Dist6list.append(np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 6')], 0.95) -
                     np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 6')], 0.05))
    Dist7list.append(np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 7')], 0.95) -
                     np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 7')], 0.05))
    # s=1.0,r=0.95
    lgDict = util.testresultsfiletotable(testingDataList, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 0.95, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=priorScale), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    Manu4list.append(np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 4')], 0.95) -
                     np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 4')], 0.05))
    Manu11list.append(np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 11')], 0.95) -
                      np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 11')], 0.05))
    Manu23list.append(np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 23')], 0.95) -
                      np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 23')], 0.05))
    Dist6list.append(np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 6')], 0.95) -
                     np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 6')], 0.05))
    Dist7list.append(np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 7')], 0.95) -
                     np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 7')], 0.05))
    # s=0.8,r=0.95
    lgDict = util.testresultsfiletotable(testingDataList, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 0.8, 'diagSpec': 0.95, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=priorScale), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    Manu4list.append(np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 4')], 0.95) -
                     np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 4')], 0.05))
    Manu11list.append(np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 11')], 0.95) -
                      np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 11')], 0.05))
    Manu23list.append(np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 23')], 0.95) -
                      np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 23')], 0.05))
    Dist6list.append(np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 6')], 0.95) -
                     np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 6')], 0.05))
    Dist7list.append(np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 7')], 0.95) -
                     np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 7')], 0.05))

    # Sensitivity analysis for Table 3; calculate interval widths for Manufacturers 4, 11, 23, and Districts 6, 7.
    # mean = -3.5, scale = 1.3, Laplace prior
    priorMean, priorScale = -3.5, 1.3
    lgDict = util.testresultsfiletotable(testingDataList, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=priorScale), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    Manu4list_prior.append(np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 4')], 0.95) -
                     np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 4')], 0.05))
    Manu11list_prior.append(np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 11')], 0.95) -
                      np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 11')], 0.05))
    Manu23list_prior.append(np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 23')], 0.95) -
                      np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 23')], 0.05))
    Dist6list_prior.append(np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 6')], 0.95) -
                     np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 6')], 0.05))
    Dist7list_prior.append(np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 7')], 0.95) -
                     np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 7')], 0.05))
    # mean = -2.5, scale = 0.87, Laplace prior
    priorMean, priorScale = -2.5, 0.87
    lgDict = util.testresultsfiletotable(testingDataList, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=priorScale), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    Manu4list_prior.append(
        np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 4')], 0.95) -
        np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 4')], 0.05))
    Manu11list_prior.append(
        np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 11')], 0.95) -
        np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 11')], 0.05))
    Manu23list_prior.append(
        np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 23')], 0.95) -
        np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 23')], 0.05))
    Dist6list_prior.append(
        np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 6')], 0.95) -
        np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 6')], 0.05))
    Dist7list_prior.append(
        np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 7')], 0.95) -
        np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 7')], 0.05))
    # mean = -2.5, variance = 3.38, normal prior
    priorMean, priorVar = -2.5, 3.38
    lgDict = util.testresultsfiletotable(testingDataList, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    Manu4list_prior.append(
        np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 4')], 0.95) -
        np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 4')], 0.05))
    Manu11list_prior.append(
        np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 11')], 0.95) -
        np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 11')], 0.05))
    Manu23list_prior.append(
        np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 23')], 0.95) -
        np.quantile(lgDict['postSamples'][:, lgDict['importerNames'].index('Manufacturer 23')], 0.05))
    Dist6list_prior.append(
        np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 6')], 0.95) -
        np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 6')], 0.05))
    Dist7list_prior.append(
        np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 7')], 0.95) -
        np.quantile(lgDict['postSamples'][:, numSN + lgDict['outletNames'].index('District 7')], 0.05))

    # Generate tables
    mainTitle = 'Interval widths for different testing sensitivity and specificity\n'
    header = '| Node Name      ' + '| s=1.0,r=1.0  ' + '| s=0.8,r=1.0  ' + '| s=1.0,r=0.95 ' + '| s=0.8,r=0.95 |\n'
    row1 = '| Manufacturer 4 | ' + ' | '.join([str(i)[:4].ljust(12) for i in Manu4list]) + ' | \n'
    row2 = '| Manufacturer 11| ' + ' | '.join([str(i)[:4].ljust(12) for i in Manu11list]) + ' | \n'
    row3 = '| Manufacturer 23| ' + ' | '.join([str(i)[:4].ljust(12) for i in Manu23list]) + ' | \n'
    row4 = '| District 6     | ' + ' | '.join([str(i)[:4].ljust(12) for i in Dist6list]) + ' | \n'
    row5 = '| District 7     | ' + ' | '.join([str(i)[:4].ljust(12) for i in Dist7list]) + ' | \n'
    print(mainTitle + header + row1 + row2 + row3 + row4 + row5)

    mainTitle = 'Interval widths for different prior selections; last column is a normal prior\n'
    header = '| Node Name       ' + '| gamma=-2.5, nu=1.3  ' + '| gamma=-3.5, nu=1.3  ' + '| gamma=-2.5, nu=0.87 ' + '| gamma=-2.5,var=3.38 |\n'
    row1 = '| Manufacturer 4  | ' + ' | '.join([str(i)[:4].ljust(19) for i in Manu4list]) + ' | \n'
    row2 = '| Manufacturer 11 | ' + ' | '.join([str(i)[:4].ljust(19) for i in Manu11list]) + ' | \n'
    row3 = '| Manufacturer 23 | ' + ' | '.join([str(i)[:4].ljust(19) for i in Manu23list]) + ' | \n'
    row4 = '| District 6      | ' + ' | '.join([str(i)[:4].ljust(19) for i in Dist6list]) + ' | \n'
    row5 = '| District 7      | ' + ' | '.join([str(i)[:4].ljust(19) for i in Dist7list]) + ' | \n'
    print(mainTitle + header + row1 + row2 + row3 + row4 + row5)

    return

def generateExampleInference():
    '''
    Use data from example of Section 3 to infer SFP rates.
    '''
    import scipy.stats as sps
    import scipy.special as spsp
    import numpy as np
    import matplotlib.pyplot as plt

    lgDict = {}
    priorMean, priorVar = -2, 1
    numPostSamps = 1000
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    int50 = sps.norm.ppf(0.50, loc=priorMean, scale=np.sqrt(priorVar))
    int05 = sps.norm.ppf(0.05, loc=priorMean, scale=np.sqrt(priorVar))
    int95 = sps.norm.ppf(0.95, loc=priorMean, scale=np.sqrt(priorVar))
    int70 = sps.norm.ppf(0.70, loc=priorMean, scale=np.sqrt(priorVar))
    #print(spsp.expit(int05), spsp.expit(int50), spsp.expit(int70), spsp.expit(int95))
    Ntoy = np.array([[6, 11], [12, 6], [2, 13]])
    Ytoy = np.array([[3, 0], [6, 0], [0, 0]])
    TNnames, SNnames = ['Test Node 1', 'Test Node 2', 'Test Node 3'], ['Supply Node 1', 'Supply Node 2']
    lgDict.update({'type': 'Tracked', 'outletNum': 3, 'importerNum': 2, 'diagSens': 1.0, 'diagSpec': 1.0,
                   'N': Ntoy, 'Y': Ytoy, 'numPostSamples': numPostSamps, 'MCMCdict': MCMCdict,
                   'outletNames': TNnames, 'importerNames': SNnames,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar)})
    lgDict = methods.GeneratePostSamples(lgDict)
    numSN, numTN = lgDict['importerNum'], lgDict['outletNum']
    lowerQuant, upperQuant = 0.05, 0.95
    priorLower = spsp.expit(sps.norm.ppf(lowerQuant, loc=priorMean, scale=np.sqrt(priorVar)))
    priorUpper = spsp.expit(sps.norm.ppf(upperQuant, loc=priorMean, scale=np.sqrt(priorVar)))
    SNindsSubset = range(numSN)

    SNnames = [lgDict['importerNames'][i] for i in SNindsSubset]
    SNlowers = [np.quantile(lgDict['postSamples'][:, l], lowerQuant) for l in SNindsSubset]
    SNuppers = [np.quantile(lgDict['postSamples'][:, l], upperQuant) for l in SNindsSubset]

    TNindsSubset = range(numTN)
    TNnames = [lgDict['outletNames'][i] for i in TNindsSubset]
    TNlowers = [np.quantile(lgDict['postSamples'][:, numSN + l], lowerQuant) for l in TNindsSubset]
    TNuppers = [np.quantile(lgDict['postSamples'][:, numSN + l], upperQuant) for l in TNindsSubset]

    floorVal = 0.05
    ceilVal = 0.2
    # First group
    SNlowers1 = [i for i in SNlowers if i > floorVal]
    SNuppers1 = [SNuppers[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    SNnames1 = [SNnames[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    midpoints1 = [SNuppers1[i] - (SNuppers1[i] - SNlowers1[i]) / 2 for i in range(len(SNuppers1))]
    zippedList1 = zip(midpoints1, SNuppers1, SNlowers1, SNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    SNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
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
    TNmidpoints3 = [TNuppers3[i] - (TNuppers3[i] - TNlowers3[i]) / 2 for i in range(len(TNuppers3))]
    TNzippedList3 = zip(TNmidpoints3, TNuppers3, TNlowers3, TNnames3)
    TNsorted_pairs3 = sorted(TNzippedList3, reverse=True)
    TNnamesSorted3 = [tup[-1] for tup in TNsorted_pairs3]

    SNuppers3 = [i for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers3 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNnames3 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNmidpoints3 = [SNuppers3[i] - (SNuppers3[i] - SNlowers3[i]) / 2 for i in range(len(SNuppers3))]
    SNzippedList3 = zip(SNmidpoints3, SNuppers3, SNlowers3, SNnames3)
    SNsorted_pairs3 = sorted(SNzippedList3, reverse=True)
    SNnamesSorted3 = [tup[-1] for tup in SNsorted_pairs3]

    midpoints3 = TNmidpoints3 + SNmidpoints3
    uppers3 = TNuppers3 + SNuppers3
    lowers3 = TNlowers3 + SNlowers3
    names3 = TNnames3 + SNnames3
    zippedList3 = zip(midpoints3, uppers3, lowers3, names3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)

    # Combine groups
    namesSorted = SNnamesSorted1.copy()
    # namesSorted.append(' ')
    namesSorted = namesSorted + TNnamesSorted2
    # namesSorted.append(' ')
    namesSorted = namesSorted + TNnamesSorted3 + SNnamesSorted3
    namesSorted.append(' ')
    namesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(5, 5), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        # plt.plot((name, name), (lower, upper), 'o-', color='red')
        plt.plot((name, name), (lower, upper), 'o-', color='blue')
    #   plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        # plt.plot((name, name), (lower, upper), 'o--', color='orange')
        plt.plot((name, name), (lower, upper), 'o-', color='blue')
    # plt.plot((' ', ' '), (np.nan, np.nan), 'o--', color='orange')
    for _, upper, lower, name in sorted_pairs3:
        # plt.plot((name, name), (lower, upper), 'o:', color='green')
        plt.plot((name, name), (lower, upper), 'o-', color='blue')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o:', color='green')
    plt.plot((namesSorted[-1], namesSorted[-1]), (priorLower, priorUpper), 'o-', color='gray')
    plt.ylim([0, 0.6])
    plt.xticks(range(len(namesSorted)), namesSorted, rotation=90)
    plt.title('Node 90% Intervals',
              fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Node Name', fontdict={'fontsize': 12, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 12, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(11)
    # plt.axhline(y=floorVal, color='r', linestyle='-', alpha=0.1)  # line for 'l'
    # plt.axhline(y=ceilVal, color='blue', linestyle='-', alpha=0.1)  # line for 'u'
    # plt.text(6.7, 0.215, 'u=0.20', color='blue', alpha=0.5)
    # plt.text(6.7, 0.065, 'l=0.05', color='r', alpha=0.5)
    fig.tight_layout()
    plt.show()
    plt.close()

    return

def timingAnalysis():
    '''
    Computation of run times under different scenarios that fills Table 1. This function may take upwards of an hour
    to complete.
    '''
    import time

    numPostSamps = 1000
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}

    # Look at difference in runtimes for HMC and LMC
    times1 = []
    times2 = []
    for runs in range(20):
        testSysDict = util.generateRandDataDict(numImp=50, numOut=50, diagSens=0.90,
                                                diagSpec=0.99, numSamples=50 * 20,
                                                dataType='Tracked', transMatLambda=1.1,
                                                randSeed=-1, trueRates=[])
        testSysDict = util.GetVectorForms(testSysDict)
        print(np.count_nonzero(testSysDict['N']))
        priorMean, priorVar = -2.4, 1
        MCMCdict.update({'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4})
        testSysDict.update({'numPostSamples': numPostSamps, 'MCMCdict': MCMCdict,
                            'prior': methods.prior_normal(mu=priorMean, var=priorVar)})
        startTime = time.time()
        testSysDict = methods.GeneratePostSamples(testSysDict)
        endTime = time.time()
        print(endTime - startTime)
        times1.append(endTime - startTime)
        MCMCdict.update({'MCMCtype': 'Langevin'})
        testSysDict.update({'MCMCdict': MCMCdict})
        startTime = time.time()
        testSysDict = methods.GeneratePostSamples(testSysDict)
        endTime = time.time()
        print(endTime - startTime)
        times2.append(endTime - startTime)
    print(np.max(times1), np.min(times1), np.mean(times1))
    print(np.max(times2), np.min(times2), np.mean(times2))
    # Look at effect of more supply-chain traces
    baseN = [346, 318, 332, 331, 361, 348, 351, 321, 334, 341, 322, 328, 315, 307, 341, 333, 331, 344, 334, 323]
    print(np.mean(baseN) / (50 * 50))
    MCMCdict.update({'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4})
    times3 = []  # Less supply-chain traces
    lowerN = []
    for runs in range(20):
        testSysDict = util.generateRandDataDict(numImp=50, numOut=50, diagSens=0.90,
                                                diagSpec=0.99, numSamples=50 * 20,
                                                dataType='Tracked', transMatLambda=0.5,
                                                randSeed=-1, trueRates=[])
        testSysDict = util.GetVectorForms(testSysDict)
        lowerN.append(np.count_nonzero(testSysDict['N']))
        priorMean, priorVar = -2.4, 1
        testSysDict.update({'numPostSamples': numPostSamps, 'MCMCdict': MCMCdict,
                            'prior': methods.prior_normal(mu=priorMean, var=priorVar)})
        startTime = time.time()
        testSysDict = methods.GeneratePostSamples(testSysDict)
        endTime = time.time()
        print(endTime - startTime)
        times3.append(endTime - startTime)
    print(np.max(times3), np.min(times3), np.mean(times3))
    print(np.average(lowerN) / (50 * 50))
    times4 = []  # More supply-chain traces
    upperN = []
    for runs in range(20):
        testSysDict = util.generateRandDataDict(numImp=50, numOut=50, diagSens=0.90,
                                                diagSpec=0.99, numSamples=50 * 20,
                                                dataType='Tracked', transMatLambda=4.5,
                                                randSeed=-1, trueRates=[])
        testSysDict = util.GetVectorForms(testSysDict)
        upperN.append(np.count_nonzero(testSysDict['N']))
        priorMean, priorVar = -2.4, 1
        testSysDict.update({'numPostSamples': numPostSamps, 'MCMCdict': MCMCdict,
                            'prior': methods.prior_normal(mu=priorMean, var=priorVar)})
        startTime = time.time()
        testSysDict = methods.GeneratePostSamples(testSysDict)
        endTime = time.time()
        print(endTime - startTime)
        times4.append(endTime - startTime)
    print(np.max(times4), np.min(times4), np.mean(times4))
    print(np.average(upperN) / (50 * 50))
    # Look at effect of less or more nodes
    times5 = []  # Less nodes
    upperN = []
    for runs in range(20):
        testSysDict = util.generateRandDataDict(numImp=25, numOut=25, diagSens=0.90,
                                                diagSpec=0.99, numSamples=50 * 20,
                                                dataType='Tracked', transMatLambda=1.1,
                                                randSeed=-1, trueRates=[])
        testSysDict = util.GetVectorForms(testSysDict)
        upperN.append(np.count_nonzero(testSysDict['N']))
        priorMean, priorVar = -2.4, 1
        testSysDict.update({'numPostSamples': numPostSamps, 'MCMCdict': MCMCdict,
                            'prior': methods.prior_normal(mu=priorMean, var=priorVar)})
        startTime = time.time()
        testSysDict = methods.GeneratePostSamples(testSysDict)
        endTime = time.time()
        print(endTime - startTime)
        times5.append(endTime - startTime)
    print(np.max(times5), np.min(times5), np.mean(times5))
    print(np.average(upperN) / (25 * 25))
    times6 = []  # More nodes
    upperN = []
    for runs in range(20):
        testSysDict = util.generateRandDataDict(numImp=100, numOut=100, diagSens=0.90,
                                                diagSpec=0.99, numSamples=50 * 20,
                                                dataType='Tracked', transMatLambda=1.1,
                                                randSeed=-1, trueRates=[])
        testSysDict = util.GetVectorForms(testSysDict)
        upperN.append(np.count_nonzero(testSysDict['N']))
        priorMean, priorVar = -2.4, 1
        testSysDict.update({'numPostSamples': numPostSamps, 'MCMCdict': MCMCdict,
                            'prior': methods.prior_normal(mu=priorMean, var=priorVar)})
        startTime = time.time()
        testSysDict = methods.GeneratePostSamples(testSysDict)
        endTime = time.time()
        print(endTime - startTime)
        times6.append(endTime - startTime)
    print(np.max(times6), np.min(times6), np.mean(times6))
    print(np.average(upperN) / (100 * 100))
    return

# Running the functions produces results similar/analagous to those featured in the noted figures and tables.
_ = generateExampleInference() # Figure 2
_ = generateSyntheticData() # Figures 3 and 4, Tables 2 and 3
_ = timingAnalysis() # Table 1; MAY TAKE UPWARDS OF AN HOUR TO COMPLETE