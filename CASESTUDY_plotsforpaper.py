"""
Script that generates plots presented in the paper.
Inference generation requires use of the logistigate package, available at https://logistigate.readthedocs.io/en/main/.
"""

from logistigate.logistigate import utilities as util  # Pull from the submodule "develop" branch
from logistigate.logistigate import methods, lg
from logistigate.logistigate import lossfunctions as lf
from logistigate.logistigate import samplingplanfunctions as sampf
from logistigate.logistigate.priors import prior_normal_assort
import os
import numpy as np
from numpy.random import choice
import scipy.special as sps
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def showriskvalues():
    """Generate a figure showcasing how the risk changes with different parameter choices"""
    x = np.linspace(0.001, 0.999, 1000)
    t = 0.3  # Our target
    y1 = (x + 2 * (0.5 - t)) * (1 - x)
    tauvec = [0.05, 0.2, 0.4, 0.6, 0.95]
    fig, ax = plt.subplots(figsize=(8, 7))
    for tauind, tau in enumerate(tauvec):
        newy = [1 - x[i] * (tau - (1 - (t / x[i]) if x[i] < t else 0)) for i in range(len(x))]
        plt.plot(x, newy, dashes=[2,tauind], color='k')
    # plt.plot(x,y1)
    import matplotlib.ticker as mtick
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.tick_params(axis='both',labelsize=12)
    plt.title('Values for selected weight terms\n$l=30\%$', fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Weight value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.xlabel('SFP rate', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.text(0.84, 0.97, '$m=0.05$', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.text(0.84, 0.84, '$m=0.2$', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.text(0.84, 0.675, '$m=0.4$', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.text(0.84, 0.50, '$m=0.6$', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.text(0.84, 0.21, '$m=0.95$', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    fig.tight_layout()
    plt.show()
    plt.close()
    return


def showpriorselicitedfromrisk():
    """Produce chart depicting SFP rate priors, as in Section 5.2 of the paper"""
    # Extremely Low Risk, Very Low Risk, Low Risk, Moderate Risk, Moderately High Risk, High Risk, Very High Risk
    riskList = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25]
    riskNames = ['Extremely Low', 'Very Low', 'Low', 'Moderate', 'Moderately High', 'High', 'Very High']
    varConst = 2.
    xArr = sps.logit(np.arange(0.001, 1., 0.001))
    for riskInd, currRisk in enumerate(riskList):
        currPriorObj = prior_normal_assort(sps.logit(currRisk), np.array([varConst]).reshape((1, 1)))
        yArr = np.exp(np.array([currPriorObj.lpdf(xArr[i]) for i in range(xArr.shape[0])]))
        plt.plot(sps.expit(xArr), yArr, label=riskNames[riskInd], dashes=[2, riskInd], color='k')
    plt.xlabel('SFP Rate')
    plt.ylabel('Density')
    plt.legend(fancybox=True, title='Risk Level', fontsize='small')
    plt.title('Densities for Assessed SFP Rate Risks')
    plt.show()

    currRisk = 0.2
    currPriorObj = prior_normal_assort(sps.logit(currRisk), np.array([varConst]).reshape((1, 1)))
    yArr = np.exp(np.array([currPriorObj.lpdf(xArr[i]) for i in range(xArr.shape[0])]))
    yArrcmsm = np.cumsum(yArr) / np.sum(yArr)
    ind1 = next(x for x, val in enumerate(yArrcmsm) if val > 0.05) - 1
    ind2 = next(x for x, val in enumerate(yArrcmsm) if val > 0.95)
    sps.expit(xArr[ind1])
    sps.expit(xArr[ind2])

    return


def nVecs(length, target):
    """Return all possible positive integer vectors with size 'length', that sum to 'target'"""
    if length == 1:
        return [[target]]
    else:
        retSet = []
        for nexttarg in range(target+1):
            for nextset in nVecs(length-1,target-nexttarg):
                retSet.append([nexttarg]+nextset)

    return retSet


def example_planutility():
    """Produce two plots of the example of plan utility"""
    baseutil_arr = np.load(os.path.join('casestudyoutputs', '31MAY', 'util_avg_arr_example_base.npy'))
    adjutil_arr = np.load(os.path.join('casestudyoutputs', '31MAY', 'util_avg_arr_example_adj.npy'))
    testmax, testint = 60, 4

    def plot_marg_util(margutilarr, testmax, testint, al=0.6, titlestr='', type='cumulative', colors=[], dashes=[],
                       labels=[], utilmax=-1, linelabels=False):
        figtup = (8, 5)
        _ = plt.figure(figsize=figtup)
        if len(colors) == 0:
            colors = cm.rainbow(np.linspace(0, 1, margutilarr.shape[0]))
        if len(dashes) == 0:
            dashes = [[1, desind] for desind in range(margutilarr.shape[0])]
        if len(labels) == 0:
            labels = ['Design ' + str(desind + 1) for desind in range(margutilarr.shape[0])]
        if type == 'cumulative':
            x1 = range(0, testmax + 1, testint)
            if utilmax > 0.:
                yMax = utilmax
            else:
                yMax = margutilarr.max() * 1.1
            for desind in range(margutilarr.shape[0]):
                plt.plot(x1, margutilarr[desind], dashes=dashes[desind], linewidth=2.5, color=colors[desind],
                         label=labels[desind], alpha=al)
            if linelabels:
                for tnind in range(margutilarr.shape[0]):
                    plt.text(testmax * 1.01, margutilarr[tnind, -1], labels[tnind].ljust(15), fontsize=5)
        elif type == 'delta':
            x1 = range(testint, testmax + 1, testint)
            deltaArr = np.zeros((margutilarr.shape[0], margutilarr.shape[1] - 1))
            for rw in range(deltaArr.shape[0]):
                for col in range(deltaArr.shape[1]):
                    deltaArr[rw, col] = margutilarr[rw, col + 1] - margutilarr[rw, col]
            if utilmax > 0.:
                yMax = utilmax
            else:
                yMax = deltaArr.max() * 1.1
            for desind in range(deltaArr.shape[0]):
                plt.plot(x1, deltaArr[desind], dashes=dashes[desind], linewidth=2.5, color=colors[desind],
                         label=labels[desind], alpha=al)
            if linelabels:
                for tnind in range(deltaArr.shape[0]):
                    plt.text(testmax * 1.01, deltaArr[tnind, -1], labels[tnind].ljust(15), fontsize=5)
        plt.legend()
        plt.ylim([0., yMax])
        plt.xlabel('Number of tests, $N$')
        if type == 'delta':
            plt.ylabel('Marginal Utility Gain')
            plt.title('Marginal Utility vs. Increasing Tests\n' + titlestr)
        else:
            plt.ylabel('Utility, $U$')
            plt.title('Utility vs. tests\n' + titlestr)
        plt.show()
        plt.close()
        return
    '''
    plot_marg_util(baseutil_arr, testmax=testmax, testint=testint,
                        colors=['blue', 'red', 'green'], titlestr='$v=1$',
                        labels=['Focused', 'Uniform', 'Adapted'])
    plot_marg_util(adjutil_arr, testmax=testmax, testint=testint,
                        colors=['blue', 'red', 'green'], titlestr='$v=10$',
                        labels=['Focused', 'Uniform', 'Adapted'])
    '''
    # Per comment, cut off at 40 tests instead in order to capture more interesting elements
    testmax=40
    plot_marg_util(baseutil_arr[:, :-5], testmax=testmax, testint=testint,
                        colors=['blue', 'red', 'green'], titlestr='Underestimation equal to overestimation $(v=1)$',
                        labels=['Least Tested', 'Uniform', 'Highest SFPs'])
    plot_marg_util(adjutil_arr[:, :-5], testmax=testmax, testint=testint,
                        colors=['blue', 'red', 'green'], titlestr='Underestimation more significant than overestimation $(v=10)$',
                        labels=['Least Tested', 'Uniform', 'Highest SFPs'])
    return


def casestudyplots_familiar():
    """
    Cleaned up plots for use in case study in paper
    """
    testmax, testint = 400, 10
    TNnames = ['Moderate(39)', 'Moderate(17)', 'ModeratelyHigh(95)', 'ModeratelyHigh(26)']
    numTN = len(TNnames)

    # Size of figure layout for all figures
    figtup = (7, 5)
    titleSz, axSz, labelSz = 12, 10, 9
    xMax = 450

    '''
    #######################
    # Plot of marginal utilities
    colors = cm.rainbow(np.linspace(0, 0.5, numTN))
    labels = [TNnames[ind] for ind in range(numTN)]

    x = range(testint, testmax + 1, testint)
    deltaArr = np.zeros((heur_util.shape[0], heur_util.shape[1] - 1))
    for rw in range(deltaArr.shape[0]):
        for col in range(deltaArr.shape[1]):
            deltaArr[rw, col] = heur_util[rw, col + 1] - heur_util[rw, col]
    yMax = np.max(deltaArr) * 1.1

    _ = plt.figure(figsize=figtup)
    for tnind in range(numTN):
        plt.plot(x, deltaArr[tnind], linewidth=2, color=colors[tnind],
                 label=labels[tnind], alpha=0.6)
    for tnind in range(numTN):
        plt.text(testint * 1.1, deltaArr[tnind, 0], labels[tnind].ljust(15), fontsize=labelSz - 1)
    plt.legend(fontsize=labelSz)
    plt.ylim([0., yMax])
    plt.xlim([0., xMax])
    plt.xlabel('Number of Tests', fontsize=axSz)
    plt.ylabel('Marginal Utility Gain', fontsize=axSz)
    plt.title('Marginal Utility with Increasing Tests\nFamiliar Setting', fontsize=titleSz)
    plt.show()
    plt.close()
    #######################
    '''

    #######################
    # Allocation plot
    allocArr = np.load(os.path.join('casestudyoutputs', 'familiar', 'fam_alloc.npy'))
    colorsset = plt.get_cmap('Set1')
    colorinds = [6, 1, 2, 3]
    colors = np.array([colorsset(i) for i in colorinds])
    labels = [TNnames[ind] for ind in range(numTN)]
    x = range(0, testmax + 1, testint)
    _ = plt.figure(figsize=figtup)
    for tnind in range(allocArr.shape[0]):
        plt.plot(x, allocArr[tnind] * testint, linewidth=3, color=colors[tnind],
                 label=labels[tnind], alpha=0.6)
    # allocMax = allocArr.max() * testInt * 1.1
    allocMax = 185
    for tnind in range(numTN):
        plt.text(testmax * 1.01, allocArr[tnind, -1] * testint, labels[tnind].ljust(15), fontsize=labelSz - 1)
    plt.legend(fontsize=labelSz)
    plt.ylim([0., allocMax])
    plt.xlim([0., xMax])
    plt.xlabel('Sampling Budget', fontsize=axSz)
    plt.ylabel('Test Node Allocation', fontsize=axSz)
    plt.title('Sampling Plan vs. Budget\nExisting Setting', fontsize=titleSz)
    # plt.tight_layout()
    plt.show()
    plt.close()
    #######################

    #######################
    # Policy utility comparison
    util_arr = np.load(os.path.join('casestudyoutputs', 'familiar', 'util_avg_arr_fam.npy'))
    util_arr_hi = np.load(os.path.join('casestudyoutputs', 'familiar', 'util_hi_arr_fam.npy'))
    util_arr_lo = np.load(os.path.join('casestudyoutputs', 'familiar', 'util_lo_arr_fam.npy'))
    heur_util = np.load(os.path.join('casestudyoutputs', 'familiar', 'fam_util_avg.npy'))
    heur_util_hi = np.load(os.path.join('casestudyoutputs', 'familiar', 'fam_util_hi.npy'))
    heur_util_lo = np.load(os.path.join('casestudyoutputs', 'familiar', 'fam_util_lo.npy'))
    util_arr = np.vstack((heur_util,util_arr))
    util_arr_hi = np.vstack((heur_util_hi, util_arr_hi))
    util_arr_lo = np.vstack((heur_util_lo, util_arr_lo))
    # Utility comparison plot
    colorsset = plt.get_cmap('Accent')
    colorinds = [0, 1, 2]
    colors = np.array([colorsset(i) for i in colorinds])
    #colors = cm.rainbow(np.linspace(0, 0.8, 3))
    labels = ['Utility-Informed', 'Uniform', 'Fixed']
    x = range(0, testmax + 1, testint)
    utilMax = -1
    for lst in util_arr:
        currMax = np.amax(np.array(lst))
        if currMax > utilMax:
            utilMax = currMax
    utilMax = utilMax * 1.1

    _ = plt.figure(figsize=figtup)
    for groupind in range(3):
        plt.plot(x, util_arr[groupind], color=colors[groupind], linewidth=0.7, alpha=1.,
                 label=labels[groupind] + ' 95% CI')
        plt.fill_between(x, util_arr_hi[groupind], util_arr_lo[groupind], color=colors[groupind], alpha=0.2)
        # Line label
        plt.text(x[-1] * 1.01, util_arr[groupind][-1], labels[groupind].ljust(15), fontsize=labelSz - 1)
    plt.ylim(0, utilMax)
    # plt.xlim(0,x[-1]*1.12)
    plt.xlim([0., xMax])
    leg = plt.legend(loc='upper left', fontsize=labelSz)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.0)
    plt.xlabel('Sampling Budget', fontsize=axSz)
    plt.ylabel('Plan Utility', fontsize=axSz)
    plt.title('Utility from Utility-Informed, Uniform and Fixed Allocations\nExisting Setting', fontsize=titleSz)
    # Add text for budgetary savings vs other policies at 100 tests
    x1, x2, x3 = 100, 132, 156
    iv = 0.015
    utilind = int(x1/testint)
    plt.plot([x1, x3], [util_arr[0][utilind], util_arr[0][utilind]], color='black', linestyle='--')
    plt.plot([100, 100], [util_arr[0][utilind] - iv, util_arr[0][utilind] + iv], color='black', linestyle='--')
    plt.plot([x2, x2], [util_arr[0][utilind] - iv, util_arr[0][utilind] + iv], color='black', linestyle='--')
    plt.plot([x3, x3], [util_arr[0][utilind] - iv, util_arr[0][utilind] + iv], color='black', linestyle='--')
    plt.text(110, util_arr[0][utilind] + iv / 2, str(int(x2-x1)), fontsize=labelSz)
    plt.text(137, util_arr[0][utilind] + iv / 2, str(int(x3-x2)), fontsize=labelSz)
    # plt.tight_layout()
    plt.show()
    plt.close()
    #######################

    return


def casestudyplots_familiar_market():
    """
    Cleaned up plots for use in case study in paper
    """
    testmax, testint = 400, 10
    TNnames = ['Moderate(39)', 'Moderate(17)', 'ModeratelyHigh(95)', 'ModeratelyHigh(26)']
    numTN = len(TNnames)

    # Size of figure layout for all figures
    figtup = (7, 5)
    titleSz, axSz, labelSz = 12, 10, 9
    xMax = 450

    '''
    #######################
    # Plot of marginal utilities
    colors = cm.rainbow(np.linspace(0, 0.5, numTN))
    labels = [TNnames[ind] for ind in range(numTN)]

    x = range(testint, testmax + 1, testint)
    deltaArr = np.zeros((heur_util.shape[0], heur_util.shape[1] - 1))
    for rw in range(deltaArr.shape[0]):
        for col in range(deltaArr.shape[1]):
            deltaArr[rw, col] = heur_util[rw, col + 1] - heur_util[rw, col]
    yMax = np.max(deltaArr) * 1.1

    _ = plt.figure(figsize=figtup)
    for tnind in range(numTN):
        plt.plot(x, deltaArr[tnind], linewidth=2, color=colors[tnind],
                 label=labels[tnind], alpha=0.6)
    for tnind in range(numTN):
        plt.text(testint * 1.1, deltaArr[tnind, 0], labels[tnind].ljust(15), fontsize=labelSz - 1)
    plt.legend(fontsize=labelSz)
    plt.ylim([0., yMax])
    plt.xlim([0., xMax])
    plt.xlabel('Number of Tests', fontsize=axSz)
    plt.ylabel('Marginal Utility Gain', fontsize=axSz)
    plt.title('Marginal Utility with Increasing Tests\nFamiliar Setting', fontsize=titleSz)
    plt.show()
    plt.close()
    #######################
    '''

    #######################
    # Allocation plot
    allocArr = np.load(os.path.join('casestudyoutputs', 'familiar', 'fam_market_alloc.npy'))
    colorsset = plt.get_cmap('Set1')
    colorinds = [6, 1, 2, 3]
    colors = np.array([colorsset(i) for i in colorinds])
    labels = [TNnames[ind] for ind in range(numTN)]
    x = range(0, testmax + 1, testint)
    _ = plt.figure(figsize=figtup)
    for tnind in range(allocArr.shape[0]):
        plt.plot(x, allocArr[tnind] * testint, linewidth=3, color=colors[tnind],
                 label=labels[tnind], alpha=0.6)
    # allocMax = allocArr.max() * testInt * 1.1
    allocMax = 185
    for tnind in range(numTN):
        plt.text(testmax * 1.01, allocArr[tnind, -1] * testint, labels[tnind].ljust(15), fontsize=labelSz - 1)
    plt.legend(fontsize=labelSz)
    plt.ylim([0., allocMax])
    plt.xlim([0., xMax])
    plt.xlabel('Sampling Budget', fontsize=axSz)
    plt.ylabel('Test Node Allocation', fontsize=axSz)
    plt.title('Sampling Plan vs. Budget\nExisting Setting with Prioritization', fontsize=titleSz)
    # plt.tight_layout()
    plt.show()
    plt.close()
    #######################

    #######################
    # Policy utility comparison
    util_arr = np.load(os.path.join('casestudyoutputs', 'familiar', 'util_avg_arr_fam_market.npy'))
    util_arr_hi = np.load(os.path.join('casestudyoutputs', 'familiar', 'util_hi_arr_fam_market.npy'))
    util_arr_lo = np.load(os.path.join('casestudyoutputs', 'familiar', 'util_lo_arr_fam_market.npy'))
    heur_util = np.load(os.path.join('casestudyoutputs', 'familiar', 'fam_market_util_avg.npy'))
    heur_util_hi = np.load(os.path.join('casestudyoutputs', 'familiar', 'fam_market_util_hi.npy'))
    heur_util_lo = np.load(os.path.join('casestudyoutputs', 'familiar', 'fam_market_util_lo.npy'))
    util_arr = np.vstack((heur_util, util_arr))
    util_arr_hi = np.vstack((heur_util_hi, util_arr_hi))
    util_arr_lo = np.vstack((heur_util_lo, util_arr_lo))
    # Utility comparison plot
    colorsset = plt.get_cmap('Accent')
    colorinds = [0, 1, 2]
    colors = np.array([colorsset(i) for i in colorinds])
    # colors = cm.rainbow(np.linspace(0, 0.8, 3))
    labels = ['Utility-Informed', 'Uniform', 'Fixed']
    x = range(0, testmax + 1, testint)
    utilMax = -1
    for lst in util_arr:
        currMax = np.amax(np.array(lst))
        if currMax > utilMax:
            utilMax = currMax
    utilMax = utilMax * 1.1

    _ = plt.figure(figsize=figtup)
    for groupind in range(3):
        plt.plot(x, util_arr[groupind], color=colors[groupind], linewidth=0.7, alpha=1.,
                 label=labels[groupind] + ' 95% CI')
        plt.fill_between(x, util_arr_hi[groupind], util_arr_lo[groupind], color=colors[groupind], alpha=0.2)
        # Line label
        plt.text(x[-1] * 1.01, util_arr[groupind][-1], labels[groupind].ljust(15), fontsize=labelSz - 1)
    plt.ylim(0, utilMax)
    # plt.xlim(0,x[-1]*1.12)
    plt.xlim([0., xMax])
    leg = plt.legend(loc='upper left', fontsize=labelSz)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.0)
    plt.xlabel('Sampling Budget', fontsize=axSz)
    plt.ylabel('Plan Utility', fontsize=axSz)
    plt.title('Utility from Utility-Informed, Uniform and Fixed Allocations\nExisting Setting with Prioritization', fontsize=titleSz)
    # Add text for budgetary savings vs other policies at 100 tests
    x1, x2, x3 = 100, 120, 146
    iv = 0.0015
    utilind = int(x1 / testint)
    plt.plot([x1, x3], [util_arr[0][utilind], util_arr[0][utilind]], color='black', linestyle='--')
    plt.plot([100, 100], [util_arr[0][utilind] - iv, util_arr[0][utilind] + iv], color='black', linestyle='--')
    plt.plot([x2, x2], [util_arr[0][utilind] - iv, util_arr[0][utilind] + iv], color='black', linestyle='--')
    plt.plot([x3, x3], [util_arr[0][utilind] - iv, util_arr[0][utilind] + iv], color='black', linestyle='--')
    plt.text(103, util_arr[0][utilind] + iv / 2, str(int(x2 - x1)), fontsize=labelSz)
    plt.text(127.5, util_arr[0][utilind] + iv / 2, str(int(x3 - x2)), fontsize=labelSz)
    # plt.tight_layout()
    plt.show()
    plt.close()
    #######################
    return


def casestudyplots_exploratory():
    """
    Cleaned up plots for use in case study in paper
    """
    testmax, testint = 400, 10
    TNnames = ['Moderate (39)', 'Moderate (17)', 'Moderately High (95)', 'Moderately High (26)',
              'Moderately High (New #1)', 'Moderate (New #1)', 'Moderately High (New #2)', 'Moderate (New #2)']
    numTN = len(TNnames)

    # Size of figure layout for all figures
    figtup = (7.5, 5)
    titleSz, axSz, labelSz = 15, 11, 11
    xMax = 450

    '''
    #######################
    # Plot of marginal utilities
    colors = cm.rainbow(np.linspace(0, 0.5, numTN))
    labels = [TNnames[ind] for ind in range(numTN)]

    x = range(testint, testmax + 1, testint)
    deltaArr = np.zeros((heur_util.shape[0], heur_util.shape[1] - 1))
    for rw in range(deltaArr.shape[0]):
        for col in range(deltaArr.shape[1]):
            deltaArr[rw, col] = heur_util[rw, col + 1] - heur_util[rw, col]
    yMax = np.max(deltaArr) * 1.1

    _ = plt.figure(figsize=figtup)
    for tnind in range(numTN):
        plt.plot(x, deltaArr[tnind], linewidth=2, color=colors[tnind],
                 label=labels[tnind], alpha=0.6)
    for tnind in range(numTN):
        plt.text(testint * 1.1, deltaArr[tnind, 0], labels[tnind].ljust(15), fontsize=labelSz - 1)
    plt.legend(fontsize=labelSz)
    plt.ylim([0., yMax])
    plt.xlim([0., xMax])
    plt.xlabel('Number of Tests', fontsize=axSz)
    plt.ylabel('Marginal Utility Gain', fontsize=axSz)
    plt.title('Marginal Utility with Increasing Tests\nFamiliar Setting', fontsize=titleSz)
    plt.show()
    plt.close()
    #######################
    '''

    #######################
    # Allocation plot
    allocArr = np.load(os.path.join('utilitypaper', 'allprovinces', 'allprov_alloc.npy'))
    colorsset = plt.get_cmap('Set1')
    colorsset2 = plt.get_cmap('Dark2')
    colorinds = [6, 1, 2, 3, 4, 0, 5, 7]
    colors = np.array([colorsset(i) for i in colorinds])
    colors[6] = colorsset2(5)
    labels = [TNnames[ind] for ind in range(numTN)]
    x = range(0, testmax + 1, testint)
    _ = plt.figure(figsize=figtup)
    for tnind in range(allocArr.shape[0]):
        if tnind < 4:
            plt.plot(x, allocArr[tnind] * testint, linewidth=3, color=colors[tnind],
                 label=labels[tnind], alpha=0.6)
        else:
            plt.plot(x, allocArr[tnind] * testint, linewidth=3, color=colors[tnind],
                     label=labels[tnind], dashes=[0.5,0.5], alpha=0.6)
    # allocMax = allocArr.max() * testInt * 1.1
    allocMax = 120
    for tnind in range(numTN):
        delt = 2
        if tnind==0:
            plt.text(testmax * 0.945, allocArr[tnind, -1] * testint + delt, labels[tnind].ljust(15), fontsize=labelSz - 1)
        elif tnind==4:
            plt.text(testmax * 0.9, allocArr[tnind, -1] * testint + delt, labels[tnind].ljust(15), fontsize=labelSz - 1)
        elif tnind==6:
            plt.text(testmax * 0.9, allocArr[tnind, -1] * testint + delt, labels[tnind].ljust(15), fontsize=labelSz - 1)
        elif tnind==3:
            plt.text(testmax * 0.945, allocArr[tnind, -1] * testint - 3*delt, labels[tnind].ljust(15), fontsize=labelSz - 1)
        #elif tnind==3:
        #    plt.text(testmax * 1.01, allocArr[tnind, -1] * testint + 0.2*delt, labels[tnind].ljust(15), fontsize=labelSz - 1)
        else:
            plt.text(testmax * 0.945, allocArr[tnind, -1] * testint + delt, labels[tnind].ljust(15), fontsize=labelSz - 1)

    plt.legend(fontsize=labelSz, loc='upper left')
    plt.ylim([0., allocMax])
    plt.xlim([0., xMax])
    plt.xlabel('Sampling Budget', fontsize=axSz)
    plt.ylabel('Test Node Allocation', fontsize=axSz)
    plt.title('Sampling Plan vs. Budget: All-Provinces Setting', fontsize=titleSz)
    # plt.tight_layout()
    plt.show()
    plt.close()
    #######################

    #######################
    # Policy utility comparison
    util_avg_greedy = np.load(os.path.join('utilitypaper', 'allprovinces', 'util_avg_greedy.npy'))[0]
    util_hi_greedy = np.load(os.path.join('utilitypaper', 'allprovinces', 'util_hi_greedy.npy'))[0]
    util_lo_greedy = np.load(os.path.join('utilitypaper', 'allprovinces', 'util_lo_greedy.npy'))[0]
    util_avg_unif = np.load(os.path.join('utilitypaper', 'allprovinces', 'util_avg_unif.npy'))[0]
    util_hi_unif = np.load(os.path.join('utilitypaper', 'allprovinces', 'util_hi_unif.npy'))[0]
    util_lo_unif = np.load(os.path.join('utilitypaper', 'allprovinces', 'util_lo_unif.npy'))[0]
    util_avg_rudi = np.load(os.path.join('utilitypaper', 'allprovinces', 'util_avg_rudi.npy'))[0]
    util_hi_rudi = np.load(os.path.join('utilitypaper', 'allprovinces', 'util_hi_rudi.npy'))[0]
    util_lo_rudi = np.load(os.path.join('utilitypaper', 'allprovinces', 'util_lo_rudi.npy'))[0]

    util_arr = np.load(os.path.join('utilitypaper', 'exploratory', 'util_avg_arr_expl.npy'))
    util_arr_hi = np.load(os.path.join('utilitypaper', 'exploratory', 'util_hi_arr_expl.npy'))
    util_arr_lo = np.load(os.path.join('utilitypaper', 'exploratory', 'util_lo_arr_expl.npy'))
    heur_util = np.load(os.path.join('utilitypaper', 'exploratory', 'expl_util_avg.npy'))
    heur_util_hi = np.load(os.path.join('utilitypaper', 'exploratory', 'expl_util_hi.npy'))
    heur_util_lo = np.load(os.path.join('utilitypaper', 'exploratory', 'expl_util_lo.npy'))
    util_arr = np.vstack((util_avg_greedy, util_avg_unif, util_avg_rudi))
    util_arr_hi = np.vstack((util_hi_greedy, util_hi_unif, util_hi_rudi))
    util_arr_lo = np.vstack((util_lo_greedy, util_lo_unif, util_lo_rudi))
    # Utility comparison plot
    colorsset = plt.get_cmap('Accent')
    colorinds = [0, 1, 2]
    colors = np.array([colorsset(i) for i in colorinds])
    #colors = cm.rainbow(np.linspace(0, 0.8, 3))
    labels = ['Utility-Informed', 'Uniform', 'Fixed']
    x = range(0, testmax + 1, testint)
    utilMax = -1
    for lst in util_arr:
        currMax = np.amax(np.array(lst))
        if currMax > utilMax:
            utilMax = currMax
    utilMax = utilMax * 1.1

    _ = plt.figure(figsize=figtup)
    for groupind in range(3):
        plt.plot(x, util_arr[groupind], color=colors[groupind], linewidth=0.7, alpha=1.,
                 label=labels[groupind] + ' 95% CI')
        plt.fill_between(x, util_arr_hi[groupind], util_arr_lo[groupind], color=colors[groupind], alpha=0.2)
        # Line label
        plt.text(x[-1] * 1.01, util_arr[groupind][-1], labels[groupind].ljust(15), fontsize=labelSz - 1)
    plt.ylim(0, utilMax)
    # plt.xlim(0,x[-1]*1.12)
    plt.xlim([0., xMax])
    leg = plt.legend(loc='upper left', fontsize=labelSz)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.0)
    plt.xlabel('Sampling Budget', fontsize=axSz)
    plt.ylabel('Plan Utility', fontsize=axSz)
    plt.title('Utility from Utility-Informed, Uniform and Fixed Allocations\nAll-Provinces Setting', fontsize=titleSz)
    # Add text for budgetary savings vs other policies at 100 tests
    x1, x2, x3 = 100, 120, 331
    iv = 0.03
    utilind = int(x1/testint)
    plt.plot([x1, x3], [util_arr[0][utilind], util_arr[0][utilind]], color='black', linestyle='--')
    plt.plot([100, 100], [util_arr[0][utilind] - iv, util_arr[0][utilind] + iv], color='black', linestyle='--')
    plt.plot([x2, x2], [util_arr[0][utilind] - iv, util_arr[0][utilind] + iv], color='black', linestyle='--')
    plt.plot([x3, x3], [util_arr[0][utilind] - iv, util_arr[0][utilind] + iv], color='black', linestyle='--')
    plt.text(103.5, util_arr[0][utilind] + iv / 2, str(int(x2-x1)), fontsize=labelSz)
    plt.text(190, util_arr[0][utilind] + iv / 2, str(int(x3-x2)), fontsize=labelSz)
    # plt.tight_layout()
    plt.show()
    plt.close()
    #######################

    return


def casestudyplots_exploratory_market():
    """
    Cleaned up plots for use in case study in paper
    """
    testmax, testint = 400, 10
    TNnames = ['Moderate(39)', 'Moderate(17)', 'ModeratelyHigh(95)', 'ModeratelyHigh(26)',
               'ModeratelyHighUnex(1)', 'ModerateUnex(1)', 'ModeratelyHighUnex(2)',
               'ModerateUnex(2)']
    numTN = len(TNnames)

    # Size of figure layout for all figures
    figtup = (7.5, 5)
    titleSz, axSz, labelSz = 12, 10, 9
    xMax = 450

    '''
    #######################
    # Plot of marginal utilities
    colors = cm.rainbow(np.linspace(0, 0.5, numTN))
    labels = [TNnames[ind] for ind in range(numTN)]

    x = range(testint, testmax + 1, testint)
    deltaArr = np.zeros((heur_util.shape[0], heur_util.shape[1] - 1))
    for rw in range(deltaArr.shape[0]):
        for col in range(deltaArr.shape[1]):
            deltaArr[rw, col] = heur_util[rw, col + 1] - heur_util[rw, col]
    yMax = np.max(deltaArr) * 1.1

    _ = plt.figure(figsize=figtup)
    for tnind in range(numTN):
        plt.plot(x, deltaArr[tnind], linewidth=2, color=colors[tnind],
                 label=labels[tnind], alpha=0.6)
    for tnind in range(numTN):
        plt.text(testint * 1.1, deltaArr[tnind, 0], labels[tnind].ljust(15), fontsize=labelSz - 1)
    plt.legend(fontsize=labelSz)
    plt.ylim([0., yMax])
    plt.xlim([0., xMax])
    plt.xlabel('Number of Tests', fontsize=axSz)
    plt.ylabel('Marginal Utility Gain', fontsize=axSz)
    plt.title('Marginal Utility with Increasing Tests\nFamiliar Setting', fontsize=titleSz)
    plt.show()
    plt.close()
    #######################
    '''

    #######################
    # Allocation plot
    allocArr = np.load(os.path.join('casestudyoutputs', 'exploratory', 'expl_market_alloc.npy'))
    colorsset = plt.get_cmap('Set1')
    colorsset2 = plt.get_cmap('Dark2')
    colorinds = [6, 1, 2, 3, 4, 0, 5, 7]
    colors = np.array([colorsset(i) for i in colorinds])
    colors[6] = colorsset2(5)
    labels = [TNnames[ind] for ind in range(numTN)]
    x = range(0, testmax + 1, testint)
    _ = plt.figure(figsize=figtup)
    for tnind in range(allocArr.shape[0]):
        if tnind < 4:
            plt.plot(x, allocArr[tnind] * testint, linewidth=3, color=colors[tnind],
                     label=labels[tnind], alpha=0.6)
        else:
            plt.plot(x, allocArr[tnind] * testint, linewidth=3, color=colors[tnind],
                     label=labels[tnind], dashes=[0.5, 0.5], alpha=0.6)
    # allocMax = allocArr.max() * testInt * 1.1
    allocMax = 120
    for tnind in range(numTN):
        delt = 4
        if tnind == 4:
            plt.text(testmax * 1.01, allocArr[tnind, -1] * testint + 0.5*delt, labels[tnind].ljust(15),
                     fontsize=labelSz - 1)
        elif tnind == 2:
            plt.text(testmax * 1.01, allocArr[tnind, -1] * testint + 1.1*delt, labels[tnind].ljust(15),
                     fontsize=labelSz - 1)
        elif tnind == 5:
            plt.text(testmax * 1.01, allocArr[tnind, -1] * testint - 0.5*delt, labels[tnind].ljust(15),
                     fontsize=labelSz - 1)
        elif tnind == 3:
            plt.text(testmax * 1.01, allocArr[tnind, -1] * testint + 0.2 * delt, labels[tnind].ljust(15),
                     fontsize=labelSz - 1)
        elif tnind == 0:
            plt.text(testmax * 1.01, allocArr[tnind, -1] * testint + 0.5 * delt, labels[tnind].ljust(15),
                     fontsize=labelSz - 1)
        elif tnind == 7:
            plt.text(testmax * 1.01, allocArr[tnind, -1] * testint - 0.5 * delt, labels[tnind].ljust(15),
                     fontsize=labelSz - 1)
        else:
            plt.text(testmax * 1.01, allocArr[tnind, -1] * testint, labels[tnind].ljust(15), fontsize=labelSz - 1)

    plt.legend(fontsize=labelSz, loc='upper left')
    plt.ylim([0., allocMax])
    plt.xlim([0., xMax])
    plt.xlabel('Sampling Budget', fontsize=axSz)
    plt.ylabel('Test Node Allocation', fontsize=axSz)
    plt.title('Sampling Plan vs. Budget\nAll-Provinces Setting with Prioritization', fontsize=titleSz)
    # plt.tight_layout()
    plt.show()
    plt.close()
    #######################

    #######################
    # Policy utility comparison
    util_arr = np.load(os.path.join('casestudyoutputs', 'exploratory', 'util_avg_arr_expl_market.npy'))
    util_arr_hi = np.load(os.path.join('casestudyoutputs', 'exploratory', 'util_hi_arr_expl_market.npy'))
    util_arr_lo = np.load(os.path.join('casestudyoutputs', 'exploratory', 'util_lo_arr_expl_market.npy'))
    heur_util = np.load(os.path.join('casestudyoutputs', 'exploratory', 'expl_market_util_avg.npy'))
    heur_util_hi = np.load(os.path.join('casestudyoutputs', 'exploratory', 'expl_market_util_hi.npy'))
    heur_util_lo = np.load(os.path.join('casestudyoutputs', 'exploratory', 'expl_market_util_lo.npy'))
    util_arr = np.vstack((heur_util,util_arr))
    util_arr_hi = np.vstack((heur_util_hi, util_arr_hi))
    util_arr_lo = np.vstack((heur_util_lo, util_arr_lo))
    # Utility comparison plot
    colorsset = plt.get_cmap('Accent')
    colorinds = [0, 1, 2]
    colors = np.array([colorsset(i) for i in colorinds])
    #colors = cm.rainbow(np.linspace(0, 0.8, 3))
    labels = ['Utility-Informed', 'Uniform', 'Fixed']
    x = range(0, testmax + 1, testint)
    utilMax = -1
    for lst in util_arr:
        currMax = np.amax(np.array(lst))
        if currMax > utilMax:
            utilMax = currMax
    utilMax = utilMax * 1.1

    _ = plt.figure(figsize=figtup)
    for groupind in range(3):
        plt.plot(x, util_arr[groupind], color=colors[groupind], linewidth=0.7, alpha=1.,
                 label=labels[groupind] + ' 95% CI')
        plt.fill_between(x, util_arr_hi[groupind], util_arr_lo[groupind], color=colors[groupind], alpha=0.2)
        # Line label
        plt.text(x[-1] * 1.01, util_arr[groupind][-1], labels[groupind].ljust(15), fontsize=labelSz - 1)
    plt.ylim(0, utilMax)
    # plt.xlim(0,x[-1]*1.12)
    plt.xlim([0., xMax])
    leg = plt.legend(loc='upper left', fontsize=labelSz)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.0)
    plt.xlabel('Sampling Budget', fontsize=axSz)
    plt.ylabel('Plan Utility', fontsize=axSz)
    plt.title('Utility from Utility-Informed, Uniform and Fixed Allocations\nAll-Provinces Setting with Prioritization', fontsize=titleSz)
    # Add text for budgetary savings vs other policies at 100 tests
    x1, x2 = 100, 133
    iv = 0.0032
    utilind = int(x1/testint)
    plt.plot([x1, x2], [util_arr[0][utilind], util_arr[0][utilind]], color='black', linestyle='--')
    plt.plot([100, 100], [util_arr[0][utilind] - iv, util_arr[0][utilind] + iv], color='black', linestyle='--')
    plt.plot([x2, x2], [util_arr[0][utilind] - iv, util_arr[0][utilind] + iv], color='black', linestyle='--')
    #plt.plot([x3, x3], [util_arr[0][utilind] - iv, util_arr[0][utilind] + iv], color='black', linestyle='--')
    plt.text(106, util_arr[0][utilind] + iv / 2, str(int(x2-x1)), fontsize=labelSz)
    #plt.text(190, util_arr[0][utilind] + iv / 2, str(int(x3-x2)), fontsize=labelSz)
    # plt.tight_layout()
    plt.show()
    plt.close()
    #######################

    return


def casestudyplots_exploratory_OLD():
    """
    Cleaned up plots for use in case study in paper
    """
    testMax, testInt = 400, 10
    tnNames = ['MOD_39', 'MOD_17', 'MODHIGH_95', 'MODHIGH_26',
               'MODHIGH_EXPL_1', 'MOD_EXPL_1', 'MODHIGH_EXPL_2', 'MOD_EXPL_2']
    numTN = len(tnNames)

    unif_utillist = [np.array([0.08841012, 0.15565143, 0.24106519, 0.29720411, 0.34670028,
                               0.40441693, 0.4520183, 0.49299778, 0.53293407, 0.56663843,
                               0.6068871, 0.64709646, 0.69159897, 0.72380886, 0.76514307,
                               0.79660616, 0.83239866, 0.86195176, 0.89731708, 0.94318546,
                               0.96943093, 1.00218572, 1.03152891, 1.06727324, 1.10070594,
                               1.11977055, 1.15212293, 1.1835634, 1.22617661, 1.24228975,
                               1.25780729, 1.2959397, 1.30176525, 1.34451775, 1.35307222,
                               1.38860051, 1.40130638, 1.41632717, 1.46246785, 1.46821944]),
                     np.array([0.10204536, 0.17203302, 0.26286992, 0.32361954, 0.37307678,
                               0.43085786, 0.47399878, 0.51503033, 0.56066216, 0.58983501,
                               0.63725473, 0.68409818, 0.72126763, 0.7579929, 0.79671422,
                               0.82811037, 0.87382789, 0.90458568, 0.93059176, 0.97441651,
                               1.00657399, 1.04076035, 1.06359108, 1.10439937, 1.13673226,
                               1.16499527, 1.19547982, 1.22635456, 1.24430843, 1.27061623,
                               1.30162455, 1.3234999, 1.34651722, 1.37148126, 1.38155695,
                               1.41888158, 1.42802216, 1.46714458, 1.47241259, 1.49288413]),
                     np.array([0.09591184, 0.16438589, 0.2464366, 0.29357483, 0.34867104,
                               0.40393655, 0.44666241, 0.48435072, 0.51989594, 0.56288725,
                               0.60492358, 0.64595947, 0.68866638, 0.720943, 0.7516753,
                               0.78981393, 0.8418654, 0.86775926, 0.90246394, 0.92999427,
                               0.97223517, 0.99205388, 1.02163245, 1.05835929, 1.0956864,
                               1.12370356, 1.14435483, 1.16793021, 1.20470649, 1.2345608,
                               1.25611448, 1.29008477, 1.309387, 1.31975911, 1.36674577,
                               1.37997659, 1.41873672, 1.41450192, 1.43315206, 1.47705606]),
                     np.array([0.0944134, 0.15753564, 0.2466545, 0.29581787, 0.34538147,
                               0.40123685, 0.43931291, 0.48930003, 0.52918134, 0.56084947,
                               0.60789135, 0.64990563, 0.69058907, 0.72247047, 0.74809329,
                               0.7948111, 0.83155204, 0.86321651, 0.90468742, 0.94411321,
                               0.96898088, 1.0108199, 1.04053446, 1.06833746, 1.09215482,
                               1.12378437, 1.1659193, 1.18378993, 1.21288696, 1.24283401,
                               1.27446602, 1.28378132, 1.32848316, 1.32849597, 1.36228587,
                               1.37765224, 1.4101449, 1.434839, 1.46371489, 1.49483553]),
                     np.array([0.08167338, 0.14426625, 0.23216857, 0.28673666, 0.34123239,
                               0.40118761, 0.4392709, 0.48147611, 0.51745304, 0.55110435,
                               0.59909138, 0.63991174, 0.67302489, 0.7198011, 0.75057478,
                               0.78746757, 0.81611783, 0.84495377, 0.90048518, 0.91688154,
                               0.95922155, 0.99517221, 1.02932481, 1.04923235, 1.07165201,
                               1.1074072, 1.13628663, 1.16238786, 1.19292199, 1.2345284,
                               1.2403106, 1.27659058, 1.30487102, 1.31820484, 1.35435169,
                               1.36126559, 1.38600291, 1.41808144, 1.42049657, 1.4753968])]
    u1 = np.array([[0., 0.01370871, 0.02721286, 0.03927982, 0.05452284,
                    0.06793247, 0.08467647, 0.09884958, 0.11188748, 0.12701369,
                    0.14242682, 0.15519745, 0.16426655, 0.17721345, 0.18874887,
                    0.20080715, 0.2174723, 0.22957547, 0.24128003, 0.25284885,
                    0.26999562, 0.27227524, 0.285285, 0.30104995, 0.3106667,
                    0.32308282, 0.34033547, 0.34960294, 0.36711967, 0.36755109,
                    0.38213335, 0.39813622, 0.40431941, 0.41807373, 0.44008736,
                    0.44621208, 0.45729339, 0.47182772, 0.48041673, 0.50118371,
                    0.50578378], [0., 0.0177643, 0.0484185, 0.08466584, 0.11830092,
                                  0.14526983, 0.17521789, 0.1959544, 0.2245966, 0.23822277,
                                  0.26394879, 0.27914504, 0.29824563, 0.31080199, 0.33435344,
                                  0.34475255, 0.35845854, 0.37206035, 0.39446089, 0.40394131,
                                  0.42265103, 0.43150062, 0.44479741, 0.46461682, 0.47565933,
                                  0.48575569, 0.49356322, 0.50997754, 0.52745362, 0.54225208,
                                  0.55063018, 0.56191794, 0.5748904, 0.59308343, 0.59980207,
                                  0.61386689, 0.62652607, 0.6404477, 0.64798804, 0.65706472,
                                  0.67933473], [0., 0.01860281, 0.03618445, 0.05157131, 0.06651107,
                                                0.08038161, 0.09281834, 0.10386108, 0.11924589, 0.13002617,
                                                0.14412546, 0.16067258, 0.16773345, 0.18438164, 0.19582281,
                                                0.20574012, 0.21796031, 0.23243894, 0.24337217, 0.2538692,
                                                0.26680333, 0.28101002, 0.29062747, 0.29779574, 0.31459065,
                                                0.32642133, 0.33884015, 0.34894659, 0.36112478, 0.37493854,
                                                0.38390646, 0.3995448, 0.41034582, 0.42423312, 0.441534,
                                                0.45350752, 0.46307694, 0.47441329, 0.48733603, 0.50068675,
                                                0.51561299],
                   [0., 0.00499312, 0.00937327, 0.01325934, 0.0201864,
                    0.02342159, 0.03017277, 0.03308807, 0.03778782, 0.0401396,
                    0.04797272, 0.05014935, 0.0498213, 0.05927459, 0.0615047,
                    0.06385748, 0.06644146, 0.07134689, 0.07430214, 0.0787555,
                    0.0795194, 0.07970275, 0.08742641, 0.08823035, 0.09368481,
                    0.09892619, 0.09545174, 0.10107404, 0.1014675, 0.10578594,
                    0.10843766, 0.11033136, 0.1121058, 0.1143788, 0.11878527,
                    0.11866421, 0.12307765, 0.12554614, 0.12820374, 0.1338453,
                    0.13209692], [0., 0.06753916, 0.11102386, 0.14171526, 0.16561019,
                                  0.18470288, 0.20110686, 0.21548701, 0.23090971, 0.24588359,
                                  0.25715521, 0.27331621, 0.27901669, 0.29116912, 0.30392201,
                                  0.31685425, 0.32447691, 0.33532122, 0.34488459, 0.34980556,
                                  0.36711906, 0.3724163, 0.38672256, 0.39754085, 0.40694953,
                                  0.41181078, 0.42506026, 0.4347105, 0.4456633, 0.45751324,
                                  0.46231055, 0.46985531, 0.48506576, 0.49429605, 0.49278221,
                                  0.51080041, 0.51818947, 0.5273377, 0.53723917, 0.54475222,
                                  0.56189272], [0., 0.04789961, 0.07651653, 0.09216094, 0.11196046,
                                                0.12604742, 0.13907624, 0.15326034, 0.15988534, 0.16789339,
                                                0.17992271, 0.18796854, 0.19548305, 0.2055738, 0.21052402,
                                                0.21848837, 0.22457757, 0.231984, 0.24354059, 0.24976444,
                                                0.25901347, 0.26271451, 0.27196188, 0.27694731, 0.28917653,
                                                0.29700992, 0.29796911, 0.309138, 0.31974951, 0.32615696,
                                                0.33441138, 0.34054645, 0.34848665, 0.35434708, 0.3557176,
                                                0.37620324, 0.37246803, 0.38837597, 0.39107254, 0.39851604,
                                                0.40797497],
                   [0., 0.06672969, 0.10620611, 0.13393342, 0.15588391,
                    0.17321403, 0.18634844, 0.1994182, 0.21172324, 0.22335018,
                    0.23195684, 0.24295284, 0.25054209, 0.25933377, 0.26967944,
                    0.27611864, 0.28248287, 0.29301519, 0.30000162, 0.30392916,
                    0.31281392, 0.32085651, 0.32841254, 0.33075745, 0.34052802,
                    0.35052362, 0.35457928, 0.36143623, 0.36908656, 0.37500426,
                    0.38275917, 0.38565137, 0.39109715, 0.40094247, 0.41008627,
                    0.41188694, 0.42282457, 0.42458837, 0.43117168, 0.43868673,
                    0.44432574], [0., 0.11955574, 0.1640861, 0.19837867, 0.22337565,
                                  0.24261872, 0.26437644, 0.27878976, 0.29118663, 0.31205011,
                                  0.32381485, 0.33999031, 0.35343822, 0.36836073, 0.38166926,
                                  0.39679554, 0.40953576, 0.41804517, 0.43979394, 0.44116064,
                                  0.45868164, 0.46856915, 0.49008209, 0.49448835, 0.51797729,
                                  0.51739829, 0.54037498, 0.55495042, 0.55546033, 0.5779394,
                                  0.59028687, 0.60248343, 0.61412209, 0.62864904, 0.64096585,
                                  0.65333774, 0.65992137, 0.68423057, 0.69196808, 0.70103968,
                                  0.71655086]])
    u2 = np.array([[0., 0.03098646, 0.05545746, 0.07545278, 0.09587002,
                    0.11218518, 0.1289216, 0.14907976, 0.16050172, 0.17975011,
                    0.19085945, 0.20455536, 0.21882758, 0.22764167, 0.24029007,
                    0.25447976, 0.26779484, 0.27510806, 0.29216032, 0.30202506,
                    0.30831239, 0.31984437, 0.34223229, 0.35177372, 0.35798304,
                    0.3709517, 0.38576783, 0.39326947, 0.40968629, 0.42754807,
                    0.431818, 0.44828232, 0.44874349, 0.47099549, 0.48051323,
                    0.49763808, 0.50824426, 0.51673666, 0.52778818, 0.5532053,
                    0.54917844],
                   [0., 0.07551708, 0.1170532, 0.15216193, 0.18166764,
                    0.20359138, 0.23135444, 0.24912772, 0.27113952, 0.29389022,
                    0.30847596, 0.32541801, 0.33943834, 0.35989591, 0.37180304,
                    0.3859453, 0.39804626, 0.42043131, 0.43533962, 0.44441893,
                    0.45749463, 0.47317309, 0.48438685, 0.49786174, 0.50686493,
                    0.5169629, 0.53642026, 0.54788681, 0.55946063, 0.57297367,
                    0.59183436, 0.60031666, 0.61064276, 0.62670694, 0.64145624,
                    0.64992796, 0.66694115, 0.67537887, 0.69711312, 0.69669367,
                    0.70691219],
                   [0., 0.02689769, 0.04637463, 0.06107751, 0.07982749,
                    0.09594169, 0.1120347, 0.12399533, 0.13631527, 0.15126529,
                    0.16050993, 0.17405488, 0.188608, 0.20339066, 0.21101378,
                    0.22069657, 0.23875572, 0.24928892, 0.25904662, 0.27115289,
                    0.28545744, 0.29950451, 0.30656425, 0.31675572, 0.3286219,
                    0.34925112, 0.35830421, 0.36912206, 0.38357135, 0.39354416,
                    0.40401191, 0.41465255, 0.42368689, 0.4505914, 0.45190246,
                    0.47308189, 0.47628597, 0.49218386, 0.50985051, 0.52424114,
                    0.53579416],
                   [0., 0.01708874, 0.02883109, 0.03667255, 0.04230664,
                    0.05451478, 0.05964277, 0.06556297, 0.06892387, 0.07205092,
                    0.07841962, 0.08109198, 0.0853806, 0.09239387, 0.09310937,
                    0.09813186, 0.10053086, 0.10639613, 0.10846062, 0.11106773,
                    0.11633562, 0.11677947, 0.11836627, 0.12378049, 0.12455875,
                    0.12804027, 0.13414478, 0.13555213, 0.13699382, 0.1417969,
                    0.14485852, 0.14680395, 0.14778798, 0.15108578, 0.1551278,
                    0.15586863, 0.15905876, 0.16184291, 0.16639193, 0.16714403,
                    0.16686271],
                   [0., 0.07672131, 0.12080113, 0.15045971, 0.17705163,
                    0.19692219, 0.21546218, 0.22979872, 0.24408165, 0.25970666,
                    0.26993065, 0.28438958, 0.29309198, 0.30756059, 0.31889584,
                    0.3279575, 0.33792407, 0.35044328, 0.36157932, 0.36997488,
                    0.38073964, 0.38907229, 0.40495923, 0.41054035, 0.42316576,
                    0.42900621, 0.44260092, 0.45901566, 0.4634223, 0.4738164,
                    0.47805477, 0.48548481, 0.50078621, 0.51390053, 0.52269021,
                    0.52380674, 0.54410166, 0.54971428, 0.56596145, 0.57210381,
                    0.58232933],
                   [0., 0.07471898, 0.11631598, 0.14296631, 0.16031253,
                    0.17638093, 0.18957802, 0.20321353, 0.21255584, 0.22103481,
                    0.23047212, 0.23954361, 0.24942612, 0.25670621, 0.2622554,
                    0.27213119, 0.27801828, 0.28188971, 0.29399333, 0.29948858,
                    0.30805492, 0.31951219, 0.32054038, 0.33139569, 0.33699455,
                    0.34272546, 0.3550778, 0.35899377, 0.36825548, 0.37299163,
                    0.38362941, 0.38840386, 0.39988028, 0.40539919, 0.40708572,
                    0.41943303, 0.42176454, 0.43467769, 0.44019444, 0.45153434,
                    0.46259523],
                   [0., 0.08240114, 0.12830295, 0.15852443, 0.1821383,
                    0.20188941, 0.21789249, 0.23301763, 0.24378633, 0.25730949,
                    0.27034973, 0.27702661, 0.28643502, 0.29958311, 0.30489884,
                    0.31298878, 0.3194941, 0.32706576, 0.33494348, 0.34318945,
                    0.35181924, 0.35823116, 0.36484217, 0.37144773, 0.37948287,
                    0.38332729, 0.39318779, 0.40124312, 0.40585105, 0.41014022,
                    0.4157716, 0.42543976, 0.42998526, 0.43793809, 0.44665563,
                    0.45205136, 0.45686589, 0.4609227, 0.46610309, 0.48129111,
                    0.48464345],
                   [0., 0.1033762, 0.15350442, 0.18626652, 0.20710819,
                    0.23075308, 0.24740932, 0.2624254, 0.28090551, 0.30083535,
                    0.31275032, 0.33039902, 0.34281, 0.35477928, 0.37116398,
                    0.38855409, 0.40102353, 0.41034802, 0.42613745, 0.43577597,
                    0.44626494, 0.46325795, 0.48054213, 0.48813205, 0.50378531,
                    0.52053394, 0.53163138, 0.5455243, 0.56083229, 0.57009311,
                    0.58534544, 0.59849949, 0.61672903, 0.62411636, 0.6327805,
                    0.64805585, 0.65686436, 0.67572655, 0.69443073, 0.71092771,
                    0.73139903]])
    u3 = np.array([[0., 0.03081863, 0.0527894, 0.07782224, 0.09537967,
                    0.11349139, 0.13168765, 0.14705853, 0.16220863, 0.17179254,
                    0.19021725, 0.20057298, 0.21408938, 0.22465646, 0.24186168,
                    0.25290945, 0.26331392, 0.27745694, 0.29045497, 0.29765751,
                    0.31266901, 0.32761463, 0.34004139, 0.34708337, 0.36000375,
                    0.37595728, 0.3883401, 0.39307808, 0.40767591, 0.41703234,
                    0.42934089, 0.43911006, 0.45923675, 0.47194087, 0.48390491,
                    0.49571493, 0.50653375, 0.51705657, 0.53829651, 0.5465961,
                    0.56116035], [0., 0.08233712, 0.12747965, 0.17129993, 0.19887041,
                                  0.22627468, 0.24999041, 0.27658098, 0.28910545, 0.31715906,
                                  0.33441932, 0.35661743, 0.36613704, 0.38149848, 0.39602802,
                                  0.41166863, 0.42847398, 0.43653212, 0.45835203, 0.46765623,
                                  0.48406193, 0.49251129, 0.50711631, 0.52052815, 0.53173583,
                                  0.54843912, 0.55819837, 0.57175478, 0.58628646, 0.59514144,
                                  0.60710949, 0.61658135, 0.64226823, 0.64275242, 0.65855208,
                                  0.67230725, 0.68014764, 0.68653887, 0.70041541, 0.7173231,
                                  0.72424264], [0., 0.02645928, 0.04533553, 0.06391671, 0.07997661,
                                                0.09524908, 0.10775454, 0.12443089, 0.13767579, 0.14919801,
                                                0.16063437, 0.17727591, 0.19107149, 0.19795911, 0.21543414,
                                                0.22402926, 0.23337132, 0.24784457, 0.2637198, 0.2704253,
                                                0.28340141, 0.29554628, 0.30505532, 0.31559309, 0.33283123,
                                                0.34709135, 0.35398513, 0.37192588, 0.38272009, 0.39190232,
                                                0.4092616, 0.42210159, 0.42844564, 0.43703027, 0.45647147,
                                                0.46965102, 0.47706894, 0.49146581, 0.51518061, 0.51729404,
                                                0.52750451],
                   [0., 0.01382872, 0.0261459, 0.03424761, 0.04204375,
                    0.04865203, 0.0525434, 0.05905673, 0.0632947, 0.06853196,
                    0.07348816, 0.07973456, 0.08129568, 0.08604582, 0.09291592,
                    0.09359603, 0.09989245, 0.10097941, 0.1088442, 0.11130437,
                    0.11468198, 0.11726117, 0.12245268, 0.12195467, 0.12633606,
                    0.12812196, 0.13314964, 0.13596483, 0.14060527, 0.14152195,
                    0.1414028, 0.14756329, 0.14678283, 0.15122965, 0.15370203,
                    0.15969286, 0.15783312, 0.16366145, 0.16389982, 0.16916611,
                    0.17042002],
                   [0., 0.07856059, 0.12286287, 0.15266845, 0.1750593,
                    0.19496588, 0.21388132, 0.22720148, 0.24348924, 0.25849352,
                    0.26823953, 0.28208186, 0.29662247, 0.30475848, 0.31744225,
                    0.33046281, 0.33605176, 0.34561967, 0.35876194, 0.36766547,
                    0.3812906, 0.3883938, 0.39899888, 0.40692039, 0.41816074,
                    0.42461135, 0.43754497, 0.44038159, 0.4562245, 0.46720164,
                    0.47013881, 0.48770168, 0.49946263, 0.5044829, 0.5083996,
                    0.52783481, 0.52769874, 0.55213556, 0.54970914, 0.56409398,
                    0.57334249],
                   [0., 0.08231723, 0.1252854, 0.14892317, 0.17054632,
                    0.18544656, 0.19769604, 0.21029603, 0.22282106, 0.2336895,
                    0.24077359, 0.25078171, 0.2579476, 0.2671546, 0.27183576,
                    0.28275129, 0.29199246, 0.29586755, 0.30350242, 0.31136387,
                    0.31567599, 0.32927513, 0.33030731, 0.33977915, 0.34625475,
                    0.35384833, 0.3622732, 0.36303362, 0.37580137, 0.38184011,
                    0.3865366, 0.39286376, 0.4015065, 0.41302165, 0.42128013,
                    0.42734722, 0.430626, 0.44084943, 0.44391602, 0.45804634,
                    0.45979836],
                   [0., 0.08910958, 0.12587999, 0.15074427, 0.16879331,
                    0.18569558, 0.19969491, 0.21404039, 0.22651253, 0.23291678,
                    0.24446909, 0.25221156, 0.26274786, 0.27239071, 0.28658094,
                    0.29197025, 0.30073359, 0.30919813, 0.31409007, 0.32005067,
                    0.32935433, 0.33854749, 0.34216383, 0.35097922, 0.35751183,
                    0.37054854, 0.37081861, 0.37965901, 0.38380931, 0.39080162,
                    0.40142117, 0.40701135, 0.41513175, 0.41802382, 0.42758642,
                    0.4363531, 0.43665116, 0.44994632, 0.45483913, 0.4600544,
                    0.46603391],
                   [0., 0.11805766, 0.16316877, 0.19595858, 0.21983699,
                    0.23899707, 0.26000733, 0.27832413, 0.29518585, 0.31487064,
                    0.32542658, 0.34299287, 0.35182493, 0.36783001, 0.38038822,
                    0.39535585, 0.40743, 0.42056828, 0.43471334, 0.44813492,
                    0.45883966, 0.47143409, 0.48850253, 0.49598404, 0.51248737,
                    0.529045, 0.5388413, 0.55375405, 0.5668353, 0.57850531,
                    0.5921315, 0.60825361, 0.6206921, 0.62899631, 0.64005373,
                    0.65394913, 0.67809021, 0.67795569, 0.70161011, 0.7019527,
                    0.7209558]])
    u4 = np.array([[0., 0.0329008, 0.05681413, 0.07320405, 0.09585632,
                    0.11026733, 0.12794564, 0.1415033, 0.1588689, 0.16732073,
                    0.18571756, 0.19289378, 0.20520341, 0.21867071, 0.23413516,
                    0.24511457, 0.25809714, 0.26620574, 0.27909579, 0.2910901,
                    0.30943515, 0.31960034, 0.32608195, 0.34641642, 0.35742107,
                    0.3625134, 0.37539401, 0.38251951, 0.40391444, 0.41061194,
                    0.42783109, 0.44164165, 0.45588619, 0.45697798, 0.4703475,
                    0.48319394, 0.49236782, 0.49956949, 0.52336387, 0.53622955,
                    0.5511688],
                   [0., 0.08076551, 0.12624508, 0.15756616, 0.18811749,
                    0.21259615, 0.23433519, 0.25772869, 0.27751808, 0.29837323,
                    0.31204992, 0.33165051, 0.3459293, 0.36162799, 0.37685366,
                    0.38806836, 0.40932215, 0.41924295, 0.43322648, 0.44979224,
                    0.45653393, 0.47807935, 0.49091664, 0.50571243, 0.51232404,
                    0.5313199, 0.53763218, 0.54979238, 0.56966317, 0.58006376,
                    0.59528592, 0.61324562, 0.62750872, 0.63095055, 0.64300852,
                    0.65688067, 0.67011357, 0.67475167, 0.69605437, 0.70676348,
                    0.71621947],
                   [0., 0.02572493, 0.04633596, 0.06213378, 0.07837258,
                    0.09102376, 0.10501791, 0.12122802, 0.13594022, 0.14526056,
                    0.15997184, 0.17241854, 0.18156726, 0.19676383, 0.20516738,
                    0.21964717, 0.23214709, 0.23853681, 0.25306497, 0.26899251,
                    0.27406361, 0.28833096, 0.30274033, 0.31404764, 0.32656506,
                    0.33207709, 0.35375365, 0.36514041, 0.37933229, 0.38923607,
                    0.39865752, 0.41506893, 0.42555193, 0.44078553, 0.45158345,
                    0.46182527, 0.47782638, 0.48839444, 0.5098406, 0.51464262,
                    0.52631549],
                   [0., 0.01705719, 0.02726291, 0.03806715, 0.0429495,
                    0.0506775, 0.05558712, 0.05862918, 0.06239197, 0.06669482,
                    0.07288592, 0.07451002, 0.08071623, 0.08144461, 0.08614294,
                    0.08954148, 0.09167323, 0.09024303, 0.09877318, 0.09750274,
                    0.09976558, 0.1095682, 0.10633132, 0.11205958, 0.10967206,
                    0.11846346, 0.11905989, 0.1241088, 0.12698743, 0.12700436,
                    0.13029252, 0.13023389, 0.13763632, 0.13425824, 0.14096799,
                    0.145361, 0.14471305, 0.14506705, 0.14685129, 0.15196524,
                    0.14961962],
                   [0., 0.07895384, 0.11422358, 0.14043638, 0.16222946,
                    0.1804383, 0.19706974, 0.21601087, 0.22906672, 0.24270489,
                    0.25592941, 0.26760058, 0.27973768, 0.28982821, 0.30225789,
                    0.31304007, 0.31994169, 0.33183069, 0.34672535, 0.3555045,
                    0.3686865, 0.37458265, 0.38434279, 0.39686141, 0.4021533,
                    0.41524194, 0.42649379, 0.43654619, 0.44479384, 0.46002004,
                    0.46512415, 0.47901508, 0.48387449, 0.49103623, 0.50025014,
                    0.51294049, 0.52592947, 0.53655326, 0.5446245, 0.55020822,
                    0.55623998],
                   [0., 0.07991151, 0.12720012, 0.14924084, 0.17089626,
                    0.18769198, 0.1971003, 0.20912042, 0.21832895, 0.22836286,
                    0.24023138, 0.24537945, 0.25398046, 0.2633968, 0.27375395,
                    0.27990551, 0.28764079, 0.29520356, 0.29953405, 0.30694862,
                    0.31350581, 0.31676465, 0.33266716, 0.33498058, 0.34442284,
                    0.34867304, 0.35832075, 0.36757402, 0.36819781, 0.38334247,
                    0.38548681, 0.39887668, 0.40280978, 0.40712101, 0.42238701,
                    0.42130341, 0.43100562, 0.43752775, 0.45294033, 0.44854182,
                    0.46593281],
                   [0., 0.07904918, 0.12340373, 0.15509668, 0.17779867,
                    0.19846265, 0.20888572, 0.22673303, 0.23797389, 0.24724668,
                    0.2580588, 0.26920367, 0.27558728, 0.28382711, 0.29422291,
                    0.30443162, 0.31174101, 0.31896156, 0.32370365, 0.33105211,
                    0.33904528, 0.34467237, 0.35105718, 0.35700094, 0.36725428,
                    0.3707328, 0.37395795, 0.38320777, 0.39053309, 0.39846676,
                    0.40162943, 0.40371759, 0.41774064, 0.41752659, 0.42135476,
                    0.42754302, 0.43811643, 0.44550828, 0.45340986, 0.46129888,
                    0.45974796],
                   [0., 0.12075714, 0.16489326, 0.19094228, 0.21543934,
                    0.24167953, 0.25481811, 0.2700129, 0.28754893, 0.30540806,
                    0.31439637, 0.32927589, 0.34473749, 0.36121412, 0.37432179,
                    0.38387276, 0.39413125, 0.41184222, 0.42117117, 0.44222711,
                    0.44712865, 0.46216598, 0.47471888, 0.49074205, 0.49983806,
                    0.51702131, 0.52090467, 0.54112512, 0.55795562, 0.57120835,
                    0.58807758, 0.59615115, 0.60424646, 0.62116589, 0.64037946,
                    0.64399186, 0.66507817, 0.67846889, 0.69863474, 0.70598133,
                    0.71343893]])
    u5 = np.array([[0., 0.03419526, 0.06106537, 0.08363693, 0.10023437,
                    0.12010886, 0.1372147, 0.14809132, 0.16644198, 0.17650712,
                    0.19525588, 0.203217, 0.21639757, 0.23067463, 0.23795989,
                    0.25551454, 0.26624964, 0.28650893, 0.29041481, 0.30689358,
                    0.3129451, 0.32606585, 0.34243403, 0.35358092, 0.358743,
                    0.37163603, 0.38809451, 0.4023986, 0.41353393, 0.42984679,
                    0.43689596, 0.44889551, 0.46399161, 0.4725532, 0.48803417,
                    0.50876203, 0.5173254, 0.52765801, 0.5388461, 0.54437467,
                    0.56799155],
                   [0., 0.08460609, 0.13151738, 0.16258146, 0.19057303,
                    0.2179877, 0.23953151, 0.26671567, 0.28155928, 0.30263759,
                    0.32418305, 0.33688382, 0.34874307, 0.36988218, 0.38543992,
                    0.3932005, 0.41100312, 0.42394904, 0.43746898, 0.44893087,
                    0.46790874, 0.47856713, 0.49655061, 0.51004686, 0.51340021,
                    0.53208669, 0.5449762, 0.54936708, 0.56667644, 0.57789963,
                    0.58545746, 0.60417324, 0.61356517, 0.62888066, 0.63707755,
                    0.64949969, 0.66736349, 0.67651526, 0.69319853, 0.71159326,
                    0.71982167],
                   [0., 0.02921259, 0.05145619, 0.06515595, 0.08181224,
                    0.09740054, 0.11069694, 0.12782116, 0.13716524, 0.1488061,
                    0.16047183, 0.17920788, 0.18651375, 0.20246075, 0.21426862,
                    0.22489649, 0.23606615, 0.24917321, 0.26085068, 0.27030882,
                    0.28507927, 0.29876992, 0.30784013, 0.32151349, 0.33405289,
                    0.33680712, 0.36047028, 0.36905788, 0.38452325, 0.39276469,
                    0.4092929, 0.41725528, 0.434302, 0.44460987, 0.45891777,
                    0.47385638, 0.48192982, 0.50045806, 0.51385026, 0.52966898,
                    0.54138884],
                   [0., 0.0180313, 0.02981642, 0.04028218, 0.04802976,
                    0.05478352, 0.06227586, 0.06784539, 0.07162594, 0.07576562,
                    0.08435649, 0.0861502, 0.08957085, 0.09328103, 0.10164419,
                    0.1052676, 0.11001538, 0.10788614, 0.11338114, 0.12121686,
                    0.12182372, 0.12768829, 0.12910016, 0.13591967, 0.13478137,
                    0.1389145, 0.14218716, 0.14192669, 0.14918019, 0.15408787,
                    0.15790428, 0.15751328, 0.15866265, 0.1641125, 0.16391763,
                    0.16620922, 0.17170878, 0.17364372, 0.17840221, 0.1800772,
                    0.18515136],
                   [0., 0.09820298, 0.13979267, 0.17236879, 0.19784291,
                    0.21769372, 0.23379349, 0.24811097, 0.26473189, 0.27958332,
                    0.29354421, 0.30346164, 0.31447766, 0.32359014, 0.33957839,
                    0.34653059, 0.36017463, 0.37135207, 0.37914871, 0.38941131,
                    0.39715482, 0.41540168, 0.42356258, 0.43054496, 0.43982175,
                    0.45364692, 0.45928162, 0.47543354, 0.47802231, 0.48573871,
                    0.5038805, 0.5173376, 0.52991526, 0.53149149, 0.54499781,
                    0.55851449, 0.56199648, 0.56713602, 0.58179126, 0.59004522,
                    0.60575868],
                   [0., 0.09108692, 0.13159347, 0.15629631, 0.17697816,
                    0.19248406, 0.20667378, 0.22033041, 0.22857403, 0.2356067,
                    0.2450487, 0.25734352, 0.2669013, 0.27524461, 0.28170424,
                    0.28503469, 0.29816797, 0.30537364, 0.31263341, 0.31995794,
                    0.32518439, 0.33254311, 0.34223871, 0.35290102, 0.35766044,
                    0.36317362, 0.37121948, 0.37687833, 0.38386578, 0.38763799,
                    0.40324415, 0.40889715, 0.41597955, 0.42106754, 0.43199889,
                    0.44068257, 0.44621341, 0.45003886, 0.45773625, 0.47635379,
                    0.47399941],
                   [0., 0.09656323, 0.13907831, 0.16643895, 0.18831079,
                    0.20416599, 0.22146826, 0.23420992, 0.24605322, 0.25781775,
                    0.27002966, 0.27712129, 0.2904694, 0.2961916, 0.30697965,
                    0.31120752, 0.3189238, 0.33065062, 0.33345647, 0.33850769,
                    0.3483043, 0.35305826, 0.36393986, 0.3689507, 0.37537252,
                    0.38460102, 0.38751136, 0.39535303, 0.40189018, 0.40623969,
                    0.41466916, 0.42075055, 0.43002419, 0.43348089, 0.43896764,
                    0.44243181, 0.45007139, 0.45319369, 0.47043444, 0.47000835,
                    0.47058773],
                   [0., 0.11371192, 0.16381933, 0.19980788, 0.22416634,
                    0.24743092, 0.26194733, 0.28439855, 0.30097705, 0.31643165,
                    0.32849502, 0.34388337, 0.35752436, 0.37122382, 0.37975955,
                    0.39899098, 0.40812569, 0.42161761, 0.43754811, 0.44768242,
                    0.46007175, 0.47146972, 0.48393068, 0.49757308, 0.50934906,
                    0.52662073, 0.54335588, 0.55344027, 0.5574864, 0.57567189,
                    0.59127342, 0.61003942, 0.61539338, 0.63108121, 0.64609013,
                    0.66202173, 0.67396187, 0.68433755, 0.69254069, 0.71184933,
                    0.7269053]])
    heurlist = [u1, u2, u3, u4, u5]
    heur_utillist = [np.array([0.11431596, 0.17342761, 0.27237211, 0.34772361, 0.42201331,
                               0.46421732, 0.52215145, 0.56258228, 0.61307315, 0.65136179,
                               0.70774646, 0.7362975, 0.79181096, 0.8173512, 0.85761582,
                               0.89614471, 0.93459148, 0.96095919, 1.00033909, 1.04123174,
                               1.0565142, 1.10377655, 1.13537857, 1.17279409, 1.1881757,
                               1.22163501, 1.26004209, 1.25844677, 1.32045081, 1.34090466,
                               1.34279745, 1.38799544, 1.39778234, 1.4122447, 1.44379363,
                               1.4545468, 1.48638373, 1.49276271, 1.52909561, 1.53955685]),
                     np.array([0.11824539, 0.17921349, 0.27227615, 0.35563589, 0.40868895,
                               0.45965295, 0.50776644, 0.55323044, 0.60542637, 0.64974086,
                               0.68383406, 0.73754184, 0.77267857, 0.80977957, 0.84993551,
                               0.87872208, 0.92224673, 0.94827536, 0.98459975, 1.00612438,
                               1.04574377, 1.07831565, 1.1074873, 1.14694138, 1.19219577,
                               1.19763791, 1.23639268, 1.2573178, 1.28074392, 1.29342311,
                               1.34334417, 1.35538405, 1.38308684, 1.40606437, 1.42666259,
                               1.46696952, 1.47990614, 1.48967596, 1.51552606, 1.54685017]),
                     np.array([0.11082292, 0.17298705, 0.25435897, 0.3419658, 0.40777705,
                               0.45528758, 0.51026334, 0.55736408, 0.60624828, 0.64403187,
                               0.69499754, 0.74124776, 0.7646112, 0.81675401, 0.85514783,
                               0.89452736, 0.92442669, 0.95527187, 1.00110561, 1.0181837,
                               1.07022645, 1.10149222, 1.12473377, 1.16534315, 1.1972839,
                               1.23593781, 1.25112424, 1.27189965, 1.3046614, 1.32271085,
                               1.34594176, 1.36502416, 1.40905441, 1.42277059, 1.43587757,
                               1.47146837, 1.50057071, 1.50579509, 1.53840853, 1.56139734]),
                     np.array([0.11593436, 0.18133894, 0.27289623, 0.36453934, 0.42639136,
                               0.47121222, 0.52601259, 0.56416475, 0.6124129, 0.65596568,
                               0.70297612, 0.74508809, 0.78056249, 0.81849921, 0.84793013,
                               0.89641819, 0.92243707, 0.95614298, 0.99470177, 1.01673434,
                               1.0432323, 1.09299947, 1.12591187, 1.15542853, 1.18767469,
                               1.22152379, 1.24914204, 1.27425745, 1.29394087, 1.33224555,
                               1.34566491, 1.37065939, 1.39051277, 1.41674257, 1.42917069,
                               1.45926123, 1.49682661, 1.51212307, 1.52563911, 1.54974496]),
                     np.array([0.10602101, 0.18191936, 0.26436622, 0.34609054, 0.4079323,
                               0.45954303, 0.51809874, 0.56184905, 0.60693437, 0.65704142,
                               0.70327859, 0.74942598, 0.77587466, 0.81581962, 0.86073208,
                               0.90353553, 0.92796626, 0.96117227, 0.99589492, 1.02585077,
                               1.05010257, 1.0853512, 1.13742249, 1.15491382, 1.19949646,
                               1.22070716, 1.25596724, 1.26850264, 1.29831861, 1.34410516,
                               1.34979522, 1.38181635, 1.39846767, 1.43422538, 1.45854528,
                               1.46974914, 1.48314679, 1.51447933, 1.52723034, 1.56089958])]
    rudi_utillist = [np.array([0.02782408, 0.04748855, 0.06840516, 0.0879303, 0.10657535,
                               0.12555678, 0.13974917, 0.16492282, 0.18010035, 0.19426166,
                               0.2153278, 0.23156652, 0.24563396, 0.26108007, 0.27959015,
                               0.30408262, 0.33713123, 0.34218969, 0.36414772, 0.40102144,
                               0.40370474, 0.4395967, 0.45107627, 0.47007509, 0.48017034,
                               0.51198321, 0.5283318, 0.54597593, 0.57088198, 0.58599377,
                               0.61389474, 0.62529131, 0.65915883, 0.68747812, 0.69685301,
                               0.71871954, 0.73572084, 0.74629047, 0.79783955, 0.79112253]),
                     np.array([0.04859418, 0.06086642, 0.08853113, 0.11049274, 0.13175328,
                               0.15241292, 0.16916202, 0.18803521, 0.20508796, 0.22351268,
                               0.24283847, 0.26074275, 0.2739954, 0.29094278, 0.30773907,
                               0.33223479, 0.37037767, 0.37209796, 0.38763903, 0.42613958,
                               0.4327728, 0.46529419, 0.471341, 0.49284701, 0.51287379,
                               0.52775766, 0.54722181, 0.56469771, 0.58150908, 0.60790747,
                               0.63083578, 0.66514928, 0.66939849, 0.69113055, 0.7003905,
                               0.72869131, 0.75964309, 0.77017693, 0.81072421, 0.81706695]),
                     np.array([0.03555813, 0.04437638, 0.06634329, 0.0834248, 0.10136096,
                               0.12448743, 0.14371782, 0.16463435, 0.1863686, 0.204936,
                               0.2250287, 0.24721582, 0.26055852, 0.28508597, 0.3052582,
                               0.32501774, 0.3643604, 0.36455279, 0.38954529, 0.42753366,
                               0.4251898, 0.47054249, 0.46903591, 0.49450484, 0.51422276,
                               0.53934689, 0.56051306, 0.57724114, 0.60352949, 0.62303856,
                               0.64193006, 0.66176272, 0.68444669, 0.70732812, 0.73515595,
                               0.73560723, 0.76790717, 0.78161387, 0.81333717, 0.81489961]),
                     np.array([0.01348256, 0.0153099, 0.03553821, 0.04718251, 0.06656089,
                               0.08151434, 0.10044076, 0.12058882, 0.13342243, 0.16056361,
                               0.1773875, 0.19423595, 0.20479916, 0.23138496, 0.25431674,
                               0.27062901, 0.30343841, 0.30128138, 0.33126151, 0.3691016,
                               0.3670993, 0.41737146, 0.41161035, 0.4324922, 0.45876234,
                               0.47349503, 0.48750493, 0.51246764, 0.5408795, 0.55339761,
                               0.57083161, 0.58973429, 0.62158921, 0.64219545, 0.67086635,
                               0.6905359, 0.7018246, 0.71182997, 0.7402795, 0.75600682]),
                     np.array([0.02551143, 0.03993898, 0.07005365, 0.09526721, 0.117477,
                               0.14293463, 0.16138723, 0.18074229, 0.20148731, 0.21889254,
                               0.24584289, 0.25945681, 0.27780987, 0.29191529, 0.31487885,
                               0.33695287, 0.37468603, 0.37310461, 0.3993615, 0.42922739,
                               0.43742476, 0.48360472, 0.48553688, 0.49965235, 0.52589264,
                               0.54465636, 0.56376548, 0.58338899, 0.61340136, 0.63284834,
                               0.64230094, 0.67870712, 0.70463026, 0.70434832, 0.7305568,
                               0.75971935, 0.76436461, 0.80001255, 0.8283422, 0.82860185]), ]

    # Size of dashes for unexplored nodes
    dshSz = 2
    # Size of figure layout
    figtup = (7, 5)
    titleSz, axSz, labelSz = 12, 10, 9
    xMax = 450

    avgHeurMat = np.average(np.array(heurlist), axis=0)

    # Plot of marginal utilities
    colors = cm.rainbow(np.linspace(0, 1., numTN))
    labels = [tnNames[ind] for ind in range(numTN)]

    x = range(testInt, testMax + 1, testInt)
    deltaArr = np.zeros((avgHeurMat.shape[0], avgHeurMat.shape[1] - 1))
    for rw in range(deltaArr.shape[0]):
        for col in range(deltaArr.shape[1]):
            deltaArr[rw, col] = avgHeurMat[rw, col + 1] - avgHeurMat[rw, col]
    yMax = np.max(deltaArr) * 1.1

    _ = plt.figure(figsize=figtup)
    for tnind in range(numTN):
        if tnind < 4:
            plt.plot(x, deltaArr[tnind], linewidth=2, color=colors[tnind],
                     label=labels[tnind], alpha=0.6)
        else:
            plt.plot(x, deltaArr[tnind], linewidth=2, color=colors[tnind],
                     label=labels[tnind], alpha=0.6, dashes=[1, dshSz])
    for tnind in range(numTN):
        plt.text(testInt * 1.1, deltaArr[tnind, 0], labels[tnind].ljust(15), fontsize=labelSz - 1)
    plt.legend(fontsize=labelSz)
    plt.ylim([0., yMax])
    plt.xlim([0., xMax])
    plt.xlabel('Number of Tests', fontsize=axSz)
    plt.ylabel('Marginal Utility Gain', fontsize=axSz)
    plt.title('Marginal Utility with Increasing Tests\nAll-Provinces Setting', fontsize=titleSz)
    plt.show()
    plt.close()

    # Allocation plot
    allocArr, objValArr = sampf.smooth_alloc_forward(avgHeurMat)

    # average distance from uniform allocation
    # np.linalg.norm(allocArr[:,-1]-np.ones((8))*4)

    colors = cm.rainbow(np.linspace(0, 1., numTN))
    labels = [tnNames[ind] for ind in range(numTN)]
    x = range(testInt, testMax + 1, testInt)
    _ = plt.figure(figsize=figtup)
    for tnind in range(numTN):
        if tnind < 4:
            plt.plot(x, allocArr[tnind] * testInt, linewidth=2, color=colors[tnind],
                     label=labels[tnind], alpha=0.6)
        else:
            plt.plot(x, allocArr[tnind] * testInt, linewidth=2, color=colors[tnind],
                     label=labels[tnind], alpha=0.6, dashes=[1, dshSz])
    # allocMax = allocArr.max() * testInt * 1.1
    allocMax = 185
    adj = 2.5
    for tnind in range(numTN):
        if tnind == 0:
            plt.text(testMax * 1.01, allocArr[tnind, -1] * testInt - adj, labels[tnind].ljust(15), fontsize=labelSz - 1)
        elif tnind == 6:
            plt.text(testMax * 1.01, allocArr[tnind, -1] * testInt + adj, labels[tnind].ljust(15), fontsize=labelSz - 1)
        else:
            plt.text(testMax * 1.01, allocArr[tnind, -1] * testInt, labels[tnind].ljust(15), fontsize=labelSz - 1)
    plt.legend(fontsize=labelSz)
    plt.ylim([0., allocMax])
    plt.xlim([0., xMax])
    plt.xlabel('Sampling Budget', fontsize=axSz)
    plt.ylabel('Test Node Allocation', fontsize=axSz)
    plt.title('Sampling Plan vs. Budget\nAll-Provinces Setting', fontsize=titleSz)
    # plt.tight_layout()
    plt.show()
    plt.close()

    # Utility comparison plot
    colors = cm.rainbow(np.linspace(0, 0.8, 3))
    labels = ['Utility-Informed', 'Uniform', 'Fixed']
    x = range(testInt, testMax + 1, testInt)
    margUtilGroupList = [heur_utillist, unif_utillist, rudi_utillist]
    utilMax = -1
    for lst in margUtilGroupList:
        currMax = np.amax(np.array(lst))
        if currMax > utilMax:
            utilMax = currMax
    utilMax = utilMax * 1.1

    _ = plt.figure(figsize=figtup)
    for groupInd, margUtilGroup in enumerate(margUtilGroupList):
        groupArr = np.array(margUtilGroup)
        groupAvgArr = np.average(groupArr, axis=0)
        # Compile error bars
        stdevs = [np.std(groupArr[:, i]) for i in range(groupArr.shape[1])]
        group05Arr = [groupAvgArr[i] - (1.96 * stdevs[i] / np.sqrt(groupArr.shape[0])) for i in
                      range(groupArr.shape[1])]
        group95Arr = [groupAvgArr[i] + (1.96 * stdevs[i] / np.sqrt(groupArr.shape[0])) for i in
                      range(groupArr.shape[1])]
        plt.plot(x, groupAvgArr, color=colors[groupInd], linewidth=0.7, alpha=1., label=labels[groupInd] + ' 95% CI')
        plt.fill_between(x, groupAvgArr, group05Arr, color=colors[groupInd], alpha=0.2)
        plt.fill_between(x, groupAvgArr, group95Arr, color=colors[groupInd], alpha=0.2)
        # Line label
        plt.text(x[-1] * 1.01, groupAvgArr[-1], labels[groupInd].ljust(15), fontsize=labelSz - 1)
    plt.ylim(0, utilMax)
    # plt.xlim(0,x[-1]*1.12)
    plt.xlim([0., xMax])
    leg = plt.legend(loc='upper left', fontsize=labelSz)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.0)
    plt.xlabel('Sampling Budget', fontsize=axSz)
    plt.ylabel('Plan Utility', fontsize=axSz)
    plt.title('Utility from Utility-Informed, Uniform and Fixed Allocations\nAll-Provinces Setting', fontsize=titleSz)
    # Add text box showing budgetary savings
    compUtilAvg = np.average(np.array(heur_utillist), axis=0)
    x2, x3 = 119, 325
    plt.plot([100, x3], [compUtilAvg[9], compUtilAvg[9]], color='black', linestyle='--')
    iv = 0.03
    plt.plot([100, 100], [compUtilAvg[9] - iv, compUtilAvg[9] + iv], color='black', linestyle='--')
    plt.plot([x2, x2], [compUtilAvg[9] - iv, compUtilAvg[9] + iv], color='black', linestyle='--')
    plt.plot([x3, x3], [compUtilAvg[9] - iv, compUtilAvg[9] + iv], color='black', linestyle='--')
    plt.text(105, compUtilAvg[9] + iv / 2, str(x2 - 100), fontsize=labelSz)
    plt.text(205, compUtilAvg[9] + iv / 2, str(x3 - x2), fontsize=labelSz)
    # plt.tight_layout()
    plt.show()
    plt.close()

    '''
    Determining the budget saved for the sensitivity table
    currCompInd = 8
    compUtilAvg = np.average(np.array(compUtilList),axis=0) 
    evenUtilArr = np.array(evenUtilList)
    evenAvgArr = np.average(evenUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(evenAvgArr.tolist()) if val > compUtilAvg[currCompInd])
    evenSampSaved = round((compUtilAvg[currCompInd] - evenAvgArr[kInd - 1]) / (evenAvgArr[kInd] - evenAvgArr[kInd - 1]) * testInt) + (
                kInd - 1) * testInt - (currCompInd*testInt)
    print(evenSampSaved)
    rudiUtilArr = np.array(origUtilList)
    rudiAvgArr = np.average(rudiUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(rudiAvgArr.tolist()) if val > compUtilAvg[currCompInd])
    rudiSampSaved = round((compUtilAvg[currCompInd] - rudiAvgArr[kInd - 1]) / (rudiAvgArr[kInd] - rudiAvgArr[kInd - 1]) * testInt) + (
                kInd - 1) * testInt - (currCompInd*testInt)
    print(rudiSampSaved)
    currCompInd = 17
    compUtilAvg = np.average(np.array(compUtilList),axis=0) 
    evenUtilArr = np.array(evenUtilList)
    evenAvgArr = np.average(evenUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(evenAvgArr.tolist()) if val > compUtilAvg[currCompInd])
    evenSampSaved = round((compUtilAvg[currCompInd] - evenAvgArr[kInd - 1]) / (evenAvgArr[kInd] - evenAvgArr[kInd - 1]) * testInt) + (
                kInd - 1) * testInt - (currCompInd*testInt)
    print(evenSampSaved)
    rudiUtilArr = np.array(origUtilList)
    rudiAvgArr = np.average(rudiUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(rudiAvgArr.tolist()) if val > compUtilAvg[currCompInd])
    rudiSampSaved = round((compUtilAvg[currCompInd] - rudiAvgArr[kInd - 1]) / (rudiAvgArr[kInd] - rudiAvgArr[kInd - 1]) * testInt) + (
                kInd - 1) * testInt - (currCompInd*testInt)
    print(rudiSampSaved)
    '''

    return


def casestudyplots_exploratory_market_OLD():
    """
    Cleaned up plots for use in case study in paper
    """
    testMax, testInt = 400, 10
    tnNames = ['MOD_39', 'MOD_17', 'MODHIGH_95', 'MODHIGH_26',
               'MODHIGH_EXPL_1', 'MOD_EXPL_1', 'MODHIGH_EXPL_2', 'MOD_EXPL_2']
    numTN = len(tnNames)

    unif_utillist = [np.array([0.01464967, 0.0229887, 0.03584062, 0.04179026, 0.04775673,
                               0.05543825, 0.06025951, 0.06475962, 0.06894527, 0.07285823,
                               0.07772007, 0.08107989, 0.08526541, 0.08870335, 0.0918858,
                               0.09471104, 0.09881675, 0.10139817, 0.10384348, 0.10754363,
                               0.1094074, 0.11270981, 0.11473015, 0.11667244, 0.12068755,
                               0.12226979, 0.12450583, 0.12648042, 0.12857005, 0.13127864,
                               0.13321517, 0.13420477, 0.13694135, 0.13838002, 0.13928167,
                               0.14230523, 0.14408117, 0.1456639, 0.14596233, 0.14952923]),
                     np.array([0.0139362, 0.0225148, 0.03599858, 0.04298375, 0.04888883,
                               0.05697708, 0.06121599, 0.06585707, 0.07055392, 0.07428761,
                               0.07974163, 0.08269428, 0.08626072, 0.0906986, 0.09361123,
                               0.09650964, 0.09964081, 0.10295796, 0.10538408, 0.10810733,
                               0.11172224, 0.11517426, 0.11641704, 0.11874652, 0.12139643,
                               0.12363176, 0.12632997, 0.12897328, 0.13049864, 0.13256257,
                               0.1345099, 0.13647788, 0.13862263, 0.13865588, 0.14197676,
                               0.14361184, 0.14621618, 0.14602534, 0.14862478, 0.14923932]),
                     np.array([0.00923481, 0.01651557, 0.03004286, 0.03602463, 0.04222123,
                               0.0493772, 0.05358076, 0.05787152, 0.06187306, 0.06536282,
                               0.07055878, 0.07438939, 0.07815394, 0.08227689, 0.0849347,
                               0.08769323, 0.09175722, 0.09364522, 0.09720825, 0.1005212,
                               0.10303259, 0.10596367, 0.10827563, 0.11047925, 0.11279049,
                               0.11394729, 0.11797724, 0.11974588, 0.12195965, 0.12356092,
                               0.12561642, 0.12684814, 0.12933415, 0.13072326, 0.13249803,
                               0.13492749, 0.13615158, 0.13719615, 0.13913036, 0.14054828]),
                     np.array([0.01439621, 0.02228214, 0.03466415, 0.04154927, 0.04738035,
                               0.05599791, 0.06049649, 0.06468376, 0.06897861, 0.07240472,
                               0.07801586, 0.08221604, 0.08460067, 0.08846472, 0.09218048,
                               0.09564121, 0.09756695, 0.10040465, 0.10381762, 0.10715562,
                               0.10937437, 0.11222642, 0.11426446, 0.11707731, 0.11870296,
                               0.12161138, 0.12412389, 0.12714744, 0.1275714, 0.13021352,
                               0.13173752, 0.13395222, 0.13763802, 0.13812745, 0.13987455,
                               0.14234941, 0.14319524, 0.14487606, 0.14707328, 0.14659313]),
                     np.array([0.01257098, 0.02139432, 0.03409391, 0.04060844, 0.04601931,
                               0.0540546, 0.05850705, 0.06350191, 0.06672185, 0.07153847,
                               0.07603903, 0.07944896, 0.08310238, 0.08768118, 0.0906845,
                               0.09334303, 0.09641097, 0.09910477, 0.10345229, 0.10515126,
                               0.10758629, 0.11188565, 0.11384016, 0.11693916, 0.11818187,
                               0.12009328, 0.1226744, 0.12459362, 0.12707462, 0.12981212,
                               0.13214225, 0.13333261, 0.13671357, 0.13758735, 0.13931797,
                               0.14110086, 0.14298402, 0.14469668, 0.14562638, 0.14757289]),
                     np.array([0.01165207, 0.02097778, 0.03484701, 0.04153442, 0.04683575,
                               0.05504851, 0.05997599, 0.06437074, 0.0685178, 0.07318873,
                               0.07826873, 0.08210943, 0.08471372, 0.08977436, 0.09254309,
                               0.096578, 0.09837596, 0.10134863, 0.10461208, 0.10879483,
                               0.11136273, 0.11329766, 0.11650803, 0.11931556, 0.12201899,
                               0.12405044, 0.12513765, 0.12932461, 0.13027671, 0.13277558,
                               0.13423078, 0.13667872, 0.13931965, 0.14067241, 0.14172243,
                               0.14490939, 0.14526402, 0.14844154, 0.14930412, 0.15060652]),
                     np.array([0.01311511, 0.02240771, 0.03605929, 0.04223084, 0.04782919,
                               0.05590389, 0.06036553, 0.06517178, 0.06986181, 0.07289917,
                               0.07827076, 0.08200794, 0.08561967, 0.09046584, 0.09287982,
                               0.09605129, 0.09922784, 0.1021162, 0.10515267, 0.10859804,
                               0.11228896, 0.11408649, 0.11570524, 0.11887276, 0.12207649,
                               0.12273191, 0.12721416, 0.12776474, 0.12958381, 0.13220313,
                               0.13528835, 0.13632453, 0.13866308, 0.14029551, 0.14271408,
                               0.14408406, 0.14431634, 0.14787882, 0.14838643, 0.15009423])
                     ]
    u1 = np.array([[0., 0.00294156, 0.00505353, 0.00622081, 0.00833352,
                    0.01016614, 0.01167477, 0.01308104, 0.01442442, 0.01605372,
                    0.01700349, 0.01872152, 0.01982057, 0.02110486, 0.02251956,
                    0.02300636, 0.0242151, 0.02536069, 0.0266946, 0.02797237,
                    0.02888131, 0.0302996, 0.03178587, 0.03214798, 0.03403775,
                    0.03438991, 0.03600066, 0.03737367, 0.03858834, 0.03999715,
                    0.04146897, 0.04215595, 0.04339687, 0.04421634, 0.04548668,
                    0.04603785, 0.04873142, 0.04920739, 0.0501841, 0.05102084,
                    0.05196587],
                   [0., 0.00491095, 0.00700718, 0.00805448, 0.00934249,
                    0.01009402, 0.01114307, 0.01176502, 0.01242792, 0.01350043,
                    0.01441502, 0.01500855, 0.01611189, 0.01687593, 0.017714,
                    0.01877624, 0.01997259, 0.02052398, 0.02163861, 0.02256561,
                    0.02371864, 0.02470197, 0.02566034, 0.02653166, 0.02744203,
                    0.02836392, 0.02931027, 0.03078558, 0.03199085, 0.03363543,
                    0.03489766, 0.03598385, 0.03651481, 0.03774292, 0.03863593,
                    0.04028127, 0.04115285, 0.04313653, 0.04300859, 0.04383249,
                    0.04595533],
                   [0., 0.00238291, 0.00344937, 0.00448634, 0.00528027,
                    0.00602451, 0.00700927, 0.00766571, 0.00815682, 0.00926578,
                    0.0096705, 0.01052858, 0.01131783, 0.01183957, 0.01292317,
                    0.01332831, 0.0145101, 0.01477779, 0.01612766, 0.01680281,
                    0.01751967, 0.01844222, 0.01975638, 0.02036133, 0.02178071,
                    0.02271226, 0.02333835, 0.02432454, 0.02553437, 0.02646583,
                    0.02742135, 0.0288147, 0.02921815, 0.03151026, 0.03150182,
                    0.03237576, 0.03412854, 0.03550132, 0.03573409, 0.03703065,
                    0.0383397],
                   [0., 0.00259007, 0.00424876, 0.00540979, 0.0065792,
                    0.00763452, 0.0085711, 0.00936231, 0.00993063, 0.01070996,
                    0.01137705, 0.01211878, 0.01273082, 0.01330849, 0.01395264,
                    0.01439195, 0.0147341, 0.01505793, 0.01570199, 0.01603922,
                    0.01639898, 0.01670605, 0.01761948, 0.01780266, 0.01828775,
                    0.01848148, 0.01883935, 0.01899126, 0.01936471, 0.01959431,
                    0.0201425, 0.02016977, 0.02073374, 0.02096085, 0.02143542,
                    0.02176227, 0.02183575, 0.02243779, 0.02244614, 0.02285064,
                    0.02317446],
                   [0., 0.02233237, 0.02885546, 0.0325918, 0.03527222,
                    0.03737751, 0.03905537, 0.04053635, 0.04178747, 0.04290894,
                    0.04423204, 0.04520843, 0.04602293, 0.04727829, 0.04790469,
                    0.04887675, 0.04983815, 0.05018195, 0.05155552, 0.05217648,
                    0.05308929, 0.05418031, 0.05473924, 0.05529253, 0.05629469,
                    0.05729903, 0.05839222, 0.05897359, 0.05968929, 0.06057427,
                    0.06178252, 0.06178343, 0.06300318, 0.06353017, 0.06476846,
                    0.0653556, 0.06647302, 0.0671801, 0.06839626, 0.06839037,
                    0.07039903],
                   [0., 0.02247959, 0.03028494, 0.03459138, 0.03819429,
                    0.04058716, 0.04254356, 0.04427986, 0.04610555, 0.04740889,
                    0.04849462, 0.04991845, 0.05083364, 0.05208966, 0.05277805,
                    0.05368482, 0.05501408, 0.05536098, 0.05643717, 0.05777378,
                    0.05849088, 0.05946219, 0.06002456, 0.06100677, 0.06186559,
                    0.06229836, 0.06290462, 0.06428467, 0.06450949, 0.06556034,
                    0.06609305, 0.0669713, 0.06804497, 0.06858331, 0.06959695,
                    0.07071425, 0.07126606, 0.07151226, 0.0723019, 0.07321877,
                    0.07382898],
                   [0., 0.01267891, 0.01548799, 0.01725044, 0.01879362,
                    0.01989068, 0.02085653, 0.02176546, 0.02294816, 0.02357287,
                    0.02441636, 0.02525693, 0.02589601, 0.02632173, 0.02730554,
                    0.02785034, 0.02858383, 0.02919281, 0.02947804, 0.03044411,
                    0.03128764, 0.03168889, 0.03242975, 0.03294639, 0.03309182,
                    0.03407238, 0.03489403, 0.03541111, 0.03575074, 0.03648439,
                    0.03728957, 0.03769298, 0.03770279, 0.03863778, 0.03977777,
                    0.03999375, 0.0407124, 0.04150927, 0.04222682, 0.04220722,
                    0.04299015],
                   [0., 0.0108917, 0.01494471, 0.01745847, 0.01895519,
                    0.0205655, 0.02155158, 0.0228554, 0.02405234, 0.02486323,
                    0.02635916, 0.02705598, 0.02788735, 0.02908409, 0.02957137,
                    0.03090377, 0.03201417, 0.03319185, 0.03383869, 0.03526646,
                    0.03647863, 0.03694083, 0.03838219, 0.04005099, 0.04085091,
                    0.04177836, 0.04278106, 0.04410066, 0.04535052, 0.04635712,
                    0.04770807, 0.04856994, 0.05026377, 0.05070915, 0.05234411,
                    0.05360933, 0.05506561, 0.05578664, 0.0580636, 0.05784155,
                    0.05910849]])
    u2 = np.array([[0., 0.0017391, 0.00413226, 0.00631547, 0.00793208,
                    0.00963334, 0.01159244, 0.01324519, 0.01425938, 0.01568667,
                    0.01721078, 0.01854201, 0.01964271, 0.02081237, 0.02175855,
                    0.02303991, 0.02424796, 0.02556294, 0.02671244, 0.02806033,
                    0.02942707, 0.02958896, 0.03155797, 0.03252214, 0.03312585,
                    0.03495868, 0.03641971, 0.03685336, 0.03829092, 0.03938805,
                    0.04095306, 0.04146367, 0.04262935, 0.04447943, 0.04481056,
                    0.04619983, 0.04810934, 0.04959416, 0.0505974, 0.05178179,
                    0.05319956],
                   [0., 0.00231791, 0.00350776, 0.00481711, 0.00613428,
                    0.00709796, 0.00824363, 0.00928286, 0.01014489, 0.01140601,
                    0.01255994, 0.01345432, 0.01436086, 0.0155068, 0.01632129,
                    0.01760341, 0.01861113, 0.01922368, 0.02056294, 0.02149764,
                    0.02265735, 0.02353863, 0.02481567, 0.02581908, 0.02722326,
                    0.02784546, 0.02950156, 0.03081676, 0.0312838, 0.03267807,
                    0.03414836, 0.03468929, 0.03632871, 0.03764755, 0.03779825,
                    0.03942321, 0.04025591, 0.04265774, 0.04306133, 0.04493257,
                    0.04606077],
                   [0., 0.0009454, 0.00213638, 0.00284136, 0.00385025,
                    0.00454305, 0.00517511, 0.00628037, 0.00695672, 0.00805026,
                    0.00907689, 0.00973237, 0.01069154, 0.01127122, 0.01216995,
                    0.01326622, 0.01430017, 0.01483581, 0.01569763, 0.01669873,
                    0.01753402, 0.01832635, 0.01932405, 0.02064235, 0.02126214,
                    0.02219918, 0.02344619, 0.02458658, 0.02541872, 0.02712247,
                    0.02738855, 0.0287634, 0.02967205, 0.03069341, 0.03214733,
                    0.03290439, 0.03378069, 0.03495422, 0.03581358, 0.03722088,
                    0.03902371],
                   [0., 0.00179716, 0.00336623, 0.00431036, 0.00570964,
                    0.00680191, 0.00792687, 0.0083432, 0.00930071, 0.00989137,
                    0.01080367, 0.01163467, 0.01208788, 0.01285389, 0.01314359,
                    0.01397297, 0.01438498, 0.01495565, 0.01524248, 0.01559405,
                    0.01616131, 0.0164277, 0.01692834, 0.01749708, 0.01774813,
                    0.0179802, 0.01799058, 0.01870734, 0.01912318, 0.01964657,
                    0.01978837, 0.02028394, 0.02082708, 0.02077518, 0.0209773,
                    0.02159124, 0.02163671, 0.02201062, 0.02259383, 0.02278231,
                    0.02310111],
                   [0., 0.02742735, 0.03302642, 0.03588228, 0.03788424,
                    0.03900204, 0.04042585, 0.04106171, 0.04228469, 0.04301551,
                    0.04425758, 0.04474978, 0.04562603, 0.04652927, 0.04740158,
                    0.04830957, 0.0491803, 0.04993549, 0.05042071, 0.0513904,
                    0.05229612, 0.05310801, 0.05375726, 0.05412177, 0.05577907,
                    0.05628909, 0.05715499, 0.05736192, 0.05900464, 0.05957257,
                    0.06053306, 0.06078261, 0.06201133, 0.06276236, 0.06328475,
                    0.06527064, 0.06491799, 0.0665144, 0.06687056, 0.06775844,
                    0.06839921],
                   [0., 0.0225182, 0.0294706, 0.03331682, 0.0363048,
                    0.03873126, 0.04086881, 0.04256863, 0.04402254, 0.04562137,
                    0.04676107, 0.0480464, 0.0492577, 0.05022138, 0.05113628,
                    0.0520775, 0.05340016, 0.05395093, 0.05512483, 0.05638137,
                    0.05736809, 0.05819165, 0.05854991, 0.05904234, 0.06012375,
                    0.0613492, 0.06183886, 0.06293424, 0.06308086, 0.06402376,
                    0.06558966, 0.06615244, 0.06660281, 0.06723777, 0.06793817,
                    0.06931329, 0.06965837, 0.07086099, 0.0711183, 0.07166538,
                    0.07279197],
                   [0., 0.00635727, 0.00944357, 0.01175358, 0.01321331,
                    0.0144822, 0.01601747, 0.01678146, 0.01793414, 0.01863792,
                    0.01987791, 0.02039072, 0.02130575, 0.0223351, 0.02295511,
                    0.02371655, 0.0244535, 0.02536343, 0.02609373, 0.02642666,
                    0.02745635, 0.02790724, 0.02837081, 0.02937632, 0.0295128,
                    0.03041032, 0.03123937, 0.03183153, 0.03258291, 0.03321887,
                    0.03395253, 0.03463422, 0.03503193, 0.03565558, 0.03649766,
                    0.03659866, 0.03745525, 0.03841801, 0.03855072, 0.03908428,
                    0.04007217],
                   [0., 0.00666261, 0.01016339, 0.0125588, 0.01485631,
                    0.01639646, 0.01780462, 0.01916847, 0.02045316, 0.0214463,
                    0.02263655, 0.02368692, 0.02480178, 0.02591129, 0.02713168,
                    0.02806906, 0.02919227, 0.03017831, 0.03130595, 0.03245392,
                    0.0332501, 0.03471176, 0.03540512, 0.03789307, 0.03823173,
                    0.03879365, 0.04012662, 0.04130752, 0.04269316, 0.04395496,
                    0.04538693, 0.04664258, 0.04754776, 0.0487676, 0.04948142,
                    0.05135684, 0.05261211, 0.05360377, 0.05513707, 0.05561504,
                    0.05716933]])
    u3 = np.array([[0., 0.00292447, 0.00480057, 0.00658342, 0.00822542,
                    0.00944735, 0.01066059, 0.01153016, 0.01281226, 0.0136289,
                    0.01495729, 0.01601854, 0.01721076, 0.01833174, 0.01929691,
                    0.01995287, 0.02122343, 0.02212817, 0.02318724, 0.02425353,
                    0.02533391, 0.02637504, 0.02706205, 0.02885997, 0.02951471,
                    0.03126011, 0.03130787, 0.03273876, 0.03424094, 0.03555594,
                    0.03664342, 0.0370949, 0.03828725, 0.03841933, 0.04056766,
                    0.04179399, 0.04264748, 0.04399446, 0.04487499, 0.04588711,
                    0.04639787],
                   [0., 0.00190322, 0.00306546, 0.00400407, 0.00482337,
                    0.00563315, 0.00657345, 0.00734054, 0.00789975, 0.00880256,
                    0.00965498, 0.0103446, 0.01148912, 0.01220097, 0.01313068,
                    0.01372557, 0.01461603, 0.0155163, 0.016531, 0.01706941,
                    0.01861269, 0.01978363, 0.02027005, 0.02145154, 0.02239371,
                    0.02272158, 0.02426778, 0.02630002, 0.02660261, 0.02744996,
                    0.02887531, 0.02993215, 0.03106347, 0.03232399, 0.03365166,
                    0.03405152, 0.03569531, 0.03682333, 0.03736247, 0.03852051,
                    0.03949081],
                   [0., 0.0011884, 0.00195679, 0.00261055, 0.00337473,
                    0.00396699, 0.00467606, 0.00506884, 0.0058727, 0.00660392,
                    0.00705081, 0.0077327, 0.00864091, 0.00896095, 0.01025279,
                    0.01057569, 0.01115508, 0.01201242, 0.01253347, 0.01318974,
                    0.01419501, 0.01488692, 0.01594896, 0.01664329, 0.01796224,
                    0.01879842, 0.0191551, 0.01985812, 0.02103185, 0.02214024,
                    0.02264387, 0.0240517, 0.0247183, 0.02580615, 0.02715531,
                    0.02856513, 0.02872878, 0.02965486, 0.03153891, 0.0321365,
                    0.03291353],
                   [0., 0.00201785, 0.00313398, 0.00382026, 0.00445497,
                    0.00508291, 0.00592151, 0.00614427, 0.0067873, 0.00757806,
                    0.00811553, 0.00850167, 0.00909221, 0.00947603, 0.00989742,
                    0.01021754, 0.01052604, 0.010977, 0.0116524, 0.0116868,
                    0.01222512, 0.01282082, 0.01275281, 0.01324527, 0.01353942,
                    0.0141051, 0.0142847, 0.01460722, 0.01482567, 0.0153177,
                    0.01566496, 0.0160387, 0.01612715, 0.01662379, 0.01659124,
                    0.01694377, 0.01753346, 0.01759724, 0.01771382, 0.01811891,
                    0.01865521],
                   [0., 0.0192249, 0.02513563, 0.02830984, 0.03053475,
                    0.03244553, 0.03363943, 0.03513867, 0.03646008, 0.03746618,
                    0.03840732, 0.03938065, 0.04012179, 0.04138335, 0.04188101,
                    0.04272292, 0.0440334, 0.04450651, 0.04571878, 0.0465795,
                    0.04693199, 0.04748132, 0.04846783, 0.04940526, 0.05049572,
                    0.05117543, 0.05187601, 0.05231591, 0.05339594, 0.05367199,
                    0.05492437, 0.05591361, 0.05658002, 0.05677028, 0.05794597,
                    0.05855944, 0.0597263, 0.0606767, 0.06089329, 0.06238665,
                    0.06230869],
                   [0., 0.01514981, 0.02251309, 0.0269226, 0.03031565,
                    0.03297344, 0.03510183, 0.03717064, 0.03843964, 0.04004105,
                    0.04151236, 0.04260006, 0.04377768, 0.04488986, 0.04611108,
                    0.04667239, 0.04762304, 0.04875662, 0.04973024, 0.05026295,
                    0.05127754, 0.05245856, 0.05270457, 0.05371764, 0.05468581,
                    0.05539387, 0.05582306, 0.05732158, 0.05753845, 0.05866103,
                    0.05890715, 0.05968901, 0.06047244, 0.06087026, 0.06175572,
                    0.06269276, 0.0630814, 0.06384716, 0.06476997, 0.06577574,
                    0.06612679],
                   [0., 0.00278294, 0.00521942, 0.00684575, 0.00841767,
                    0.00972256, 0.01077953, 0.01161165, 0.01283437, 0.01348916,
                    0.01448782, 0.01527355, 0.01607493, 0.01696502, 0.01780507,
                    0.01800857, 0.019037, 0.01949349, 0.02046644, 0.0206756,
                    0.02175168, 0.02241395, 0.0232497, 0.02354939, 0.02405069,
                    0.02501095, 0.02539841, 0.02549399, 0.02679611, 0.02716407,
                    0.02783542, 0.02859171, 0.02907126, 0.02962453, 0.03026597,
                    0.03108157, 0.03187858, 0.03239498, 0.03295831, 0.03326473,
                    0.03433141],
                   [0., 0.00336105, 0.00702542, 0.00936218, 0.01124918,
                    0.01278707, 0.01427432, 0.01550372, 0.01669429, 0.01769135,
                    0.01889866, 0.01998421, 0.02095295, 0.02134471, 0.02272742,
                    0.02359614, 0.02464932, 0.02556556, 0.02697096, 0.02769694,
                    0.02873375, 0.02993691, 0.03096722, 0.03164894, 0.03320473,
                    0.03414116, 0.03513923, 0.03620067, 0.03703005, 0.03848555,
                    0.03997776, 0.04062479, 0.04170903, 0.04380891, 0.04449142,
                    0.04524801, 0.04686168, 0.04805262, 0.04900197, 0.04956221,
                    0.05110342]])
    u4 = np.array([[0., 0.00299187, 0.00564652, 0.00766154, 0.00899766,
                    0.01086988, 0.01295404, 0.01392214, 0.0156104, 0.01688021,
                    0.01826993, 0.01923553, 0.02083954, 0.0218393, 0.02279299,
                    0.0240834, 0.02524449, 0.02633193, 0.02788655, 0.02914818,
                    0.03003303, 0.03076753, 0.03204568, 0.03341971, 0.0340526,
                    0.03595448, 0.03652788, 0.03823819, 0.03967212, 0.04062831,
                    0.04157212, 0.0428609, 0.04365788, 0.044913, 0.04527096,
                    0.04730649, 0.04833768, 0.04970853, 0.05125308, 0.05185667,
                    0.05305843],
                   [0., 0.00164329, 0.00335756, 0.00456881, 0.00588376,
                    0.00696404, 0.00804094, 0.00931999, 0.0101686, 0.01128158,
                    0.01245819, 0.01348578, 0.01429449, 0.0150016, 0.01611414,
                    0.01724728, 0.01797072, 0.01942409, 0.0205775, 0.0211677,
                    0.02214802, 0.02359303, 0.02432193, 0.02604374, 0.02678142,
                    0.02824566, 0.02848257, 0.02996579, 0.0309389, 0.03240033,
                    0.03257107, 0.03450454, 0.03531074, 0.03639482, 0.03830676,
                    0.03898794, 0.04038775, 0.04202926, 0.04232813, 0.04376946,
                    0.04526533],
                   [0., 0.00195623, 0.00313435, 0.00430352, 0.00508364,
                    0.00631071, 0.00692016, 0.00803667, 0.00896746, 0.00951922,
                    0.01029418, 0.01122376, 0.01195596, 0.01301098, 0.01358393,
                    0.01426997, 0.01516676, 0.01629274, 0.01709336, 0.01775326,
                    0.01854061, 0.01940287, 0.02046429, 0.02129094, 0.02209033,
                    0.02315537, 0.02391866, 0.02459942, 0.02627035, 0.02749608,
                    0.02799257, 0.02934424, 0.03043411, 0.03097683, 0.03185161,
                    0.03251535, 0.03402147, 0.03545626, 0.0365558, 0.03805145,
                    0.03906855],
                   [0., 0.0021746, 0.00353874, 0.0046552, 0.00555986,
                    0.00653635, 0.00761999, 0.00826737, 0.00903378, 0.00993461,
                    0.010261, 0.01094116, 0.011698, 0.01192285, 0.01279554,
                    0.01317448, 0.01364013, 0.01429263, 0.01451903, 0.01511449,
                    0.01532313, 0.0160785, 0.01610688, 0.01663202, 0.01688395,
                    0.01749619, 0.01757378, 0.01802439, 0.01865731, 0.01898152,
                    0.01926489, 0.01933638, 0.0196417, 0.02008224, 0.02030172,
                    0.02047304, 0.02122644, 0.02135261, 0.02142715, 0.02237125,
                    0.02221484],
                   [0., 0.01732472, 0.02426673, 0.0286151, 0.03129383,
                    0.03362499, 0.03574057, 0.03727293, 0.03886267, 0.04007774,
                    0.04137585, 0.04290368, 0.04379427, 0.04490745, 0.04594037,
                    0.04651331, 0.04744643, 0.04916956, 0.04943996, 0.05033304,
                    0.05090518, 0.05225187, 0.0532764, 0.05370949, 0.0550821,
                    0.05613071, 0.0562659, 0.05761182, 0.05832129, 0.05930197,
                    0.0601629, 0.06084673, 0.06190002, 0.06294693, 0.06313865,
                    0.06396791, 0.06511249, 0.06589405, 0.06728554, 0.06774715,
                    0.06855804],
                   [0., 0.02313029, 0.03032454, 0.03490739, 0.03813865,
                    0.0405665, 0.04261851, 0.0445413, 0.04621839, 0.04767336,
                    0.04881545, 0.04991295, 0.0511499, 0.05226189, 0.05326962,
                    0.05386925, 0.05495153, 0.05601518, 0.05656711, 0.05764901,
                    0.05829754, 0.05935144, 0.05976864, 0.06071527, 0.06156729,
                    0.06214216, 0.06321068, 0.06351268, 0.06483901, 0.06502547,
                    0.06629314, 0.06682985, 0.06761676, 0.06802545, 0.06860748,
                    0.06954411, 0.07054443, 0.07116551, 0.07151624, 0.07286335,
                    0.07348336],
                   [0., 0.00623538, 0.00993809, 0.01238207, 0.01390676,
                    0.01575628, 0.01682851, 0.01811502, 0.01904157, 0.01959724,
                    0.0207932, 0.02152678, 0.02207561, 0.02331984, 0.02383459,
                    0.02463086, 0.025442, 0.02634532, 0.02694441, 0.02731631,
                    0.02744516, 0.02841034, 0.02881465, 0.02965037, 0.03080854,
                    0.03127242, 0.03172746, 0.03231916, 0.03303221, 0.03342555,
                    0.03373128, 0.0348793, 0.03550129, 0.0359825, 0.03732511,
                    0.03707814, 0.03750257, 0.03886392, 0.03918173, 0.04020236,
                    0.03985249],
                   [0., 0.00970621, 0.01303195, 0.01510036, 0.016829,
                    0.01836864, 0.01914594, 0.02037703, 0.02154067, 0.02232246,
                    0.02348166, 0.02448321, 0.02557887, 0.02701856, 0.02745,
                    0.02878297, 0.02957426, 0.03059549, 0.03225116, 0.03308672,
                    0.03322218, 0.0349125, 0.03564646, 0.03661597, 0.03806268,
                    0.03903763, 0.04009641, 0.04126003, 0.04217487, 0.04346801,
                    0.0452342, 0.04670526, 0.04723606, 0.04831737, 0.04940036,
                    0.05040349, 0.05205119, 0.0537115, 0.0546, 0.05549706,
                    0.05709012]])
    u5 = np.array([[0., 0.00354605, 0.00641502, 0.00902709, 0.01111765,
                    0.01290165, 0.01445205, 0.01565319, 0.01764713, 0.01872563,
                    0.02012479, 0.02118426, 0.02274831, 0.02407065, 0.02464102,
                    0.02585802, 0.0273357, 0.02888128, 0.02976888, 0.03089727,
                    0.0316335, 0.03315846, 0.03395349, 0.03524524, 0.03641384,
                    0.0384498, 0.03867669, 0.04036621, 0.04083845, 0.04199802,
                    0.04330614, 0.04503045, 0.04571009, 0.04668384, 0.048489,
                    0.04882283, 0.04987642, 0.05192447, 0.05262581, 0.05365192,
                    0.05424676],
                   [0., 0.00568969, 0.00806513, 0.0094273, 0.01067373,
                    0.01202865, 0.01297933, 0.01374903, 0.01499907, 0.0157926,
                    0.01671927, 0.01749251, 0.01837352, 0.01936557, 0.02009969,
                    0.02090668, 0.02216278, 0.02315338, 0.0238359, 0.02485679,
                    0.02550925, 0.0266067, 0.02775716, 0.0287001, 0.02963042,
                    0.03102693, 0.03223705, 0.03309253, 0.03470795, 0.03520903,
                    0.03613517, 0.03759352, 0.03844606, 0.04016956, 0.04102021,
                    0.04201013, 0.04286284, 0.04402149, 0.04534031, 0.04627435,
                    0.04784973],
                   [0., 0.00155291, 0.003, 0.0040832, 0.00519177,
                    0.00630471, 0.00712972, 0.00793885, 0.0090591, 0.01009493,
                    0.01063462, 0.01184199, 0.01251603, 0.01314171, 0.01422003,
                    0.01460924, 0.01590664, 0.01666344, 0.01755424, 0.01817345,
                    0.01936577, 0.02041186, 0.02093993, 0.02205261, 0.0228527,
                    0.02406599, 0.0249931, 0.02604072, 0.02680271, 0.02799266,
                    0.02885915, 0.03047001, 0.0310637, 0.03162753, 0.03345594,
                    0.03395548, 0.03502555, 0.03586228, 0.03765317, 0.03902283,
                    0.03946157],
                   [0., 0.00146087, 0.00294314, 0.00395593, 0.00534393,
                    0.00627078, 0.00702149, 0.00792084, 0.00860498, 0.0094903,
                    0.01037337, 0.01080641, 0.01113008, 0.01233803, 0.01274078,
                    0.01301592, 0.01352018, 0.01420004, 0.01441482, 0.01470987,
                    0.01577561, 0.01588495, 0.01579751, 0.01670791, 0.01710777,
                    0.01779289, 0.01776832, 0.0181922, 0.01879385, 0.01940854,
                    0.01932005, 0.02019403, 0.02007807, 0.02038137, 0.02082543,
                    0.02089632, 0.02161, 0.02146148, 0.0217766, 0.02237086,
                    0.02275099],
                   [0., 0.02018111, 0.0269432, 0.03082883, 0.03343219,
                    0.03585655, 0.03791821, 0.03954878, 0.04074643, 0.04211369,
                    0.04338633, 0.04451844, 0.04553696, 0.0467601, 0.04784031,
                    0.0487724, 0.04971502, 0.05055031, 0.05150602, 0.05267229,
                    0.0536437, 0.05433267, 0.05567003, 0.05578841, 0.05642481,
                    0.05803695, 0.05805357, 0.05930942, 0.06066082, 0.06092719,
                    0.06152486, 0.06273818, 0.06371157, 0.06424276, 0.06570193,
                    0.06575781, 0.0671785, 0.06768339, 0.06881301, 0.06929803,
                    0.06979846],
                   [0., 0.01910227, 0.0271765, 0.03192722, 0.03519914,
                    0.03814525, 0.04054042, 0.04231339, 0.04393665, 0.04561284,
                    0.0470659, 0.048786, 0.04991142, 0.05115235, 0.05226107,
                    0.05311122, 0.05412391, 0.05498471, 0.05628442, 0.05712792,
                    0.05799241, 0.05873256, 0.05965568, 0.06061292, 0.06143874,
                    0.06253082, 0.06301898, 0.063885, 0.06488644, 0.06527671,
                    0.06648461, 0.06696046, 0.06808834, 0.06906722, 0.06932177,
                    0.07004143, 0.07119462, 0.07103399, 0.0721715, 0.07319553,
                    0.07434806],
                   [0., 0.01087611, 0.01387919, 0.01548825, 0.01700026,
                    0.01823521, 0.01948419, 0.02045358, 0.02167235, 0.02230127,
                    0.02354183, 0.0240812, 0.02526564, 0.0256969, 0.02642544,
                    0.02715899, 0.02789689, 0.02860417, 0.02961441, 0.03004638,
                    0.03063597, 0.03140226, 0.03219211, 0.03247066, 0.03309092,
                    0.03388173, 0.0348307, 0.03503889, 0.03566248, 0.03631732,
                    0.03700422, 0.03773381, 0.0382796, 0.03861399, 0.0401057,
                    0.04064091, 0.0410691, 0.04111583, 0.04239887, 0.04276704,
                    0.04357626],
                   [0., 0.01296731, 0.01650029, 0.01888234, 0.02058507,
                    0.02195327, 0.02332989, 0.02431407, 0.02547793, 0.02648238,
                    0.02777087, 0.02871126, 0.02974498, 0.03045685, 0.03171766,
                    0.0322698, 0.03322524, 0.03493512, 0.03589817, 0.0365492,
                    0.03785283, 0.03852952, 0.03997539, 0.04087655, 0.04238689,
                    0.04358438, 0.04408245, 0.04613824, 0.04675192, 0.04853688,
                    0.0487029, 0.05041095, 0.05138137, 0.05264454, 0.05411054,
                    0.05466933, 0.05603835, 0.05710342, 0.05803283, 0.05924297,
                    0.06051867]])
    heurlist = [u1, u2, u3, u4, u5]
    heur_utillist = [np.array([0.01814256, 0.0392398, 0.04833426, 0.05598427, 0.06314142,
                               0.06976089, 0.07494779, 0.07871695, 0.08296518, 0.08628346,
                               0.09047718, 0.09309045, 0.09515336, 0.09841416, 0.10077472,
                               0.10376161, 0.10731877, 0.10940703, 0.11178777, 0.11369807,
                               0.11701572, 0.12044481, 0.12139491, 0.1238926, 0.12494185,
                               0.12580324, 0.12787467, 0.13054121, 0.13346172, 0.13423636,
                               0.13693912, 0.13996881, 0.14134538, 0.14150722, 0.14267676,
                               0.14398828, 0.14716366, 0.14733987, 0.14834699, 0.15000153]),
                     np.array([0.02329047, 0.03944378, 0.04663968, 0.05411333, 0.06144854,
                               0.06784217, 0.07244782, 0.07641531, 0.0805262, 0.08438177,
                               0.08747199, 0.09121573, 0.09377057, 0.09688332, 0.09925528,
                               0.10370465, 0.10570696, 0.10842042, 0.11174701, 0.11465724,
                               0.11651213, 0.11911722, 0.12087279, 0.12237647, 0.12439484,
                               0.12664851, 0.12827595, 0.13076115, 0.13263908, 0.13566195,
                               0.13667025, 0.13897559, 0.14039614, 0.14173603, 0.14428475,
                               0.14545878, 0.14614048, 0.14893251, 0.15048789, 0.15091192]),
                     np.array([0.01695832, 0.0382108, 0.04639173, 0.05446646, 0.06140647,
                               0.06859343, 0.07326426, 0.07669097, 0.08183018, 0.08481751,
                               0.08874762, 0.09183676, 0.09525752, 0.09865699, 0.10023648,
                               0.10363625, 0.10711158, 0.10994669, 0.11162405, 0.11549179,
                               0.1172276, 0.11967426, 0.12194647, 0.12369962, 0.12531369,
                               0.12769047, 0.12981065, 0.13218526, 0.13437024, 0.13536095,
                               0.13898425, 0.14071677, 0.14154419, 0.1443899, 0.14434034,
                               0.14540799, 0.14779547, 0.14848819, 0.14998711, 0.15122533]),
                     np.array([0.02124987, 0.04191757, 0.0500854, 0.0579956, 0.06561595,
                               0.0720533, 0.07712795, 0.08093198, 0.08512157, 0.08799387,
                               0.0924802, 0.09552747, 0.09858149, 0.10135565, 0.10374803,
                               0.10759322, 0.11045134, 0.11268488, 0.11487223, 0.11788325,
                               0.12171286, 0.12337377, 0.12512656, 0.12646354, 0.12951048,
                               0.13116879, 0.13308644, 0.13425978, 0.13778158, 0.14010257,
                               0.14086833, 0.14308825, 0.14393847, 0.14664934, 0.14804355,
                               0.14892363, 0.15055838, 0.15097085, 0.15409871, 0.15516216]),
                     np.array([0.02104651, 0.03919003, 0.04724104, 0.05494644, 0.06290493,
                               0.07031784, 0.07496161, 0.07956153, 0.08356809, 0.08697792,
                               0.0913731, 0.09417241, 0.09699998, 0.10092451, 0.10340344,
                               0.10711787, 0.1096748, 0.11151794, 0.11484896, 0.11765569,
                               0.12059403, 0.12342986, 0.12464453, 0.12662809, 0.12855005,
                               0.13135516, 0.13280989, 0.13443131, 0.13763338, 0.13927827,
                               0.1406853, 0.14289964, 0.14370643, 0.14554851, 0.14529921,
                               0.14823558, 0.15058108, 0.15022587, 0.15218993, 0.15362771])]
    rudi_utillist = [np.array([0.00592904, 0.00418302, 0.00634326, 0.00764617, 0.00908217,
                               0.010689, 0.01221645, 0.01332686, 0.01500069, 0.01600048,
                               0.01729633, 0.01896349, 0.02036388, 0.0217542, 0.02299005,
                               0.02435636, 0.02964544, 0.02805402, 0.02907389, 0.03460694,
                               0.03271205, 0.03740925, 0.03556156, 0.03769725, 0.03901393,
                               0.04095898, 0.04302215, 0.04503518, 0.04611126, 0.04827051,
                               0.05050588, 0.05263409, 0.0531344, 0.05549862, 0.05665526,
                               0.05885047, 0.06109977, 0.06215455, 0.06701077, 0.06552782]),
                     np.array([0.00694586, 0.00279358, 0.00416905, 0.00554578, 0.0063777,
                               0.00767235, 0.00891596, 0.01010921, 0.0111275, 0.01253385,
                               0.01389775, 0.01468506, 0.01629588, 0.01728261, 0.01909634,
                               0.01985745, 0.02552302, 0.02261479, 0.02417485, 0.02898049,
                               0.0269803, 0.03269441, 0.03029667, 0.03175096, 0.03301152,
                               0.03404298, 0.03664335, 0.03815271, 0.04021758, 0.04138837,
                               0.04302354, 0.04484734, 0.04622785, 0.0472827, 0.05015989,
                               0.05045147, 0.05252364, 0.05434396, 0.05974822, 0.05848724]),
                     np.array([0.00461086, 0.00384497, 0.00554389, 0.00705809, 0.00828575,
                               0.00921552, 0.01040475, 0.01199432, 0.01320823, 0.01464502,
                               0.01562525, 0.01688675, 0.0186291, 0.01977712, 0.02058289,
                               0.02217214, 0.02692035, 0.0254793, 0.02745764, 0.03254391,
                               0.03071939, 0.03548135, 0.03413378, 0.03548403, 0.03744606,
                               0.03772309, 0.04067221, 0.04187363, 0.04379567, 0.04494479,
                               0.04694609, 0.04927933, 0.05075456, 0.05206539, 0.05482818,
                               0.05537164, 0.05783975, 0.05873665, 0.06313941, 0.06187772]),
                     np.array([0.00763499, 0.00246421, 0.00378872, 0.00457872, 0.00614876,
                               0.00685967, 0.00817996, 0.00944489, 0.01063237, 0.0120274,
                               0.0131469, 0.01424402, 0.01587989, 0.01671808, 0.01753638,
                               0.01944326, 0.02466656, 0.02185225, 0.02321271, 0.0285509,
                               0.02673051, 0.0331634, 0.03017542, 0.03101633, 0.03231672,
                               0.03470929, 0.03543255, 0.03775326, 0.0390266, 0.0412813,
                               0.04286953, 0.04444233, 0.04649168, 0.04747674, 0.04924038,
                               0.0508214, 0.05290599, 0.05439339, 0.06031112, 0.0575003]),
                     np.array([0.00602332, 0.00451044, 0.00665905, 0.00835994, 0.0099727,
                               0.01124371, 0.01291957, 0.01463127, 0.01570543, 0.01723665,
                               0.01837455, 0.02034364, 0.02106383, 0.02223824, 0.02414705,
                               0.02564654, 0.03017238, 0.02890664, 0.03040098, 0.03463143,
                               0.03297644, 0.03834997, 0.03698688, 0.03730653, 0.03986618,
                               0.04110855, 0.04285007, 0.04518375, 0.04672349, 0.04833095,
                               0.0504122, 0.05267026, 0.05387019, 0.05561619, 0.05724393,
                               0.05909958, 0.06073187, 0.06206293, 0.06809557, 0.06513665])]

    # Size of dashes for unexplored nodes
    dshSz = 2
    # Size of figure layout
    figtup = (7, 5)
    titleSz, axSz, labelSz = 12, 10, 9
    xMax = 450

    avgHeurMat = np.average(np.array(heurlist), axis=0)

    # Plot of marginal utilities
    colors = cm.rainbow(np.linspace(0, 1., numTN))
    labels = [tnNames[ind] for ind in range(numTN)]

    x = range(testInt, testMax + 1, testInt)
    deltaArr = np.zeros((avgHeurMat.shape[0], avgHeurMat.shape[1] - 1))
    for rw in range(deltaArr.shape[0]):
        for col in range(deltaArr.shape[1]):
            deltaArr[rw, col] = avgHeurMat[rw, col + 1] - avgHeurMat[rw, col]
    yMax = np.max(deltaArr) * 1.1

    _ = plt.figure(figsize=figtup)
    for tnind in range(numTN):
        if tnind < 4:
            plt.plot(x, deltaArr[tnind], linewidth=2, color=colors[tnind],
                     label=labels[tnind], alpha=0.6)
        else:
            plt.plot(x, deltaArr[tnind], linewidth=2, color=colors[tnind],
                     label=labels[tnind], alpha=0.6, dashes=[1, dshSz])
    adj = 0.0002
    for tnind in range(numTN):
        if tnind == 0:
            plt.text(testInt * 1.1, deltaArr[tnind, 0], labels[tnind].ljust(15), fontsize=labelSz - 1)
        elif tnind == 1:
            plt.text(testInt * 1.1, deltaArr[tnind, 0] + 2 * adj, labels[tnind].ljust(15), fontsize=labelSz - 1)
        elif tnind == 3:
            plt.text(testInt * 1.1, deltaArr[tnind, 0] + adj, labels[tnind].ljust(15), fontsize=labelSz - 1)
        elif tnind == 2:
            plt.text(testInt * 1.1, deltaArr[tnind, 0] - adj, labels[tnind].ljust(15), fontsize=labelSz - 1)
        else:
            plt.text(testInt * 1.1, deltaArr[tnind, 0], labels[tnind].ljust(15), fontsize=labelSz - 1)
    plt.legend(fontsize=labelSz)
    plt.ylim([0., yMax])
    plt.xlim([0., xMax])
    plt.xlabel('Number of Tests', fontsize=axSz)
    plt.ylabel('Marginal Utility Gain', fontsize=axSz)
    plt.title('Marginal Utility with Increasing Tests\nAll-Provinces Setting with Market Term', fontsize=titleSz)
    plt.show()
    plt.close()

    # Allocation plot
    allocArr, objValArr = sampf.smooth_alloc_forward(avgHeurMat)
    # average distance from uniform allocation
    # np.linalg.norm(allocArr[:,-1]-np.ones((8))*4)

    colors = cm.rainbow(np.linspace(0, 1., numTN))
    labels = [tnNames[ind] for ind in range(numTN)]
    x = range(testInt, testMax + 1, testInt)
    _ = plt.figure(figsize=figtup)
    for tnind in range(numTN):
        if tnind < 4:
            plt.plot(x, allocArr[tnind] * testInt, linewidth=2, color=colors[tnind],
                     label=labels[tnind], alpha=0.6)
        else:
            plt.plot(x, allocArr[tnind] * testInt, linewidth=2, color=colors[tnind],
                     label=labels[tnind], alpha=0.6, dashes=[1, dshSz])
    # allocMax = allocArr.max() * testInt * 1.1
    allocMax = 185
    adj = 2.5
    for tnind in range(numTN):
        if tnind == 7:
            plt.text(testMax * 1.01, allocArr[tnind, -1] * testInt - adj, labels[tnind].ljust(15), fontsize=labelSz - 1)
        elif tnind == 6:
            plt.text(testMax * 1.01, allocArr[tnind, -1] * testInt + adj, labels[tnind].ljust(15), fontsize=labelSz - 1)
        elif tnind == 1:
            plt.text(testMax * 1.01, allocArr[tnind, -1] * testInt - adj, labels[tnind].ljust(15), fontsize=labelSz - 1)
        elif tnind == 3:
            plt.text(testMax * 1.01, allocArr[tnind, -1] * testInt + adj, labels[tnind].ljust(15), fontsize=labelSz - 1)
        else:
            plt.text(testMax * 1.01, allocArr[tnind, -1] * testInt, labels[tnind].ljust(15), fontsize=labelSz - 1)
    plt.legend(fontsize=labelSz)
    plt.ylim([0., allocMax])
    plt.xlim([0., xMax])
    plt.xlabel('Sampling Budget', fontsize=axSz)
    plt.ylabel('Test Node Allocation', fontsize=axSz)
    plt.title('Sampling Plan vs. Budget\nAll-Provinces Setting with Market Term', fontsize=titleSz)
    # plt.tight_layout()
    plt.show()
    plt.close()

    # Utility comparison plot
    colors = cm.rainbow(np.linspace(0, 0.8, 3))
    labels = ['Utility-Informed', 'Uniform', 'Fixed']
    x = range(testInt, testMax + 1, testInt)
    margUtilGroupList = [heur_utillist, unif_utillist, rudi_utillist]
    utilMax = -1
    for lst in margUtilGroupList:
        currMax = np.amax(np.array(lst))
        if currMax > utilMax:
            utilMax = currMax
    utilMax = utilMax * 1.1

    _ = plt.figure(figsize=figtup)
    for groupInd, margUtilGroup in enumerate(margUtilGroupList):
        groupArr = np.array(margUtilGroup)
        groupAvgArr = np.average(groupArr, axis=0)
        # Compile error bars
        stdevs = [np.std(groupArr[:, i]) for i in range(groupArr.shape[1])]
        group05Arr = [groupAvgArr[i] - (1.96 * stdevs[i] / np.sqrt(groupArr.shape[0])) for i in
                      range(groupArr.shape[1])]
        group95Arr = [groupAvgArr[i] + (1.96 * stdevs[i] / np.sqrt(groupArr.shape[0])) for i in
                      range(groupArr.shape[1])]
        plt.plot(x, groupAvgArr, color=colors[groupInd], linewidth=0.7, alpha=1., label=labels[groupInd] + ' 95% CI')
        plt.fill_between(x, groupAvgArr, group05Arr, color=colors[groupInd], alpha=0.2)
        plt.fill_between(x, groupAvgArr, group95Arr, color=colors[groupInd], alpha=0.2)
        # Line label
        plt.text(x[-1] * 1.01, groupAvgArr[-1], labels[groupInd].ljust(15), fontsize=labelSz - 1)
    plt.ylim(0, utilMax)
    # plt.xlim(0,x[-1]*1.12)
    plt.xlim([0., xMax])
    leg = plt.legend(loc='upper left', fontsize=labelSz)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.0)
    plt.xlabel('Sampling Budget', fontsize=axSz)
    plt.ylabel('Plan Utility', fontsize=axSz)
    plt.title('Utility from Utility-Informed, Uniform, and Fixed Allocations\nAll-Provinces Setting with Market Term',
              fontsize=titleSz)
    # Add text box showing budgetary savings
    compUtilAvg = np.average(np.array(heur_utillist), axis=0)
    x2, x3 = 134, 332
    plt.plot([100, x2], [compUtilAvg[9], compUtilAvg[9]], color='black', linestyle='--')
    iv = 0.003
    plt.plot([100, 100], [compUtilAvg[9] - iv, compUtilAvg[9] + iv], color='black', linestyle='--')
    plt.plot([x2, x2], [compUtilAvg[9] - iv, compUtilAvg[9] + iv], color='black', linestyle='--')
    # plt.plot([x3, x3], [compUtilAvg[9] - iv, compUtilAvg[9] + iv], color='black', linestyle='--')
    plt.text(110, compUtilAvg[9] + iv / 2, str(x2 - 100), fontsize=labelSz)
    # plt.text(205, compUtilAvg[9] + iv/2, str(x3-x2), fontsize=labelSz)

    # plt.tight_layout()
    plt.show()
    plt.close()

    '''
    Determining the budget saved for the sensitivity table
    currCompInd = 8
    compUtilAvg = np.average(np.array(compUtilList),axis=0) 
    evenUtilArr = np.array(evenUtilList)
    evenAvgArr = np.average(evenUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(evenAvgArr.tolist()) if val > compUtilAvg[currCompInd])
    evenSampSaved = round((compUtilAvg[currCompInd] - evenAvgArr[kInd - 1]) / (evenAvgArr[kInd] - evenAvgArr[kInd - 1]) * testInt) + (
                kInd - 1) * testInt - (currCompInd*testInt)
    print(evenSampSaved)
    rudiUtilArr = np.array(origUtilList)
    rudiAvgArr = np.average(rudiUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(rudiAvgArr.tolist()) if val > compUtilAvg[currCompInd])
    rudiSampSaved = round((compUtilAvg[currCompInd] - rudiAvgArr[kInd - 1]) / (rudiAvgArr[kInd] - rudiAvgArr[kInd - 1]) * testInt) + (
                kInd - 1) * testInt - (currCompInd*testInt)
    print(rudiSampSaved)
    currCompInd = 17
    compUtilAvg = np.average(np.array(compUtilList),axis=0) 
    evenUtilArr = np.array(evenUtilList)
    evenAvgArr = np.average(evenUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(evenAvgArr.tolist()) if val > compUtilAvg[currCompInd])
    evenSampSaved = round((compUtilAvg[currCompInd] - evenAvgArr[kInd - 1]) / (evenAvgArr[kInd] - evenAvgArr[kInd - 1]) * testInt) + (
                kInd - 1) * testInt - (currCompInd*testInt)
    print(evenSampSaved)
    rudiUtilArr = np.array(origUtilList)
    rudiAvgArr = np.average(rudiUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(rudiAvgArr.tolist()) if val > compUtilAvg[currCompInd])
    rudiSampSaved = round((compUtilAvg[currCompInd] - rudiAvgArr[kInd - 1]) / (rudiAvgArr[kInd] - rudiAvgArr[kInd - 1]) * testInt) + (
                kInd - 1) * testInt - (currCompInd*testInt)
    print(rudiSampSaved)
    '''

    return


def allocationsensitivityplots():
    alloc_list, util_avg_list, util_hi_list, util_lo_list = [], [], [], []
    for i in range(9):
        alloc_list.append(
            np.load(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'allocsens_alloc_' + str(i) + '.npy')))
        util_avg_list.append(np.load(
            os.path.join('casestudyoutputs', 'allocation_sensitivity', 'allocsens_util_avg_' + str(i) + '.npy')))
        util_hi_list.append(
            np.load(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'allocsens_util_hi_' + str(i) + '.npy')))
        util_lo_list.append(
            np.load(os.path.join('casestudyoutputs', 'allocation_sensitivity', 'allocsens_util_lo_' + str(i) + '.npy')))

    util_avg_list = np.array(util_avg_list)
    util_hi_list = np.array(util_hi_list)
    util_lo_list = np.array(util_lo_list)

    colors = ['g', 'r']
    dashes = [[5, 2], [2, 4]]
    x1 = range(0, 401, 10)
    yMax = 1.4
    for desind in range(util_avg_list.shape[0]):
        if desind == 0:
            plt.plot(x1, util_hi_list[desind], dashes=dashes[0],
                     linewidth=0.7, color=colors[0], label='Upper 95% CI')
            plt.plot(x1, util_lo_list[desind], dashes=dashes[1], label='Lower 95% CI',
                     linewidth=0.7, color=colors[1])
        else:
            plt.plot(x1, util_hi_list[desind], dashes=dashes[0],
                     linewidth=0.7, color=colors[0])
            plt.plot(x1, util_lo_list[desind], dashes=dashes[1],
                     linewidth=0.7, color=colors[1])
        # plt.fill_between(x1, margutilarr_lo[desind], margutilarr_hi[desind],
        #                color=colors[desind], alpha=0.3 * al)
    plt.legend()
    plt.ylim([0., yMax])
    plt.xlabel('Number of Tests')
    plt.ylabel('Utility')
    plt.title('Utility vs. Number of Tests\n10 Replications of Heuristic in Existing Setting')
    plt.show()
    plt.close()

    ### PLOT OF ALLOCATIONS
    testmax, testint = 400, 10
    alloc_list = np.array(alloc_list)

    colorsset = plt.get_cmap('Set1')
    colorinds = [6, 1, 2, 3]
    colors = np.array([colorsset(i) for i in colorinds])
    x1 = range(0, 401, 10)

    _ = plt.figure(figsize=(13, 8))
    labels = ['Moderate(39)', 'Moderate(17)', 'ModeratelyHigh(95)', 'ModeratelyHigh(26)']
    for allocarr_ind in range(alloc_list.shape[0]):
        allocarr = alloc_list[allocarr_ind]
        if allocarr_ind == 0:
            for tnind in range(allocarr.shape[0]):
                plt.plot(x1, allocarr[tnind] * testint,
                         linewidth=3, color=colors[tnind],
                         label=labels[tnind], alpha=0.2)
        else:
            for tnind in range(allocarr.shape[0]):
                plt.plot(x1, allocarr[tnind] * testint,
                         linewidth=3, color=colors[tnind], alpha=0.2)
    allocmax = 200
    plt.legend(fontsize=12)
    plt.ylim([0., allocmax])
    plt.xlabel('Number of Tests', fontsize=14)
    plt.ylabel('Test Node Allocation', fontsize=14)
    plt.title('Test Node Allocation\n10 Replications of Heuristic in Existing Setting',
              fontsize=18)
    plt.tight_layout()
    plt.show()
    plt.close()

    return