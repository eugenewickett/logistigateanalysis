# -*- coding: utf-8 -*-
"""
Christopher Lee

This python script is meant to evaluate diagnostic tools for their utility in a system. 

threshold - value from 0 to 1 measuring the expected level of substandard medicine that will lead to intervention
unitCost - array of the cost of each test
sensitivity - array of the sensitivities of the tests from 0 to 1
specificity - array of the specificities of the tests from 0 to 1
budget - the overall budget in dollars


"""
from importlib import reload
import numpy as np
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, 'logistigate')))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, 'logistigate','logistigate')))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, 'logistigate','logistigate','mcmcsamplers')))

import utilities as util # Pull from the submodule "develop" branch
import methods # Pull from the submodule "develop" branch
import lg # Pull from the submodule "develop" branch
reload(methods)
reload(util)
import statistics
import random
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings

'''
Example Code
runfile('C:/Users/chris/Dropbox/SFP_Research/DiagnosticTool/DiagnosticTool.py',
       wdir='C:/Users/chris/Dropbox/SFP_Research/DiagnosticTool')

# Real life example of Amoxicillin in Kenya
DiagnosticTool(['PAD', 'aPAD', 'Mobile Lab', 'HPLC'], .3, [3, 5, 60, 606], [.90, .94, .97, 1], [.9, .85, .97, 1], 10000, 10, 4)
'''


def DiagnosticTool(names, threshold, unit_cost, sensitivity, specificity, budget,
                   num_out, num_imp, setting_iterations=5, track_type='Tracked', SFP_rate_type='random'):  # todo: trans_mat_store = []):

    #start_time = time.time()
    num_diagnostics = len(unit_cost)  # Number of diagnostics to be evaluated

    inspection_credible = .9  # Size of the credible interval of the SFP rates


    # Store the accuracy numbers
    imp_high_count = [0] * num_diagnostics  # Counts the number of importers identified as above the threshold
    out_high_count = [0] * num_diagnostics  # Counts the number of outlets identified as above the threshold

    # Store the Type 1 Errors
    imp_type_1 = [0] * num_diagnostics  # Counts the number of importers INCORRECTLY identified as above the threshold
    out_type_1 = [0] * num_diagnostics  # Counts the number of outlets INCORRECTLY identified as above the threshold

    # Store the Type 2 Errors
    imp_type_2 = [0] * num_diagnostics  # Counts the number of importers NOT identified as above the threshold that are
    out_type_2 = [0] * num_diagnostics  # Counts the number of outlets NOT identified as above the threshold that are

    # Store the average interval size
    avg_imp_interval_size = [0] * num_diagnostics  # importer interval size
    avg_out_interval_size = [0] * num_diagnostics  # outlet interval size

    # h, w = setting_iterations, num_diagnostics         [[0 for x in range(h)] for y in range(w)]
    lower_bound_store_imp = np.zeros((1, num_diagnostics))
    upper_bound_store_imp = np.zeros((1, num_diagnostics))
    lower_bound_store_out = np.zeros((1, num_diagnostics))
    upper_bound_store_out = np.zeros((1, num_diagnostics))
    first_iter_imp = 1
    first_iter_out = 1
    setting_bound_iter_imp = 0
    setting_bound_iter_out = 0

    # Outer loop defining new settings for each diagnostic to be tested in
    for setting in range(setting_iterations):
        print("Percent Completed: ", round(100*setting/setting_iterations), "%")
        # predicted_rate = 0.1
        # todo: add scenarios where this is deterministic to show diagnostic efficacy in different situations

        # Case with random values for the importers and outlets. Sampled from a beta(2,9) distribution
        trans_mat_store = np.zeros(shape=(num_out, num_imp))

        if SFP_rate_type == 'random':
            true_rates_store = []

        # Case with one importer with high SFP rates
        elif SFP_rate_type == 'importer':
            true_rates_store = [0] * (num_out + num_imp)
            true_rates_store[0] = threshold + random.uniform(.1*(1-threshold), 0.9*(1-threshold))
            for imp_out in range(1, num_out + num_imp):
                true_rates_store[imp_out] = random.random() * threshold

        # Case with one outlet with high SFP rates
        elif SFP_rate_type == 'outlet':
            true_rates_store = [0] * (num_out + num_imp)
            true_rates_store[num_imp] = threshold+random.uniform(.1*(1-threshold), 0.9*(1-threshold))
            for imp_out_range in [range(num_imp), range(num_imp+1, num_out+num_imp)]:
                for imp_out in imp_out_range:
                    true_rates_store[imp_out] = random.random() * threshold

        # Case with one importer AND one outlet with high SFP rates
        elif SFP_rate_type == 'both':
            true_rates_store = [0] * (num_out + num_imp)
            true_rates_store[0] = threshold+random.uniform(.1*(1-threshold), 0.9*(1-threshold))
            true_rates_store[num_imp] = threshold+random.uniform(.1*(1-threshold), 0.9*(1-threshold))
            for imp_out_range in [range(1, num_imp), range(num_imp + 1, num_out + num_imp)]:
                for imp_out in imp_out_range:
                    true_rates_store[imp_out] = random.random() * threshold

        # Case for the Jupyter Notebook example
        elif SFP_rate_type == 'example':
            true_rates_store = [.05, .1, .5, .05, .05, .1, .1, 0, 0]
        # todo: Add functionality to calculate alpha and beta based on the expected mean and variance
        # alpha_val, beta_val = BetaVariable(predicted_rate, 0.0125)
        alpha_val = 2
        beta_val = 9

        # Run the function for each of the possible diagnostics
        diagnostic = 0
        for diagnostic in range(num_diagnostics):
            #now = time.time()

            # determine n based on the budget
            n = int(np.floor(budget / unit_cost[diagnostic]))

            # Generate random data to use with logistigate
            dataTblDict = generateRandDataDict(numImp=num_imp, numOut=num_out, diagSens=sensitivity[diagnostic],
                                               diagSpec=specificity[diagnostic], numSamples=n,
                                               dataType=track_type, transMatLambda=1.1,
                                               transMat=trans_mat_store,
                                               randSeed=-1, trueRates=true_rates_store, alpha=alpha_val, beta=beta_val)

            # Store the true rates and transMat so they are the same for each iteration
            if diagnostic == 0:
                true_rates_store = dataTblDict['trueRates']
                trans_mat_store = dataTblDict['transMat']

            dataTblDict.update({'diagSens': sensitivity[diagnostic], 'diagSpec': specificity[diagnostic],
                                'numPostSamples': 100, 'prior': methods.prior_normal(),
                                'Madapt': 100})

            # Add a row for outlets to check the upper and lower bounds of the bad actors
            row_iter_out = 0
            bad_outlets = sum([1 if i > threshold else 0 for i in true_rates_store[num_imp:]])
            while row_iter_out < bad_outlets and diagnostic == 0:
                if first_iter_out == 1:
                    first_iter_out = 0
                    row_iter_out += 1
                else:
                    lower_bound_store_out = np.vstack([lower_bound_store_out, np.zeros((1, num_diagnostics))])
                    upper_bound_store_out = np.vstack([upper_bound_store_out, np.zeros((1, num_diagnostics))])
                    row_iter_out += 1
            # Do the same as above for importers
            row_iter_imp = 0
            bad_importers = sum([1 if i > threshold else 0 for i in true_rates_store[:num_imp]])
            print("diag ", diagnostic)
            print("importers ", bad_importers)
            while row_iter_imp < bad_importers and diagnostic == 0:
                if first_iter_imp == 1:
                    first_iter_imp = 0
                    row_iter_imp += 1
                else:
                    lower_bound_store_imp = np.vstack([lower_bound_store_imp, np.zeros((1, num_diagnostics))])
                    upper_bound_store_imp = np.vstack([upper_bound_store_imp, np.zeros((1, num_diagnostics))])
                    row_iter_imp += 1



            # Run Logistigate
            logistigateDict = lg.runLogistigate(dataTblDict)

            # Evaluate the importer data
            bad_imp_index = 0
            for importers in range(num_imp):
                lower_bound = np.quantile(logistigateDict['postSamples'][:, importers],
                                          (1-inspection_credible)/2)

                upper_bound = np.quantile(logistigateDict['postSamples'][:, importers],
                                          (1+inspection_credible)/2)
                if true_rates_store[importers] > threshold:
                    lower_bound_store_imp[setting_bound_iter_imp+bad_imp_index][diagnostic] \
                        = lower_bound - true_rates_store[importers]
                    upper_bound_store_imp[setting_bound_iter_imp+bad_imp_index][diagnostic] \
                        = upper_bound - true_rates_store[importers]
                    bad_imp_index += 1

                # Calculate the average interval size across all outlets
                avg_imp_interval_size[diagnostic] += (upper_bound - lower_bound)/(setting_iterations*num_out)
                # Check if we are confident the importer is above the threshold
                if threshold < lower_bound:
                    imp_high_count[diagnostic] += 1
                    # Check for a type 1 error
                    if true_rates_store[importers] < threshold:  # Check if incorrect
                        imp_type_1[diagnostic] += 1
                # Check for a type 2 error
                elif true_rates_store[importers] > threshold:
                    imp_type_2[diagnostic] += 1

                #print(names[diagnostic])
                #print(time.time()-now)

            # Evaluate the outlet data
            bad_out_index = 0
            for outlets in range(num_out):
                lower_bound = np.quantile(logistigateDict['postSamples'][:, num_imp + outlets],
                                          (1-inspection_credible)/2)
                upper_bound = np.quantile(logistigateDict['postSamples'][:, num_imp + outlets],
                                          (1+inspection_credible)/2)

                # Calculate the average interval size across all outlets
                avg_out_interval_size[diagnostic] += (upper_bound - lower_bound)/(setting_iterations*num_out)

                if true_rates_store[outlets] > threshold:
                    lower_bound_store_out[setting_bound_iter_out+bad_out_index][diagnostic] \
                        = lower_bound - true_rates_store[outlets]
                    upper_bound_store_out[setting_bound_iter_out+bad_out_index][diagnostic] \
                        = upper_bound - true_rates_store[outlets]
                    bad_out_index += 1

                # Check if we are confident the outlet is above the threshold
                if threshold < lower_bound:
                    out_high_count[diagnostic] += 1
                    # Check for a type 1 error
                    if true_rates_store[num_imp + outlets] < threshold:  # Check if correct
                        out_type_1[diagnostic] += 1

                # Check for a type 2 error
                elif true_rates_store[num_imp + outlets] > threshold:
                    out_type_2[diagnostic] += 1

        setting_bound_iter_imp += bad_importers
        setting_bound_iter_out += bad_outlets



                # Uncomment to print a summary of each run of logistigate
                # util.printEstimates(logistigateDict)
                # util.plotPostSamples(logistigateDict)




    avg_imp_high = [i / setting_iterations for i in imp_high_count]
    avg_imp_type_1 = [i / setting_iterations for i in imp_type_1]
    avg_imp_type_2 = [i / setting_iterations for i in imp_type_2]

    avg_out_high = [i / setting_iterations for i in out_high_count]
    avg_out_type_1 = [i / setting_iterations for i in out_type_1]
    avg_out_type_2 = [i / setting_iterations for i in out_type_2]

    '''
    print("Importers")
    print(avg_imp_high)
    print(avg_imp_type_1)
    print(avg_imp_type_2)
    print("Outlets")
    print(avg_out_high)
    print(avg_out_type_1)
    print(avg_out_type_2)
    '''

    print(avg_imp_high)
    print(avg_imp_interval_size)

    DiagnosticToolDict = {
        'avg_out_interval_size': avg_out_interval_size,
        'avg_imp_interval_size': avg_imp_interval_size,
        'avg_imp_detected': avg_imp_high,
        'avg_out_detected': avg_out_high,
        'avg_imp_type_1': avg_imp_type_1,
        'avg_out_type_1': avg_out_type_1,
        'avg_imp_type_2': avg_imp_type_2,
        'avg_out_type_2': avg_out_type_2,
        'setting_iterations': setting_iterations,
        'names': names,
        'SFP_rate_type': SFP_rate_type,
        'num_imp': num_imp,
        'num_out': num_out,
        'threshold': threshold,
    }

    if SFP_rate_type == 'example':
        DiagnosticToolDict.update({'lower_bound_store_imp': lower_bound_store_imp, 'upper_bound_store_imp': upper_bound_store_imp })

    return DiagnosticToolDict


def plotLowerBounds(DiagnosticToolDict, size="default"):
    """
    plotLowerBounds plots the lower bounds of importers in the example in a histogram. This demonstrates the ability of
    a diagnostic to detect a bad importer based on a certain threshold.
    """

    # Initialize variables from input dictionary
    threshold = DiagnosticToolDict['threshold']
    names = DiagnosticToolDict['names']

    # Define subplot figure
    if size == "default":
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    elif size == "example":
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))
    else:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(size[0], size[1]))

    axs_mat = [ax1, ax2, ax3, ax4]  # Iterable axes

    # Bin sizes for histograms
    bin_size = 0.025
    num_bins = round(1/bin_size)

    # For loop across all diagnostics
    for i in range(len(names)):
        N, bins, patches = axs_mat[i].hist(DiagnosticToolDict['lower_bound_store_imp'][i],
                                           alpha=0.7, bins=np.linspace(0, 1, num_bins+1))
        axs_mat[i].set_xlim([0, 1])
        axs_mat[i].axvline(x=threshold, color='b')  # Draw vertical line at threshold
        title_text = names[i]  # Set title
        axs_mat[i].set_title(title_text)

        # Recolor bins based on detection of bad importer
        for bars in range(round(threshold*num_bins)):
            patches[bars].set_facecolor('r')
        for bars in range(round(threshold*num_bins), num_bins):
            patches[bars].set_facecolor('g')

        # Display detection rate
        text_string = str(np.round(100*DiagnosticToolDict['avg_imp_detected'][i], decimals=1)) + "%"
        props = dict(boxstyle='round', facecolor='yellow', alpha=0.5)
        axs_mat[i].text(0.05, 0.95, text_string, transform=axs_mat[i].transAxes,
                     verticalalignment='top', bbox=props)

        # Create custom legend
        if i == 1:
            custom_lines = [Line2D([0], [0], color='b', lw=4),
                            Line2D([0], [0], color='g', lw=4),
                            Line2D([0], [0], color='r', lw=4),
                            Line2D([0], [0], color='yellow', lw=4)]
            axs_mat[i].legend(custom_lines, ['Threshold', 'Detected', 'Undetected', 'Detection Rate'], bbox_to_anchor=(1.05, .8, 0.3, 0.2), loc='upper left')
        plt.tight_layout()

    return


def plotCredibleIntervals(DiagnosticToolDict, size="default"):
    """
    The purpose of this function is to graph the middle interval for the example. It will display the average interval
    for the bad importer for each diagnostic
    """

    numImp, numOut = DiagnosticToolDict['num_imp'], DiagnosticToolDict['num_out']

    # Declare the figure

    if size == "default":
        fig = plt.figure()
    elif size == "example":
        fig = plt.figure(figsize=(10, 6))
    else:
        fig = plt.figure(figsize=(size[0], size[1]))

    num_vals = len(DiagnosticToolDict['names'])
    if DiagnosticToolDict['SFP_rate_type'] == 'example':
        adjust = 50
    else:
        adjust = 0

    for i in range(num_vals):
        # Average lower and upper bounds
        average_lb = np.average(DiagnosticToolDict['lower_bound_store_imp'][:, i])*100 + adjust
        average_ub = np.average(DiagnosticToolDict['upper_bound_store_imp'][:, i])*100 + adjust
        print(average_ub - average_lb)
        plt.vlines(x=i, ymin=average_lb, ymax=average_ub, color='r', lw=1)
        plt.scatter([i, i], [average_lb, average_ub], color='r')
        actual_value = plt.scatter(i, 50, color='b', zorder=10, s=25)

    plt.title("Average 90% Interval")
    plt.legend([actual_value], ["Actual SFP Rate"])
    plt.ylabel("SFP Rate (%)")
    plt.xticks(range(num_vals), DiagnosticToolDict['names'])
    plt.hlines(50, 0, 3, color='b', linestyles='dashed')
    return

def BetaVariable(mu=.5, v=0.025):
    """
    This function defines alpha and beta for a beta distribution based on
    a given mean and variance
    """
    a = ((1 - mu) / v - 1 / mu) * mu ** 2
    b = a * (1 / mu - 1)
    return a, b

def generateRandDataDict(numImp=5,
                         numOut=50,
                         diagSens=0.90,
                         diagSpec=0.99,
                         numSamples=50 * 20,
                         dataType='Tracked',
                         transMatLambda=1.1,
                         transMat=np.zeros(shape=(50, 5)),
                         randSeed=-1,
                         trueRates=None,
                         alpha=2,
                         beta=9):
    """
    Randomly generates an example input data dicionary for the entered inputs.
    SFP rates are generated according to a beta(2,9) distribution, while
    transition rates are distributed according to a scaled Pareto(1.1) distribution
    by default.

    INPUTS
    ------
    Takes the following arguments:
        numImp, numOut: integer
            Number of importers and outlets
        diagSens, diagSpec: float
            Diagnostic sensitivity, specificity
        numSamples: integer
            Total number of data points to generate
        dataType: string
            'Tracked' or 'Untracked'

    OUTPUTS
    -------
    Returns dataTblDict dictionary with the following keys:
        dataTbl: list
            If Tracked, each list entry should have three elements, as follows:
                Element 1: string; Name of outlet/lower echelon entity
                Element 2: string; Name of importer/upper echelon entity
                Element 3: integer; 0 or 1, where 1 signifies aberration detection
            If Untracked, each list entry should have two elements, as follows:
                Element 1: string; Name of outlet/lower echelon entity
                Element 2: integer; 0 or 1, where 1 signifies aberration detection
        outletNames/importerNames: list of strings
        transMat: Numpy matrix
            Matrix of transition probabilities between importers and outlets
        diagSens, diagSpec, type
            From inputs, where 'type' = 'dataType'
    """
    if trueRates is None:
        trueRates = []
    dataTblDict = {}

    impNames = ['Importer ' + str(i + 1) for i in range(numImp)]
    outNames = ['Outlet ' + str(i + 1) for i in range(numOut)]

    # Generate random true SFP rates
    if len(trueRates) == 0:
        trueRates = np.zeros(numImp + numOut)  # importers first, outlets second
        if randSeed >= 0:
            random.seed(randSeed)
        trueRates[:numImp] = [random.betavariate(alpha, beta) for i in range(numImp)]
        trueRates[numImp:] = [random.betavariate(alpha, beta) for i in range(numOut)]

    # Generate random transition matrix
    if np.sum(transMat) == 0.0:
        if randSeed >= 0:
            random.seed(randSeed + 1)
        for outInd in range(numOut):
            rowRands = [random.paretovariate(transMatLambda) for i in range(numImp)]
            if numImp > 10:  # Only keep 10 randomly chosen importers, if numImp > 10
                rowRands[10:] = [0.0 for i in range(numImp - 10)]
                random.shuffle(rowRands)

            normalizedRands = [rowRands[i] / sum(rowRands) for i in range(numImp)]
            # only keep transition probabilities above 2%
            # normalizedRands = [normalizedRands[i] if normalizedRands[i]>0.02 else 0.0 for i in range(numImp)]

            # normalizedRands = [normalizedRands[i] / sum(normalizedRands) for i in range(numImp)]
            transMat[outInd, :] = normalizedRands

    # np.linalg.det(transMat.T @ transMat) / numOut
    # 1.297 for n=50

    # Generate testing data
    testingDataList = []
    if dataType == 'Tracked':
        if randSeed >= 0:
            random.seed(randSeed + 2)
        for currSamp in range(numSamples):
            currOutlet = random.sample(outNames, 1)[0]
            currImporter = random.choices(impNames, weights=transMat[outNames.index(currOutlet)], k=1)[0]
            currOutRate = trueRates[numImp + outNames.index(currOutlet)]
            currImpRate = trueRates[impNames.index(currImporter)]
            realRate = currOutRate + currImpRate - currOutRate * currImpRate
            realResult = np.random.binomial(1, p=realRate)
            if realResult == 1:
                result = np.random.binomial(1, p=diagSens)
            if realResult == 0:
                result = np.random.binomial(1, p=1 - diagSpec)
            testingDataList.append([currOutlet, currImporter, result])
    elif dataType == 'Untracked':
        if randSeed >= 0:
            random.seed(randSeed + 3)
        for currSamp in range(numSamples):
            currOutlet = random.sample(outNames, 1)[0]
            currImporter = random.choices(impNames, weights=transMat[outNames.index(currOutlet)], k=1)[0]
            currOutRate = trueRates[numImp + outNames.index(currOutlet)]
            currImpRate = trueRates[impNames.index(currImporter)]
            realRate = currOutRate + currImpRate - currOutRate * currImpRate
            realResult = np.random.binomial(1, p=realRate)
            if realResult == 1:
                result = np.random.binomial(1, p=diagSens)
            if realResult == 0:
                result = np.random.binomial(1, p=1 - diagSpec)
            testingDataList.append([currOutlet, result])

    dataTblDict.update({'outletNames': outNames, 'importerNames': impNames,
                        'diagSens': diagSens, 'diagSpec': diagSpec, 'type': dataType,
                        'dataTbl': testingDataList, 'transMat': transMat,
                        'trueRates': trueRates})
    return dataTblDict
