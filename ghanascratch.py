
from logistigate.logistigate import utilities as util # Pull from the submodule "develop" branch
from logistigate.logistigate import methods
from logistigate.logistigate import lg
import numpy as np
from numpy.random import choice
import scipy.special as sps
import scipy.stats as spstat
import matplotlib
import matplotlib.pyplot as plt
import random
import os
import pickle
import pandas as pd

def GhanaInference():
    '''Script for analyzing 2022 Ghana data'''

    SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
    filesPath = os.path.join(SCRIPT_DIR, 'uspfiles')
    GHA_df1 = pd.read_csv(os.path.join(filesPath, 'FACILID_MNFR.csv'), low_memory=False) # Facilities as test nodes
    GHAlist_FCLY = GHA_df1.values.tolist()
    GHA_df2 = pd.read_csv(os.path.join(filesPath, 'CITY_MNFR.csv'), low_memory=False) # Cities as test nodes
    GHAlist_CITY = GHA_df2.values.tolist()
    GHA_df3= pd.read_csv(os.path.join(filesPath, 'PROV_MNFR.csv'), low_memory=False) # Provinces as test nodes
    GHAlist_PROV = GHA_df3.values.tolist()

    '''
    283 observed FACILITIES
    34 observed CITIES
    5 observed PROVINCES
    41 observed MANUFACTURERS
    16.7% SFP rate (63 of 377 tests)
    '''
    # Set MCMC parameters
    numPostSamps = 1000
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    # Establish a prior
    priorMean = -2.5
    priorVar = 3.5

    # Create a logistigate dictionary and conduct inference
    # FACILITIES
    lgDict = util.testresultsfiletotable(GHAlist_FCLY, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum())) # Check that everything is in line with expectations
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nGhana - Facility ID/Manufacturer Analysis', '\nGhana - Facility ID/Manufacturer Analysis'])
    # Now print 90% intervals and write intervals to file
    # Supply nodes first
    import csv
    outputFileName = os.path.join(filesPath, 'SNintervals_FCLY.csv')
    f = open(outputFileName, 'w', newline='')
    writer = csv.writer(f)


    writelist = []
    for i in range(lgDict['N'].shape[1]):
        interval = [np.quantile(lgDict['postSamples'][:,i], 0.05),np.quantile(lgDict['postSamples'][:,i], 0.95)]
        name = lgDict['importerNames'][i]
        writelist.append([name, interval[0], interval[1]])
        writer.writerow([name, interval[0], interval[1]])
        print(name + ': ' + str(interval))
    writelist = []
    numSN = lgDict['N'].shape[1]
    for i in range(lgDict['N'].shape[0]):
        interval = [np.quantile(lgDict['postSamples'][:,numSN + i], 0.05), np.quantile(lgDict['postSamples'][:,numSN + i], 0.95)]
        name = lgDict['outletNames'][i]
        writelist.append([name, interval[0], interval[1]])
        writer.writerow([name, interval[0], interval[1]])
        print(name+': '+str(interval))

    f.close()

    # CITIES
    lgDict = util.testresultsfiletotable(GHAlist_CITY, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))  # Check that everything is in line with expectations
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nGhana - City/Manufacturer Analysis', '\nGhana - City/Manufacturer Analysis'])
    # Now print 90% intervals
    import csv
    outputFileName = os.path.join(filesPath, 'SNintervals_CITY.csv')
    f = open(outputFileName, 'w', newline='')
    writer = csv.writer(f)

    # Supply nodes first
    writelist = []
    for i in range(lgDict['N'].shape[1]):
        interval = [np.quantile(lgDict['postSamples'][:, i], 0.05), np.quantile(lgDict['postSamples'][:, i], 0.95)]
        name = lgDict['importerNames'][i]
        writelist.append([name, interval[0], interval[1]])
        writer.writerow([name, interval[0], interval[1]])
        print(name + ': ' + str(interval))
    # outputFileName = os.path.join(filesPath, 'SNintervals_FCLY.csv')
    writelist = []
    numSN = lgDict['N'].shape[1]
    for i in range(lgDict['N'].shape[0]):
        interval = [np.quantile(lgDict['postSamples'][:, numSN + i], 0.05),
                    np.quantile(lgDict['postSamples'][:, numSN + i], 0.95)]
        name = lgDict['outletNames'][i]
        writelist.append([name, interval[0], interval[1]])
        writer.writerow([name, interval[0], interval[1]])
        print(name + ': ' + str(interval))

    f.close()

    # PROVINCES
    lgDict = util.testresultsfiletotable(GHAlist_PROV, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))  # Check that everything is in line with expectations
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nGhana - Province/Manufacturer Analysis', '\nGhana - Province/Manufacturer Analysis'])
    # Now print 90% intervals
    import csv
    outputFileName = os.path.join(filesPath, 'SNintervals_PROV.csv')
    f = open(outputFileName, 'w', newline='')
    writer = csv.writer(f)

    # Supply nodes first
    writelist = []
    for i in range(lgDict['N'].shape[1]):
        interval = [np.quantile(lgDict['postSamples'][:, i], 0.05), np.quantile(lgDict['postSamples'][:, i], 0.95)]
        name = lgDict['importerNames'][i]
        writelist.append([name, interval[0], interval[1]])
        writer.writerow([name, interval[0], interval[1]])
        print(name + ': ' + str(interval))
    # outputFileName = os.path.join(filesPath, 'SNintervals_FCLY.csv')
    writelist = []
    numSN = lgDict['N'].shape[1]
    for i in range(lgDict['N'].shape[0]):
        interval = [np.quantile(lgDict['postSamples'][:, numSN + i], 0.05),
                    np.quantile(lgDict['postSamples'][:, numSN + i], 0.95)]
        name = lgDict['outletNames'][i]
        writelist.append([name, interval[0], interval[1]])
        writer.writerow([name, interval[0], interval[1]])
        print(name + ': ' + str(interval))
        ###### WRITE TO CSV
    f.close()

    return

def GhanaSamplingDesignAnalysis():
    '''Script for analyzing 2022 Ghana data'''

    SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
    filesPath = os.path.join(SCRIPT_DIR, 'uspfiles')
    GHA_df1 = pd.read_csv(os.path.join(filesPath, 'FACILID_MNFR.csv'), low_memory=False) # Facilities as test nodes
    GHAlist_FCLY = GHA_df1.values.tolist()
    GHA_df2 = pd.read_csv(os.path.join(filesPath, 'CITY_MNFR.csv'), low_memory=False) # Cities as test nodes
    GHAlist_CITY = GHA_df2.values.tolist()
    GHA_df3= pd.read_csv(os.path.join(filesPath, 'PROV_MNFR.csv'), low_memory=False) # Provinces as test nodes
    GHAlist_PROV = GHA_df3.values.tolist()

    '''
    283 observed FACILITIES
    34 observed CITIES
    5 observed PROVINCES
    41 observed MANUFACTURERS
    16.7% SFP rate (63 of 377 tests)
    '''
    # Set MCMC parameters
    numPostSamps = 1000
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    # Establish a prior
    priorMean = -2.5
    priorVar = 3.5

    # Create a logistigate dictionary and conduct inference
    # FACILITIES
    lgDict = util.testresultsfiletotable(GHAlist_FCLY, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum())) # Check that everything is in line with expectations
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nGhana - Facility ID/Manufacturer Analysis', '\nGhana - Facility ID/Manufacturer Analysis'])
    # Now print 90% intervals and write intervals to file
    # Supply nodes first
    import csv
    outputFileName = os.path.join(filesPath, 'SNintervals_FCLY.csv')
    f = open(outputFileName, 'w', newline='')
    writer = csv.writer(f)


    writelist = []
    for i in range(lgDict['N'].shape[1]):
        interval = [np.quantile(lgDict['postSamples'][:,i], 0.05),np.quantile(lgDict['postSamples'][:,i], 0.95)]
        name = lgDict['importerNames'][i]
        writelist.append([name, interval[0], interval[1]])
        writer.writerow([name, interval[0], interval[1]])
        print(name + ': ' + str(interval))
    writelist = []
    numSN = lgDict['N'].shape[1]
    for i in range(lgDict['N'].shape[0]):
        interval = [np.quantile(lgDict['postSamples'][:,numSN + i], 0.05), np.quantile(lgDict['postSamples'][:,numSN + i], 0.95)]
        name = lgDict['outletNames'][i]
        writelist.append([name, interval[0], interval[1]])
        writer.writerow([name, interval[0], interval[1]])
        print(name+': '+str(interval))

    f.close()

    # CITIES
    lgDict = util.testresultsfiletotable(GHAlist_CITY, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))  # Check that everything is in line with expectations
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nGhana - City/Manufacturer Analysis', '\nGhana - City/Manufacturer Analysis'])
    # Now print 90% intervals
    import csv
    outputFileName = os.path.join(filesPath, 'SNintervals_CITY.csv')
    f = open(outputFileName, 'w', newline='')
    writer = csv.writer(f)

    # Supply nodes first
    writelist = []
    for i in range(lgDict['N'].shape[1]):
        interval = [np.quantile(lgDict['postSamples'][:, i], 0.05), np.quantile(lgDict['postSamples'][:, i], 0.95)]
        name = lgDict['importerNames'][i]
        writelist.append([name, interval[0], interval[1]])
        writer.writerow([name, interval[0], interval[1]])
        print(name + ': ' + str(interval))
    # outputFileName = os.path.join(filesPath, 'SNintervals_FCLY.csv')
    writelist = []
    numSN = lgDict['N'].shape[1]
    for i in range(lgDict['N'].shape[0]):
        interval = [np.quantile(lgDict['postSamples'][:, numSN + i], 0.05),
                    np.quantile(lgDict['postSamples'][:, numSN + i], 0.95)]
        name = lgDict['outletNames'][i]
        writelist.append([name, interval[0], interval[1]])
        writer.writerow([name, interval[0], interval[1]])
        print(name + ': ' + str(interval))

    f.close()

    # PROVINCES
    lgDict = util.testresultsfiletotable(GHAlist_PROV, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))  # Check that everything is in line with expectations
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nGhana - Province/Manufacturer Analysis', '\nGhana - Province/Manufacturer Analysis'])
    # Now print 90% intervals
    import csv
    outputFileName = os.path.join(filesPath, 'SNintervals_PROV.csv')
    f = open(outputFileName, 'w', newline='')
    writer = csv.writer(f)

    # Supply nodes first
    writelist = []
    for i in range(lgDict['N'].shape[1]):
        interval = [np.quantile(lgDict['postSamples'][:, i], 0.05), np.quantile(lgDict['postSamples'][:, i], 0.95)]
        name = lgDict['importerNames'][i]
        writelist.append([name, interval[0], interval[1]])
        writer.writerow([name, interval[0], interval[1]])
        print(name + ': ' + str(interval))
    # outputFileName = os.path.join(filesPath, 'SNintervals_FCLY.csv')
    writelist = []
    numSN = lgDict['N'].shape[1]
    for i in range(lgDict['N'].shape[0]):
        interval = [np.quantile(lgDict['postSamples'][:, numSN + i], 0.05),
                    np.quantile(lgDict['postSamples'][:, numSN + i], 0.95)]
        name = lgDict['outletNames'][i]
        writelist.append([name, interval[0], interval[1]])
        writer.writerow([name, interval[0], interval[1]])
        print(name + ': ' + str(interval))
        ###### WRITE TO CSV
    f.close()

    return