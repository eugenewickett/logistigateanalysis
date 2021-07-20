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

def cleanMQD():
    '''
    Script that cleans up raw Medicines Quality Database data for use in logistigate.
    It reads in a CSV file with columns 'Country,' 'Province,' 'Manufacturer,' and
    'Final Test Result,' and returns a dictionary for use with logistigate.
    '''



    return

def MQDdataScript():
    '''Script looking at the MQD data'''
    import scipy.special as sps
    import numpy as np
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}

    # Run with Country as outlets
    dataTblDict = util.testresultsfiletotable('../examples/data/MQD_TRIMMED1.csv')
    dataTblDict.update({'diagSens': 1.0,
                        'diagSpec': 1.0,
                        'numPostSamples': 500,
                        'prior': methods.prior_normal(mu=sps.logit(0.038)),
                        'MCMCdict': MCMCdict})
    logistigateDict = lg.runlogistigate(dataTblDict)

    util.plotPostSamples(logistigateDict)
    util.printEstimates(logistigateDict)

    # Run with Country-Province as outlets
    dataTblDict2 = util.testresultsfiletotable('../examples/data/MQD_TRIMMED2.csv')
    dataTblDict2.update({'diagSens': 1.0,
                        'diagSpec': 1.0,
                        'numPostSamples': 500,
                        'prior': methods.prior_normal(mu=sps.logit(0.038)),
                        'MCMCdict': MCMCdict})
    logistigateDict2 = lg.runlogistigate(dataTblDict2)

    util.plotPostSamples(logistigateDict2)
    util.printEstimates(logistigateDict2)

    # Run with Cambodia provinces
    dataTblDict_CAM = util.testresultsfiletotable('../examples/data/MQD_CAMBODIA.csv')
    countryMean = np.sum(dataTblDict_CAM['Y']) / np.sum(dataTblDict_CAM['N'])
    dataTblDict_CAM.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 1000,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_CAM = lg.runlogistigate(dataTblDict_CAM)
    numCamImps_fourth = int(np.floor(logistigateDict_CAM['importerNum'] / 4))
    util.plotPostSamples(logistigateDict_CAM, plotType='int90',
                         importerIndsSubset=np.arange(numCamImps_fourth).tolist(),
                         subTitleStr=['\nCambodia - 1st Quarter', '\nCambodia'])
    util.plotPostSamples(logistigateDict_CAM, plotType='int90',
                         importerIndsSubset=np.arange(numCamImps_fourth,numCamImps_fourth*2).tolist(),
                         subTitleStr=['\nCambodia - 2nd Quarter', '\nCambodia'])
    util.plotPostSamples(logistigateDict_CAM, plotType='int90',
                         importerIndsSubset=np.arange(numCamImps_fourth * 2, numCamImps_fourth * 3).tolist(),
                         subTitleStr=['\nCambodia - 3rd Quarter', '\nCambodia'])
    util.plotPostSamples(logistigateDict_CAM, plotType='int90',
                         importerIndsSubset=np.arange(numCamImps_fourth * 3, numCamImps_fourth * 4).tolist(),
                         subTitleStr=['\nCambodia - 4th Quarter', '\nCambodia'])
    util.printEstimates(logistigateDict_CAM)
    # Plot importers subset where median sample is above 0.4
    totalEntities = logistigateDict_CAM['importerNum'] + logistigateDict_CAM['outletNum']
    sampMedians = [np.median(logistigateDict_CAM['postSamples'][:,i]) for i in range(totalEntities)]
    highImporterInds = [i for i, x in enumerate(sampMedians[:logistigateDict_CAM['importerNum']]) if x > 0.4]
    util.plotPostSamples(logistigateDict_CAM,importerIndsSubset=highImporterInds,subTitleStr=['\nCambodia - Subset','\nCambodia'])
    util.printEstimates(logistigateDict_CAM,importerIndsSubset=highImporterInds)
    # Run with Cambodia provinces filtered for outlet-type samples
    dataTblDict_CAM_filt = util.testresultsfiletotable('../examples/data/MQD_CAMBODIA_FACILITYFILTER.csv')
    countryMean = np.sum(dataTblDict_CAM_filt['Y']) / np.sum(dataTblDict_CAM_filt['N'])
    dataTblDict_CAM_filt.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 1000,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_CAM_filt = lg.runlogistigate(dataTblDict_CAM_filt)
    numCamImps_fourth = int(np.floor(logistigateDict_CAM_filt['importerNum'] / 4))
    util.plotPostSamples(logistigateDict_CAM_filt, plotType='int90',
                         importerIndsSubset=np.arange(numCamImps_fourth).tolist(),
                         subTitleStr=['\nCambodia (filtered) - 1st Quarter', '\nCambodia (filtered)'])
    util.plotPostSamples(logistigateDict_CAM_filt, plotType='int90',
                         importerIndsSubset=np.arange(numCamImps_fourth, numCamImps_fourth * 2).tolist(),
                         subTitleStr=['\nCambodia (filtered) - 2nd Quarter', '\nCambodia (filtered)'])
    util.plotPostSamples(logistigateDict_CAM_filt, plotType='int90',
                         importerIndsSubset=np.arange(numCamImps_fourth * 2, numCamImps_fourth * 3).tolist(),
                         subTitleStr=['\nCambodia (filtered) - 3rd Quarter', '\nCambodia (filtered)'])
    util.plotPostSamples(logistigateDict_CAM_filt, plotType='int90',
                         importerIndsSubset=np.arange(numCamImps_fourth * 3, logistigateDict_CAM_filt['importerNum']).tolist(),
                         subTitleStr=['\nCambodia (filtered) - 4th Quarter', '\nCambodia (filtered)'])
    # Run with Cambodia provinces filtered for antibiotics
    dataTblDict_CAM_antibiotic = util.testresultsfiletotable('../examples/data/MQD_CAMBODIA_ANTIBIOTIC.csv')
    countryMean = np.sum(dataTblDict_CAM_antibiotic['Y']) / np.sum(dataTblDict_CAM_antibiotic['N'])
    dataTblDict_CAM_antibiotic.update({'diagSens': 1.0,
                                 'diagSpec': 1.0,
                                 'numPostSamples': 1000,
                                 'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                                 'MCMCdict': MCMCdict})
    logistigateDict_CAM_antibiotic = lg.runlogistigate(dataTblDict_CAM_antibiotic)
    numCamImps_third = int(np.floor(logistigateDict_CAM_antibiotic['importerNum'] / 3))
    util.plotPostSamples(logistigateDict_CAM_antibiotic, plotType='int90',
                         importerIndsSubset=np.arange(numCamImps_third).tolist(),
                         subTitleStr=['\nCambodia - 1st Third (Antibiotics)', '\nCambodia (Antibiotics)'])
    util.plotPostSamples(logistigateDict_CAM_antibiotic, plotType='int90',
                         importerIndsSubset=np.arange(numCamImps_third, numCamImps_third * 2).tolist(),
                         subTitleStr=['\nCambodia - 2nd Third (Antibiotics)', '\nCambodia (Antibiotics)'])
    util.plotPostSamples(logistigateDict_CAM_antibiotic, plotType='int90',
                         importerIndsSubset=np.arange(numCamImps_third * 2, logistigateDict_CAM_antibiotic['importerNum']).tolist(),
                         subTitleStr=['\nCambodia - 3rd Third (Antibiotics)', '\nCambodia (Antibiotics)'])
    util.printEstimates(logistigateDict_CAM_antibiotic)
    # Run with Cambodia provinces filtered for antibiotics
    dataTblDict_CAM_antimalarial = util.testresultsfiletotable('../examples/data/MQD_CAMBODIA_ANTIMALARIAL.csv')
    countryMean = np.sum(dataTblDict_CAM_antimalarial['Y']) / np.sum(dataTblDict_CAM_antimalarial['N'])
    dataTblDict_CAM_antimalarial.update({'diagSens': 1.0,
                                       'diagSpec': 1.0,
                                       'numPostSamples': 1000,
                                       'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                                       'MCMCdict': MCMCdict})
    logistigateDict_CAM_antimalarial = lg.runlogistigate(dataTblDict_CAM_antimalarial)
    numCamImps_half = int(np.floor(logistigateDict_CAM_antimalarial['importerNum'] / 2))
    util.plotPostSamples(logistigateDict_CAM_antimalarial, plotType='int90',
                         importerIndsSubset=np.arange(numCamImps_half).tolist(),
                         subTitleStr=['\nCambodia - 1st Half (Antimalarials)', '\nCambodia (Antimalarials)'])
    util.plotPostSamples(logistigateDict_CAM_antimalarial, plotType='int90',
                         importerIndsSubset=np.arange(numCamImps_half,
                                                      logistigateDict_CAM_antimalarial['importerNum']).tolist(),
                         subTitleStr=['\nCambodia - 2nd Half (Antimalarials)', '\nCambodia (Antimalarials)'])
    util.printEstimates(logistigateDict_CAM_antimalarial)

    # Run with Ethiopia provinces
    dataTblDict_ETH = util.testresultsfiletotable('../examples/data/MQD_ETHIOPIA.csv')
    countryMean = np.sum(dataTblDict_ETH['Y']) / np.sum(dataTblDict_ETH['N'])
    dataTblDict_ETH.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 500,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_ETH = lg.runlogistigate(dataTblDict_ETH)
    util.plotPostSamples(logistigateDict_ETH)
    util.printEstimates(logistigateDict_ETH)

    # Run with Ghana provinces
    dataTblDict_GHA = util.testresultsfiletotable('../examples/data/MQD_GHANA.csv')
    countryMean = np.sum(dataTblDict_GHA['Y']) / np.sum(dataTblDict_GHA['N'])
    dataTblDict_GHA.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 1000,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_GHA = lg.runlogistigate(dataTblDict_GHA)
    util.plotPostSamples(logistigateDict_GHA, plotType='int90',
                         subTitleStr=['\nGhana', '\nGhana'])
    util.printEstimates(logistigateDict_GHA)
    # Plot importers subset where median sample is above 0.4
    totalEntities = logistigateDict_GHA['importerNum'] + logistigateDict_GHA['outletNum']
    sampMedians = [np.median(logistigateDict_GHA['postSamples'][:, i]) for i in range(totalEntities)]
    highImporterInds = [i for i, x in enumerate(sampMedians[:logistigateDict_GHA['importerNum']]) if x > 0.4]
    highOutletInds = [i for i, x in enumerate(sampMedians[logistigateDict_GHA['importerNum']:]) if x > 0.15]
    util.plotPostSamples(logistigateDict_GHA, importerIndsSubset=highImporterInds,
                         outletIndsSubset=highOutletInds,
                         subTitleStr=['\nGhana - Subset', '\nGhana - Subset'])
    util.printEstimates(logistigateDict_GHA, importerIndsSubset=highImporterInds,outletIndsSubset=highOutletInds)
    # Run with Ghana provinces filtered for outlet-type samples
    dataTblDict_GHA_filt = util.testresultsfiletotable('../examples/data/MQD_GHANA_FACILITYFILTER.csv')
    countryMean = np.sum(dataTblDict_GHA_filt['Y']) / np.sum(dataTblDict_GHA_filt['N'])
    dataTblDict_GHA_filt.update({'diagSens': 1.0,
                                 'diagSpec': 1.0,
                                 'numPostSamples': 1000,
                                 'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                                 'MCMCdict': MCMCdict})
    logistigateDict_GHA_filt = lg.runlogistigate(dataTblDict_GHA_filt)
    util.plotPostSamples(logistigateDict_GHA_filt, plotType='int90',
                         subTitleStr=['\nGhana (filtered)', '\nGhana (filtered)'])
    util.printEstimates(logistigateDict_GHA_filt)
    # Run with Ghana provinces filtered for antimalarials
    dataTblDict_GHA_antimalarial = util.testresultsfiletotable('../examples/data/MQD_GHANA_ANTIMALARIAL.csv')
    countryMean = np.sum(dataTblDict_GHA_antimalarial['Y']) / np.sum(dataTblDict_GHA_antimalarial['N'])
    dataTblDict_GHA_antimalarial.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 1000,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_GHA_antimalarial = lg.runlogistigate(dataTblDict_GHA_antimalarial)
    util.plotPostSamples(logistigateDict_GHA_antimalarial, plotType='int90',
                         subTitleStr=['\nGhana (Antimalarials)', '\nGhana (Antimalarials)'])
    util.printEstimates(logistigateDict_GHA_antimalarial)

    # Run with Kenya provinces
    dataTblDict_KEN = util.testresultsfiletotable('../examples/data/MQD_KENYA.csv')
    countryMean = np.sum(dataTblDict_KEN['Y']) / np.sum(dataTblDict_KEN['N'])
    dataTblDict_KEN.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 500,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_KEN = lg.runlogistigate(dataTblDict_KEN)
    util.plotPostSamples(logistigateDict_KEN)
    util.printEstimates(logistigateDict_KEN)


    # Run with Laos provinces
    dataTblDict_LAO = util.testresultsfiletotable('../examples/data/MQD_LAOS.csv')
    countryMean = np.sum(dataTblDict_LAO['Y']) / np.sum(dataTblDict_LAO['N'])
    dataTblDict_LAO.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 500,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_LAO = lg.runlogistigate(dataTblDict_LAO)
    util.plotPostSamples(logistigateDict_LAO)
    util.printEstimates(logistigateDict_LAO)


    # Run with Mozambique provinces
    dataTblDict_MOZ = util.testresultsfiletotable('../examples/data/MQD_MOZAMBIQUE.csv')
    countryMean = np.sum(dataTblDict_MOZ['Y']) / np.sum(dataTblDict_MOZ['N'])
    dataTblDict_MOZ.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 500,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_MOZ = lg.runlogistigate(dataTblDict_MOZ)
    util.plotPostSamples(logistigateDict_MOZ)
    util.printEstimates(logistigateDict_MOZ)

    # Run with Nigeria provinces
    dataTblDict_NIG = util.testresultsfiletotable('../examples/data/MQD_NIGERIA.csv')
    countryMean = np.sum(dataTblDict_NIG['Y']) / np.sum(dataTblDict_NIG['N'])
    dataTblDict_NIG.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 500,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_NIG = lg.runlogistigate(dataTblDict_NIG)
    util.plotPostSamples(logistigateDict_NIG)
    util.printEstimates(logistigateDict_NIG)

    # Run with Peru provinces
    dataTblDict_PER = util.testresultsfiletotable('../examples/data/MQD_PERU.csv')
    countryMean = np.sum(dataTblDict_PER['Y']) / np.sum(dataTblDict_PER['N'])
    dataTblDict_PER.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 1000,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_PER = lg.runlogistigate(dataTblDict_PER)
    numPeruImps_half = int(np.floor(logistigateDict_PER['importerNum']/2))
    util.plotPostSamples(logistigateDict_PER, plotType='int90',
                         importerIndsSubset=np.arange(0,numPeruImps_half).tolist(), subTitleStr=['\nPeru - 1st Half', '\nPeru'])
    util.plotPostSamples(logistigateDict_PER, plotType='int90',
                         importerIndsSubset=np.arange(numPeruImps_half,logistigateDict_PER['importerNum']).tolist(),
                         subTitleStr=['\nPeru - 2nd Half', '\nPeru'])
    util.printEstimates(logistigateDict_PER)
    # Plot importers subset where median sample is above 0.4
    totalEntities = logistigateDict_PER['importerNum'] + logistigateDict_PER['outletNum']
    sampMedians = [np.median(logistigateDict_PER['postSamples'][:, i]) for i in range(totalEntities)]
    highImporterInds = [i for i, x in enumerate(sampMedians[:logistigateDict_PER['importerNum']]) if x > 0.4]
    highImporterInds = [highImporterInds[i] for i in [3,6,7,8,9,12,13,16]] # Only manufacturers with more than 1 sample
    highOutletInds = [i for i, x in enumerate(sampMedians[logistigateDict_PER['importerNum']:]) if x > 0.12]
    util.plotPostSamples(logistigateDict_PER, importerIndsSubset=highImporterInds,
                         outletIndsSubset=highOutletInds,
                         subTitleStr=['\nPeru - Subset', '\nPeru - Subset'])
    util.printEstimates(logistigateDict_PER, importerIndsSubset=highImporterInds, outletIndsSubset=highOutletInds)
    # Run with Peru provinces filtered for outlet-type samples
    dataTblDict_PER_filt = util.testresultsfiletotable('../examples/data/MQD_PERU_FACILITYFILTER.csv')
    countryMean = np.sum(dataTblDict_PER_filt['Y']) / np.sum(dataTblDict_PER_filt['N'])
    dataTblDict_PER_filt.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 1000,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_PER_filt = lg.runlogistigate(dataTblDict_PER_filt)
    numPeruImps_half = int(np.floor(logistigateDict_PER_filt['importerNum'] / 2))
    util.plotPostSamples(logistigateDict_PER_filt, plotType='int90',
                         importerIndsSubset=np.arange(0, numPeruImps_half).tolist(),
                         subTitleStr=['\nPeru - 1st Half (filtered)', '\nPeru (filtered)'])
    util.plotPostSamples(logistigateDict_PER_filt, plotType='int90',
                         importerIndsSubset=np.arange(numPeruImps_half, logistigateDict_PER_filt['importerNum']).tolist(),
                         subTitleStr=['\nPeru - 2nd Half (filtered)', '\nPeru (filtered)'])
    util.printEstimates(logistigateDict_PER_filt)
    # Run with Peru provinces filtered for antibiotics
    dataTblDict_PER_antibiotics = util.testresultsfiletotable('../examples/data/MQD_PERU_ANTIBIOTIC.csv')
    countryMean = np.sum(dataTblDict_PER_antibiotics['Y']) / np.sum(dataTblDict_PER_antibiotics['N'])
    dataTblDict_PER_antibiotics.update({'diagSens': 1.0,
                                 'diagSpec': 1.0,
                                 'numPostSamples': 1000,
                                 'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                                 'MCMCdict': MCMCdict})
    logistigateDict_PER_antibiotics = lg.runlogistigate(dataTblDict_PER_antibiotics)
    numPeruImps_half = int(np.floor(logistigateDict_PER_antibiotics['importerNum'] / 2))
    util.plotPostSamples(logistigateDict_PER_antibiotics, plotType='int90',
                         importerIndsSubset=np.arange(numPeruImps_half).tolist(),
                         subTitleStr=['\nPeru - 1st Half (Antibiotics)', '\nPeru (Antibiotics)'])
    util.plotPostSamples(logistigateDict_PER_antibiotics, plotType='int90',
                         importerIndsSubset=np.arange(numPeruImps_half, logistigateDict_PER_antibiotics['importerNum']).tolist(),
                         subTitleStr=['\nPeru - 2nd Half (Antibiotics)', '\nPeru (Antibiotics)'])
    util.printEstimates(logistigateDict_PER_antibiotics)

    # Run with Philippines provinces
    dataTblDict_PHI = util.testresultsfiletotable('../examples/data/MQD_PHILIPPINES.csv')
    countryMean = np.sum(dataTblDict_PHI['Y']) / np.sum(dataTblDict_PHI['N'])
    dataTblDict_PHI.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 1000,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_PHI = lg.runlogistigate(dataTblDict_PHI)
    util.plotPostSamples(logistigateDict_PHI,plotType='int90',subTitleStr=['\nPhilippines','\nPhilippines'])
    util.printEstimates(logistigateDict_PHI)
    # Plot importers subset where median sample is above 0.1
    totalEntities = logistigateDict_PHI['importerNum'] + logistigateDict_PHI['outletNum']
    sampMedians = [np.median(logistigateDict_PHI['postSamples'][:, i]) for i in range(totalEntities)]
    highImporterInds = [i for i, x in enumerate(sampMedians[:logistigateDict_PHI['importerNum']]) if x > 0.1]
    #highImporterInds = [highImporterInds[i] for i in
    #                    [3, 6, 7, 8, 9, 12, 13, 16]]  # Only manufacturers with more than 1 sample
    highOutletInds = [i for i, x in enumerate(sampMedians[logistigateDict_PHI['importerNum']:]) if x > 0.1]
    util.plotPostSamples(logistigateDict_PHI, importerIndsSubset=highImporterInds,
                         outletIndsSubset=highOutletInds,
                         subTitleStr=['\nPhilippines - Subset', '\nPhilippines - Subset'])
    util.printEstimates(logistigateDict_PHI, importerIndsSubset=highImporterInds, outletIndsSubset=highOutletInds)
    # Run with Philippines provinces filtered for outlet-type samples
    dataTblDict_PHI_filt = util.testresultsfiletotable('../examples/data/MQD_PHILIPPINES_FACILITYFILTER.csv')
    countryMean = np.sum(dataTblDict_PHI_filt['Y']) / np.sum(dataTblDict_PHI_filt['N'])
    dataTblDict_PHI_filt.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 1000,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_PHI_filt = lg.runlogistigate(dataTblDict_PHI_filt)
    util.plotPostSamples(logistigateDict_PHI_filt, plotType='int90', subTitleStr=['\nPhilippines (filtered)', '\nPhilippines (filtered)'])
    util.printEstimates(logistigateDict_PHI_filt)

    # Run with Thailand provinces
    dataTblDict_THA = util.testresultsfiletotable('../examples/data/MQD_THAILAND.csv')
    countryMean = np.sum(dataTblDict_THA['Y']) / np.sum(dataTblDict_THA['N'])
    dataTblDict_THA.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 500,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_THA = lg.runlogistigate(dataTblDict_THA)
    util.plotPostSamples(logistigateDict_THA)
    util.printEstimates(logistigateDict_THA)

    # Run with Viet Nam provinces
    dataTblDict_VIE = util.testresultsfiletotable('../examples/data/MQD_VIETNAM.csv')
    countryMean = np.sum(dataTblDict_VIE['Y']) / np.sum(dataTblDict_VIE['N'])
    dataTblDict_VIE.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 500,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_VIE = lg.runlogistigate(dataTblDict_VIE)
    util.plotPostSamples(logistigateDict_VIE)
    util.printEstimates(logistigateDict_VIE)

    return


