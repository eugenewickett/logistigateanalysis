# -*- coding: utf-8 -*-
"""
Script for analyzing the case study data. Inference and plan utility require use of the logistigate package, available
at https://logistigate.readthedocs.io/en/main/.
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
import scipy.stats as spstat
import matplotlib.pyplot as plt
import random
import time
from math import comb
import matplotlib.cm as cm


def casestudy_sensitivity():
    """
    Use the Exploratory setting to probe different choices and parameters. We inspect allocations at 180 tests (the
    original budget was 177 tests) and 90 tests (half of 180). We also inspect the budget saved under these choices
    and parameters.
    """
    Nexpl = np.array([[1., 1., 10., 1., 3., 0., 1., 6., 7., 5., 0., 0., 4.],
                      [1., 1., 4., 2., 0., 1., 1., 2., 0., 4., 0., 0., 1.],
                      [3., 17., 31., 4., 2., 0., 1., 6., 0., 23., 1., 2., 5.],
                      [1., 1., 15., 2., 0., 0., 0., 1., 0., 6., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    Yexpl = np.array([[0., 0., 7., 0., 3., 0., 1., 0., 1., 0., 0., 0., 4.],
                      [0., 0., 2., 2., 0., 1., 1., 0., 0., 1., 0., 0., 1.],
                      [0., 0., 15., 3., 2., 0., 0., 2., 0., 1., 1., 2., 5.],
                      [0., 0., 5., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

    (numTN, numSN) = Nexpl.shape

    from logistigate.logistigate import utilities as util
    csdict_expl = util.initDataDict(Nexpl, Yexpl)  # Initialize necessary logistigate keys

    csdict_expl['TNnames'] = ['MOD_39', 'MOD_17', 'MODHIGH_95', 'MODHIGH_26',
                              'MODHIGH_EXPL_1', 'MOD_EXPL_1', 'MODHIGH_EXPL_2', 'MOD_EXPL_2']
    csdict_expl['SNnames'] = ['MNFR ' + str(i + 1) for i in range(numSN)]

    # Use observed data to form Q for tested nodes; use bootstrap data for untested nodes
    numBoot = 44  # Average across each TN in original data set
    SNprobs = np.sum(csdict_expl['N'], axis=0) / np.sum(csdict_expl['N'])
    np.random.seed(33)
    Qvecs = np.random.multinomial(numBoot, SNprobs, size=4) / numBoot
    csdict_expl['Q'] = np.vstack((csdict_expl['Q'][:4], Qvecs))

    # Region catchment proportions, for market terms
    TNcach = np.array([0.17646, 0.05752, 0.09275, 0.09488, 0.17695, 0.22799, 0.07805, 0.0954])
    tempQ = csdict_expl['N'][:4] / np.sum(csdict_expl['N'][:4], axis=1).reshape(4, 1)
    tempTNcach = TNcach[:4] / np.sum(TNcach[:4])
    SNcach = np.matmul(tempTNcach, tempQ)

    # Build prior
    SNpriorMean = np.repeat(sps.logit(0.1), numSN)
    # Establish test nodes according to assessment by regulators
    TNpriorMean = sps.logit(np.array([0.1, 0.1, 0.15, 0.15, 0.15, 0.1, 0.15, 0.1]))
    TNvar, SNvar = 2., 4.
    csdict_expl['prior'] = prior_normal_assort(np.concatenate((SNpriorMean, TNpriorMean)),
                                               np.diag(np.concatenate((np.repeat(SNvar, numSN), np.repeat(TNvar, numTN)))))

    # Set up MCMC
    csdict_expl['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    # Generate posterior draws
    numdraws = 80000
    csdict_expl['numPostSamples'] = numdraws
    np.random.seed(1000)  # To replicate draws later
    csdict_expl = methods.GeneratePostSamples(csdict_expl)

    ########################
    ### OUTLINE FOR EACH SECTION OF CASE STUDY
    ########################
    # 1) Generate some heuristic runs to form allocation
    # 2) Estimate comprehensive utility for 3 methods: heuristic, uniform, rudimentary

    # Loss specification
    paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                                  marketvec=np.ones(numTN+numSN), candneighnum=1000)

    # Set limits of data collection and intervals for calculation
    # For sensitivity, set testmax to highest expected allocation for any one node
    testmax, testint = 100, 10
    testarr = np.arange(testint, testmax + testint, testint)

    # Set MCMC draws to use in fast algorithm
    numcanddraws, numtruthdraws, numdatadraws = 5000, 5000, 3000
    numReps = 10  # Number of repeat runs to make


    
    # np.save(os.path.join('casestudyoutputs', 'PREVIOUS', 'SENSITIVITY', 'margutilset'), np.array(margUtilSet))

    #todo: #####################################
    #todo: STOPPED HERE 17-MAY



    # We now check the allocation and budgetary savings sensitivity to various parameters at a particular budget
    sampBudget = 180
    unifDes = np.zeros(numTN) + 1 / numTN
    origDes = np.sum(rd3_N, axis=1) / np.sum(rd3_N)

    # Use different loss parameters
    ###########
    # todo: checkSlope = 0.3
    ###########
    paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.3,
                                                  marketvec=np.ones(numTN+numSN), candneighnum=1000)


    utilMatList = []


    for rep in range(numReps):
        # Withdraw a subset of MCMC prior draws
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New loss draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        # Get new data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        # Get marginal utilities at each test node
        currMargUtilMat = GetMargUtilAtNodes(dictTemp.copy(), testMax, testInt, lossDict.copy(), utilDict.copy(),
                                             masterDraws=CSdict3['postSamples'], printUpdate=True)
        print(repr(currMargUtilMat))
        utilMatList.append(currMargUtilMat)
    '''21-MAR run

    array([[0.        , 0.02500037, 0.05453001, 0.08209539, 0.10406642,
        0.12635247, 0.14372454, 0.16378819, 0.18127501, 0.19769524,
        0.21806803],
       [0.        , 0.06780398, 0.12588295, 0.17474233, 0.2130504 ,
        0.25103884, 0.27618104, 0.30498385, 0.32892034, 0.35248336,
        0.3769582 ],
       [0.        , 0.0231166 , 0.04776599, 0.0695312 , 0.08821245,
        0.10760009, 0.12111096, 0.14170292, 0.15748052, 0.17533819,
        0.18795814],
       [0.        , 0.01274688, 0.02207516, 0.03094908, 0.03848035,
        0.04623635, 0.05366907, 0.06208322, 0.06854747, 0.07078396,
        0.07963057],
       [0.        , 0.12318974, 0.17220442, 0.20022558, 0.22574309,
        0.24298633, 0.26482079, 0.27670645, 0.29379246, 0.30827047,
        0.32425711],
       [0.        , 0.06763408, 0.11554658, 0.14757023, 0.16712204,
        0.18568884, 0.20369205, 0.21839598, 0.23067107, 0.24024285,
        0.25794557],
       [0.        , 0.09427582, 0.15597956, 0.19570334, 0.22112415,
        0.24196177, 0.26310672, 0.28087477, 0.29439486, 0.30353926,
        0.3170758 ],
       [0.        , 0.08791447, 0.14834522, 0.18427286, 0.22501034,
        0.25045911, 0.2781266 , 0.30292225, 0.33079005, 0.34387117,
        0.36923574]])
    array([[0.        , 0.04263898, 0.0805571 , 0.11234862, 0.1403126 ,
        0.16875213, 0.18675966, 0.20837249, 0.22678839, 0.250339  ,
        0.26368037],
       [0.        , 0.10668159, 0.17626777, 0.23209587, 0.27828299,
        0.31760766, 0.34928672, 0.3738254 , 0.40164507, 0.42471862,
        0.4535201 ],
       [0.        , 0.03326988, 0.06463894, 0.09013261, 0.11849815,
        0.13636318, 0.15945141, 0.18186024, 0.197517  , 0.22064851,
        0.23738432],
       [0.        , 0.02260935, 0.04522903, 0.05965912, 0.070499  ,
        0.08017769, 0.09357638, 0.1044922 , 0.1115661 , 0.12266323,
        0.12643848],
       [0.        , 0.11983738, 0.17577183, 0.21207856, 0.24248467,
        0.26709557, 0.28931028, 0.31106719, 0.32662903, 0.34681699,
        0.36575629],
       [0.        , 0.16976545, 0.22001581, 0.25354531, 0.27824367,
        0.29792475, 0.30811732, 0.32223136, 0.33516808, 0.3471305 ,
        0.35538685],
       [0.        , 0.14437884, 0.20555892, 0.24294385, 0.27550474,
        0.30383918, 0.32267132, 0.34398143, 0.35795046, 0.3729873 ,
        0.39160028],
       [0.        , 0.0947162 , 0.15201215, 0.20010508, 0.23995104,
        0.26819929, 0.30217706, 0.3262525 , 0.35324365, 0.3766628 ,
        0.40135085]])
    array([[0.        , 0.01587516, 0.04593522, 0.07223151, 0.09514553,
        0.11917388, 0.13999842, 0.16216311, 0.1805129 , 0.20201705,
        0.2126472 ],
       [0.        , 0.07058362, 0.13200391, 0.17571572, 0.21648197,
        0.2498789 , 0.27963138, 0.30645987, 0.32903156, 0.34734348,
        0.37093671],
       [0.        , 0.01119609, 0.02935217, 0.05450204, 0.06986139,
        0.08942349, 0.10814692, 0.12211996, 0.14378273, 0.15544419,
        0.17274479],
       [0.        , 0.00779084, 0.02195131, 0.03396341, 0.04757296,
        0.05536284, 0.06179379, 0.07038192, 0.07845354, 0.08556913,
        0.08999636],
       [0.        , 0.07451111, 0.12030676, 0.16170979, 0.19042464,
        0.2129504 , 0.23801667, 0.25476103, 0.27441601, 0.29344206,
        0.3072715 ],
       [0.        , 0.11605207, 0.16142447, 0.18783149, 0.21283935,
        0.22902721, 0.24190524, 0.26048196, 0.27020456, 0.2792825 ,
        0.29289628],
       [0.        , 0.12127702, 0.17564384, 0.21138176, 0.24317561,
        0.26714787, 0.28969989, 0.3025183 , 0.31865848, 0.33093037,
        0.34516775],
       [0.        , 0.08597358, 0.14435954, 0.19249878, 0.22742487,
        0.26077066, 0.28591513, 0.31405436, 0.3391699 , 0.36238359,
        0.38004766]])
    utilMatList = [np.array([[0.        , 0.02987273, 0.06525859, 0.09093212, 0.12042844,
        0.14311871, 0.16312245, 0.18114201, 0.19560633, 0.21208205,
        0.23025551],
       [0.        , 0.05167086, 0.11339466, 0.16032641, 0.20332102,
        0.23676281, 0.26918115, 0.29902935, 0.32377649, 0.34639625,
        0.37086335],
       [0.        , 0.02448826, 0.05228473, 0.07657187, 0.09806873,
        0.11981395, 0.14070131, 0.15774672, 0.17648881, 0.19393561,
        0.21304374],
       [0.        , 0.01636897, 0.02975276, 0.03622167, 0.05152468,
        0.05760275, 0.06538814, 0.07191806, 0.08184431, 0.09189486,
        0.09492135],
       [0.        , 0.11032845, 0.16476965, 0.20177127, 0.23169792,
        0.25892656, 0.28164654, 0.30245426, 0.31937686, 0.33805109,
        0.35663242],
       [0.        , 0.0892706 , 0.14144644, 0.1773953 , 0.20140365,
        0.22173483, 0.23987947, 0.25359025, 0.2649582 , 0.28201554,
        0.29154265],
       [0.        , 0.09411578, 0.15417961, 0.19097058, 0.21931113,
        0.23784049, 0.25735508, 0.27423437, 0.28904966, 0.3022289 ,
        0.31878039],
       [0.        , 0.08576752, 0.14682849, 0.19408626, 0.23024442,
        0.25924388, 0.2881656 , 0.31408096, 0.33713355, 0.35968549, 0.38057977]]), 
        np.array([[0.        , 0.02500037, 0.05453001, 0.08209539, 0.10406642,
        0.12635247, 0.14372454, 0.16378819, 0.18127501, 0.19769524, 0.21806803],
       [0.        , 0.06780398, 0.12588295, 0.17474233, 0.2130504 ,
        0.25103884, 0.27618104, 0.30498385, 0.32892034, 0.35248336, 0.3769582 ],
       [0.        , 0.0231166 , 0.04776599, 0.0695312 , 0.08821245,
        0.10760009, 0.12111096, 0.14170292, 0.15748052, 0.17533819, 0.18795814],
       [0.        , 0.01274688, 0.02207516, 0.03094908, 0.03848035,
        0.04623635, 0.05366907, 0.06208322, 0.06854747, 0.07078396, 0.07963057],
       [0.        , 0.12318974, 0.17220442, 0.20022558, 0.22574309,
        0.24298633, 0.26482079, 0.27670645, 0.29379246, 0.30827047, 0.32425711],
       [0.        , 0.06763408, 0.11554658, 0.14757023, 0.16712204,
        0.18568884, 0.20369205, 0.21839598, 0.23067107, 0.24024285, 0.25794557],
       [0.        , 0.09427582, 0.15597956, 0.19570334, 0.22112415,
        0.24196177, 0.26310672, 0.28087477, 0.29439486, 0.30353926, 0.3170758 ],
       [0.        , 0.08791447, 0.14834522, 0.18427286, 0.22501034,
        0.25045911, 0.2781266 , 0.30292225, 0.33079005, 0.34387117, 0.36923574]]), 
        np.array([[0.        , 0.04263898, 0.0805571 , 0.11234862, 0.1403126 ,
        0.16875213, 0.18675966, 0.20837249, 0.22678839, 0.250339  ,
        0.26368037],
       [0.        , 0.10668159, 0.17626777, 0.23209587, 0.27828299,
        0.31760766, 0.34928672, 0.3738254 , 0.40164507, 0.42471862,
        0.4535201 ],
       [0.        , 0.03326988, 0.06463894, 0.09013261, 0.11849815,
        0.13636318, 0.15945141, 0.18186024, 0.197517  , 0.22064851,
        0.23738432],
       [0.        , 0.02260935, 0.04522903, 0.05965912, 0.070499  ,
        0.08017769, 0.09357638, 0.1044922 , 0.1115661 , 0.12266323,
        0.12643848],
       [0.        , 0.11983738, 0.17577183, 0.21207856, 0.24248467,
        0.26709557, 0.28931028, 0.31106719, 0.32662903, 0.34681699,
        0.36575629],
       [0.        , 0.16976545, 0.22001581, 0.25354531, 0.27824367,
        0.29792475, 0.30811732, 0.32223136, 0.33516808, 0.3471305 ,
        0.35538685],
       [0.        , 0.14437884, 0.20555892, 0.24294385, 0.27550474,
        0.30383918, 0.32267132, 0.34398143, 0.35795046, 0.3729873 ,
        0.39160028],
       [0.        , 0.0947162 , 0.15201215, 0.20010508, 0.23995104,
        0.26819929, 0.30217706, 0.3262525 , 0.35324365, 0.3766628 ,
        0.40135085]]), 
        np.array([[0.        , 0.01587516, 0.04593522, 0.07223151, 0.09514553,
        0.11917388, 0.13999842, 0.16216311, 0.1805129 , 0.20201705,
        0.2126472 ],
       [0.        , 0.07058362, 0.13200391, 0.17571572, 0.21648197,
        0.2498789 , 0.27963138, 0.30645987, 0.32903156, 0.34734348,
        0.37093671],
       [0.        , 0.01119609, 0.02935217, 0.05450204, 0.06986139,
        0.08942349, 0.10814692, 0.12211996, 0.14378273, 0.15544419,
        0.17274479],
       [0.        , 0.00779084, 0.02195131, 0.03396341, 0.04757296,
        0.05536284, 0.06179379, 0.07038192, 0.07845354, 0.08556913,
        0.08999636],
       [0.        , 0.07451111, 0.12030676, 0.16170979, 0.19042464,
        0.2129504 , 0.23801667, 0.25476103, 0.27441601, 0.29344206,
        0.3072715 ],
       [0.        , 0.11605207, 0.16142447, 0.18783149, 0.21283935,
        0.22902721, 0.24190524, 0.26048196, 0.27020456, 0.2792825 ,
        0.29289628],
       [0.        , 0.12127702, 0.17564384, 0.21138176, 0.24317561,
        0.26714787, 0.28969989, 0.3025183 , 0.31865848, 0.33093037,
        0.34516775],
       [0.        , 0.08597358, 0.14435954, 0.19249878, 0.22742487,
        0.26077066, 0.28591513, 0.31405436, 0.3391699 , 0.36238359,
        0.38004766]]), 
        np.array([[0.        , 0.03251971, 0.06843361, 0.09871152, 0.12714824,
        0.14767408, 0.1691276 , 0.18636503, 0.20809668, 0.22473232,
        0.23949598],
       [0.        , 0.05917184, 0.11805847, 0.16443774, 0.20516079,
        0.23630029, 0.27258007, 0.2930291 , 0.31589052, 0.34184075,
        0.36247042],
       [0.        , 0.01805197, 0.03653106, 0.05896414, 0.07931535,
        0.10320268, 0.1190687 , 0.13519066, 0.15517717, 0.17141938,
        0.18448667],
       [0.        , 0.01665995, 0.02790249, 0.04230972, 0.05156995,
        0.06198682, 0.07041216, 0.07866921, 0.08645476, 0.09203822,
        0.09839002],
       [0.        , 0.10347124, 0.14993074, 0.18895728, 0.21806734,
        0.24492117, 0.26628176, 0.28702734, 0.30282592, 0.32122612,
        0.33556131],
       [0.        , 0.11525652, 0.16614162, 0.20308849, 0.22684349,
        0.24589503, 0.26209648, 0.27615994, 0.28847559, 0.29948998,
        0.31038607],
       [0.        , 0.13738655, 0.19818119, 0.23976322, 0.2697537 ,
        0.29209653, 0.31080454, 0.32872838, 0.34258857, 0.35536578,
        0.36757619],
       [0.        , 0.07321535, 0.13626956, 0.17958915, 0.21335394,
        0.24568622, 0.2720279 , 0.29436191, 0.31913059, 0.3384415 ,
        0.35818161]])]
    '''
    # Get average utility matrix
    avgUtilMat = np.average(np.array(utilMatList), axis=0)
    '''21-MAR
    avgUtilMat = np.array([[0.        , 0.02918139, 0.06294291, 0.09126383, 0.11742025,
        0.14101425, 0.16054653, 0.18036617, 0.19845586, 0.21737313,
        0.23282942],
       [0.        , 0.07118238, 0.13312155, 0.18146362, 0.22325944,
        0.2583177 , 0.28937207, 0.31546551, 0.3398528 , 0.36255649,
        0.38694976],
       [0.        , 0.02202456, 0.04611458, 0.06994037, 0.09079121,
        0.11128068, 0.12969586, 0.1477241 , 0.16608925, 0.18335718,
        0.19912353],
       [0.        , 0.0152352 , 0.02938215, 0.0406206 , 0.05192939,
        0.06027329, 0.06896791, 0.07750892, 0.08537324, 0.09258988,
        0.09787536],
       [0.        , 0.10626758, 0.15659668, 0.19294849, 0.22168353,
        0.245376  , 0.26801521, 0.28640325, 0.30340806, 0.32156135,
        0.33789573],
       [0.        , 0.11159575, 0.16091498, 0.19388616, 0.21729044,
        0.23605413, 0.25113811, 0.2661719 , 0.2778955 , 0.28963228,
        0.30163148],
       [0.        , 0.1182868 , 0.17790863, 0.21615255, 0.24577387,
        0.26857717, 0.28872751, 0.30606745, 0.32052841, 0.33301032,
        0.34804008],
       [0.        , 0.08551742, 0.14556299, 0.19011043, 0.22719692,
        0.25687183, 0.28528246, 0.3103344 , 0.33589355, 0.35620891,
        0.37787913]])
    '''
    # Find allocation for sample budget
    allocArr = sampf.forwardAllocateWithBudget(avgUtilMat, int(sampBudget / testInt))
    designArr = allocArr / np.sum(allocArr, axis=0)
    # Get utility for this allocation at the sample budget
    utilDict.update({'method': 'weightsNodeDraw4linear'})
    compUtilList, unifUtilList, origUtilList = [], [], []
    numReps = 5
    for rep in range(1, numReps):
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New Bayes draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        print('Generating loss matrix...')
        tempLossMat = lf.lossMatSetBayesDraws(dictTemp['postSamples'], lossDict.copy(), lossDict['bayesDraws'])
        tempLossDict = lossDict.copy()
        tempLossDict.update({'lossMat': tempLossMat})
        newBayesDraws, newLossMat = lf.add_cand_neighbors(tempLossDict.copy(), CSdict3['postSamples'],
                                                          dictTemp['postSamples'])
        tempLossDict.update({'bayesDraws': newBayesDraws, 'lossMat': newLossMat})
        baseLoss = (np.sum(newLossMat, axis=1) / newLossMat.shape[1]).min()
        # Get a new set of data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        currCompUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[designArr],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Heuristic utility: ' + str(currCompUtil))
        compUtilList.append(currCompUtil)
        '''21-MAR
        compUtilList = [1.1555343560205777, 1.169845828163453, 1.1571208715429826, 1.1785292518865123, 1.1233927643720962]
        '''
        # Find the equivalent uniform allocation
        currUnifUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[unifDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Uniform utility: ' + str(currUnifUtil))
        unifUtilList.append([currUnifUtil])
        unifAdd, contUnif, unifCount = 0, False, 0
        if currUnifUtil < currCompUtil:
            contUnif = True
        while contUnif:
            unifAdd += testInt
            print('Adding ' + str(unifAdd) + ' for uniform')
            currUnifUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[unifDes], numtests=sampBudget + unifAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currUnifUtil))
            unifUtilList[rep].append(currUnifUtil)
            if currUnifUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if unifCount < 3:
                    unifCount += 1
                else:
                    contUnif = False
        '''21-MAR
        unifUtilList = [[1.031142558195473, 1.061522586421201, 1.1176784774252675, 1.1474028826094198, 1.1899354564358111, 1.2324895284848263, 1.2583891316018003, 1.2921952403491423], [1.0447047709077837, 1.0857378106176627, 1.129815610714283, 1.1630465378451271, 1.20025428755357, 1.2444289498126726, 1.276577444371557, 1.3131865510457974], [1.0209996730632724, 1.068026736414653, 1.1083006457178581, 1.152707515865997, 1.189846887805912, 1.2281802711796428, 1.2635915391489316, 1.305063065296511], [1.0554050579136507, 1.092289676069889, 1.133351111753512, 1.1646405516534637, 1.2078565722192378, 1.2353260019379881, 1.285901151171858, 1.3116839058895984], [1.0028931084280042, 1.0432022510929304, 1.0899931237876999, 1.1428924632681179, 1.1644789963331155, 1.1948377915689488, 1.2298735698120855]]
        '''
        # Find the equivalent rudimentary allocation
        currOrigUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[origDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Rudimentary utility: ' + str(currOrigUtil))
        origUtilList.append([currOrigUtil])
        origAdd, contOrig, origCount = 0, False, 0
        if currOrigUtil < currCompUtil:
            contOrig = True
        while contOrig:
            origAdd += testInt * 3
            print('Adding ' + str(origAdd) + ' for rudimentary')
            currOrigUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[origDes], numtests=sampBudget + origAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currOrigUtil))
            origUtilList[rep].append(currOrigUtil)
            if currOrigUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if origCount < 3:
                    origCount += 1
                else:
                    contOrig = False
        '''21-MAR
        origUtilList = [[0.40996292275522084, 0.48116195885241275, 0.552516409556481, 0.6197232833362456, 0.6990862314445803, 0.7743225519969417, 0.8423718128097923, 0.9354915374575268, 0.9803577957439944, 1.0315007726401042, 1.113079354051902, 1.1581318693666618, 1.2336130216369794, 1.2947456645664617, 1.358923566497555], [0.4400268024964147, 0.5064038755380915, 0.5855235608294036, 0.6602452938110739, 0.7278114252396648, 0.8078427591747226, 0.8729922061368613, 0.9558521695666298, 1.015931558732917, 1.0920797027163367, 1.141092219900505, 1.2034323968671807, 1.2582343090801937, 1.3349740646391357, 1.3837173722879896], [0.4060955510683293, 0.47673964762492194, 0.5423746687647171, 0.6216786373783982, 0.6872344712896221, 0.7809158229884572, 0.8512258964671933, 0.9272845147909763, 0.9809198222804616, 1.0466317238565366, 1.1181968169145904, 1.172473202782672, 1.2318265980181677, 1.2988686696215277, 1.343830895112268], [0.47137569750967234, 0.5368771991639587, 0.6115672541016073, 0.6742713181789588, 0.7435888585880082, 0.8163750796583376, 0.8863976312492627, 0.9803371880422653, 1.0193672686950968, 1.0662706713646637, 1.1559525752294286, 1.2078963406336714, 1.272010945526898, 1.3119062078521946, 1.3927944416074842], [0.39700463041508893, 0.47670282702661737, 0.5498976811827445, 0.6163164141165351, 0.6960169119537714, 0.7693656144728136, 0.8431513043057022, 0.923789604307538, 0.984781017140691, 1.041217347973657, 1.1059348862321734, 1.169439734433464, 1.2115367225600915, 1.2713744137990135, 1.3390837766629637]]
        '''
    compAvg = np.average(compUtilList)
    # Locate closest sample point for uniform and rudimentary to compAvg
    minListLen = np.min([len(i) for i in unifUtilList])
    unifUtilArr = np.array([i[:minListLen] for i in unifUtilList])
    unifAvgArr = np.average(unifUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(unifAvgArr.tolist()) if val > compAvg)
    unifSampSaved = round((compAvg - unifAvgArr[kInd - 1]) / (unifAvgArr[kInd] - unifAvgArr[kInd - 1]) * testInt) + (
            kInd - 1) * testInt
    '''21-MAR: 31 saved'''
    # Rudimentary
    minListLen = np.min([len(i) for i in origUtilList])
    origUtilArr = np.array([i[:minListLen] for i in origUtilList])
    origAvgArr = np.average(origUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(origAvgArr.tolist()) if val > compAvg)
    origSampSaved = round(
        (compAvg - origAvgArr[kInd - 1]) / (origAvgArr[kInd] - origAvgArr[kInd - 1]) * testInt * 3) + (
                            kInd - 1) * testInt * 3
    '''21-MAR: 316 saved'''

    # Use different loss parameters
    ###########
    # todo: checkSlope = 0.9
    ###########
    paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.9,
                                                  marketvec=np.ones(numTN + numSN), candneighnum=1000)

    numtargetdraws, numDataDraws = 5100, 5000

    # Find heuristic allocation first
    utilDict = {'method': 'weightsNodeDraw3linear'}

    numReps = 5
    utilMatList = []
    # set testMax to highest expected allocation for any one node
    testMax, testInt = 100, 10
    for rep in range(numReps):
        # Withdraw a subset of MCMC prior draws
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New loss draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        # Get new data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        # Get marginal utilities at each test node
        currMargUtilMat = GetMargUtilAtNodes(dictTemp.copy(), testMax, testInt, lossDict.copy(), utilDict.copy(),
                                             masterDraws=CSdict3['postSamples'], printUpdate=True)
        print(repr(currMargUtilMat))
        utilMatList.append(currMargUtilMat)
    '''22-MAR run
    utilMatList = [np.array([[0.        , 0.03075288, 0.0565451 , 0.0746321 , 0.09659966,
        0.11144783, 0.12698643, 0.14335713, 0.15641624, 0.1671006 ,
        0.18339319],
       [0.        , 0.06572466, 0.11378502, 0.14692445, 0.17433724,
        0.20018974, 0.22139891, 0.23813687, 0.25383738, 0.26964095,
        0.2851109 ],
       [0.        , 0.00955448, 0.0218536 , 0.0369972 , 0.05127719,
        0.06266118, 0.07858402, 0.09027435, 0.10331226, 0.11345728,
        0.12888646],
       [0.        , 0.00292594, 0.0114615 , 0.01887061, 0.03076254,
        0.03732542, 0.04456915, 0.05187669, 0.05964612, 0.06482972,
        0.06894663],
       [0.        , 0.09000419, 0.12589776, 0.1496807 , 0.16664233,
        0.18527906, 0.19807776, 0.21022732, 0.22143811, 0.23512691,
        0.24555148],
       [0.        , 0.05694585, 0.09415969, 0.12074849, 0.14017877,
        0.1554259 , 0.16750299, 0.17997499, 0.19218658, 0.19868762,
        0.20879519],
       [0.        , 0.07948077, 0.11859158, 0.14214012, 0.16206143,
        0.17784438, 0.19096196, 0.20068877, 0.21200407, 0.22172608,
        0.23045235],
       [0.        , 0.08349351, 0.11998393, 0.14773368, 0.17195863,
        0.19004298, 0.21192951, 0.22522178, 0.24097913, 0.25461054,
        0.26861671]]), np.array([[0.        , 0.0354193 , 0.05683372, 0.0782085 , 0.09669785,
        0.11198916, 0.13005225, 0.14684066, 0.15800112, 0.17123796,
        0.18228022],
       [0.        , 0.08267887, 0.12932618, 0.16144411, 0.191978  ,
        0.21518033, 0.23835322, 0.25391705, 0.26936901, 0.28910589,
        0.30199693],
       [0.        , 0.01891859, 0.03628884, 0.05092801, 0.06371052,
        0.07499908, 0.08887482, 0.09660051, 0.11284919, 0.12393414,
        0.13437873],
       [0.        , 0.02670743, 0.04023619, 0.05012686, 0.05686753,
        0.06523699, 0.07319036, 0.07950473, 0.08326539, 0.09111585,
        0.09381023],
       [0.        , 0.10111285, 0.1400115 , 0.16537856, 0.18529892,
        0.20262119, 0.21805694, 0.23197443, 0.24246257, 0.25466958,
        0.26796038],
       [0.        , 0.08938192, 0.1227288 , 0.14629541, 0.16152294,
        0.17283923, 0.185839  , 0.19607428, 0.20534831, 0.21425957,
        0.22112171],
       [0.        , 0.08959404, 0.12935687, 0.15235996, 0.17156702,
        0.18549643, 0.20298363, 0.21328721, 0.22521348, 0.23644074,
        0.24501391],
       [0.        , 0.09804337, 0.13615147, 0.16147446, 0.18473583,
        0.20338647, 0.21960846, 0.23761779, 0.25168935, 0.26537889,
        0.27647035]]), np.array([[0.        , 0.03513559, 0.06497641, 0.0882293 , 0.10427151,
        0.12455744, 0.13967437, 0.15553761, 0.16846617, 0.17881889,
        0.19385513],
       [0.        , 0.0828142 , 0.13571465, 0.17237   , 0.20211414,
        0.22464037, 0.24597637, 0.26405134, 0.28334982, 0.299907  ,
        0.31328436],
       [0.        , 0.0179283 , 0.03548164, 0.04584786, 0.05990511,
        0.07347835, 0.08506584, 0.09387507, 0.11142229, 0.11924694,
        0.12852292],
       [0.        , 0.02231515, 0.03931223, 0.04985942, 0.05796447,
        0.06589037, 0.0726527 , 0.07639682, 0.08806971, 0.08943918,
        0.09785195],
       [0.        , 0.10918078, 0.14608481, 0.17116112, 0.18998398,
        0.20613658, 0.22084961, 0.23323429, 0.24464794, 0.25703572,
        0.26354536],
       [0.        , 0.08590209, 0.11398255, 0.1344181 , 0.14981653,
        0.16140318, 0.17338802, 0.18165752, 0.18994836, 0.2000064 ,
        0.20690274],
       [0.        , 0.12069917, 0.1658702 , 0.19364535, 0.21143149,
        0.22928947, 0.24181656, 0.25333774, 0.26320408, 0.27221885,
        0.27904696],
       [0.        , 0.09974374, 0.13489977, 0.15909539, 0.18164873,
        0.20194355, 0.21698583, 0.23270839, 0.24835775, 0.26025708,
        0.2773961 ]]), np.array([[0.        , 0.03196404, 0.05760283, 0.08082335, 0.10020703,
        0.11885041, 0.13620186, 0.14771982, 0.16333645, 0.17561741,
        0.18685865],
       [0.        , 0.0680918 , 0.11569436, 0.15115868, 0.18104093,
        0.2073635 , 0.22808381, 0.24730727, 0.26399898, 0.28340489,
        0.2996577 ],
       [0.        , 0.02681898, 0.04852998, 0.06440018, 0.08369152,
        0.09539988, 0.1090851 , 0.12184843, 0.12987958, 0.14316872,
        0.15247667],
       [0.        , 0.01573695, 0.02839274, 0.03708781, 0.04523733,
        0.05410254, 0.06181502, 0.07061853, 0.07495894, 0.07920308,
        0.08275896],
       [0.        , 0.07078331, 0.11629243, 0.14702114, 0.16860698,
        0.19187182, 0.20756216, 0.22173809, 0.2340515 , 0.24491832,
        0.25908863],
       [0.        , 0.09205322, 0.12629978, 0.14458926, 0.16438938,
        0.17519035, 0.18756234, 0.19720792, 0.20605966, 0.21606601,
        0.22231906],
       [0.        , 0.0967696 , 0.13817887, 0.16422165, 0.18711883,
        0.20314808, 0.21525835, 0.23043861, 0.24159869, 0.24965842,
        0.25771903],
       [0.        , 0.08216984, 0.12600731, 0.15360078, 0.17772598,
        0.19768218, 0.21718576, 0.23357722, 0.24831773, 0.26046049,
        0.27608534]]), np.array([[0.        , 0.03471602, 0.05691319, 0.08416546, 0.10539777,
        0.12394867, 0.13983056, 0.15616778, 0.16871067, 0.18467579,
        0.18898689],
       [0.        , 0.06682056, 0.1182276 , 0.15460222, 0.18697697,
        0.21012775, 0.22805276, 0.25022063, 0.27215807, 0.2826246 ,
        0.30061884],
       [0.        , 0.01873002, 0.03693331, 0.05506479, 0.07187386,
        0.08828952, 0.10252277, 0.11629563, 0.12957653, 0.14289278,
        0.15120639],
       [0.        , 0.01001747, 0.02626064, 0.03763449, 0.04872725,
        0.05798881, 0.06516531, 0.06999929, 0.08018826, 0.0858269 ,
        0.09380951],
       [0.        , 0.07485467, 0.11746858, 0.14472599, 0.16704539,
        0.18309145, 0.19973911, 0.2154869 , 0.22485419, 0.23773503,
        0.25144521],
       [0.        , 0.08289615, 0.11494089, 0.13634728, 0.15149236,
        0.16650355, 0.17643583, 0.18629472, 0.19552417, 0.20491242,
        0.2144102 ],
       [0.        , 0.08451949, 0.11908208, 0.14377945, 0.16103633,
        0.17597314, 0.19321247, 0.20513748, 0.21709345, 0.2295761 ,
        0.23638159],
       [0.        , 0.07698847, 0.12337949, 0.15410276, 0.18050144,
        0.2038486 , 0.21785035, 0.23948056, 0.25477226, 0.27003047,
        0.28568715]])]
    '''
    # Get average utility matrix
    avgUtilMat = np.average(np.array(utilMatList), axis=0)
    '''22-MAR
    avgUtilMat = np.array([[0.        , 0.03359756, 0.05857425, 0.08121174, 0.10063477,
        0.1181587 , 0.13454909, 0.1499246 , 0.16298613, 0.17549013,
        0.18707481],
       [0.        , 0.07322602, 0.12254956, 0.15729989, 0.18728946,
        0.21150034, 0.23237301, 0.25072663, 0.26854265, 0.28493667,
        0.30013375],
       [0.        , 0.01839007, 0.03581747, 0.05064761, 0.06609164,
        0.0789656 , 0.09282651, 0.1037788 , 0.11740797, 0.12853997,
        0.13909423],
       [0.        , 0.01554059, 0.02913266, 0.03871584, 0.04791182,
        0.05610883, 0.06347851, 0.06967921, 0.07722568, 0.08208295,
        0.08743546],
       [0.        , 0.08918716, 0.12915102, 0.1555935 , 0.17551552,
        0.19380002, 0.20885712, 0.22253221, 0.23349086, 0.24589711,
        0.25751821],
       [0.        , 0.08143585, 0.11442234, 0.13647971, 0.15347999,
        0.16627244, 0.17814564, 0.18824189, 0.19781342, 0.2067864 ,
        0.21470978],
       [0.        , 0.09421261, 0.13421592, 0.15922931, 0.17864302,
        0.1943503 , 0.2088466 , 0.22057796, 0.23182275, 0.24192404,
        0.24972277],
       [0.        , 0.08808778, 0.12808439, 0.15520141, 0.17931412,
        0.19938076, 0.21671198, 0.23372115, 0.24882324, 0.26214749,
        0.27685113]])
    '''
    # Find allocation for sample budget
    allocArr = forwardAllocateWithBudget(avgUtilMat, int(sampBudget / testInt))
    designArr = allocArr / np.sum(allocArr, axis=0)
    # Get utility for this allocation at the sample budget
    utilDict.update({'method': 'weightsNodeDraw4linear'})
    compUtilList, unifUtilList, origUtilList = [], [], []
    numReps = 5
    for rep in range(numReps):
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New Bayes draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        print('Generating loss matrix...')
        tempLossMat = lf.lossMatSetBayesDraws(dictTemp['postSamples'], lossDict.copy(), lossDict['bayesDraws'])
        tempLossDict = lossDict.copy()
        tempLossDict.update({'lossMat': tempLossMat})
        newBayesDraws, newLossMat = lf.add_cand_neighbors(tempLossDict.copy(), CSdict3['postSamples'],
                                                          dictTemp['postSamples'])
        tempLossDict.update({'bayesDraws': newBayesDraws, 'lossMat': newLossMat})
        baseLoss = (np.sum(newLossMat, axis=1) / newLossMat.shape[1]).min()
        # Get a new set of data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        currCompUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[designArr],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Heuristic utility: ' + str(currCompUtil))
        compUtilList.append(currCompUtil)
        '''22-MAR
        compUtilList = [0.8454785840572736, 0.7962203640098924, 0.8779341831037586, 0.7972260625226433, 0.7756807232972012]
        '''
        # Find the equivalent uniform allocation
        currUnifUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[unifDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Uniform utility: ' + str(currUnifUtil))
        unifUtilList.append([currUnifUtil])
        unifAdd, contUnif, unifCount = 0, False, 0
        if currUnifUtil < currCompUtil:
            contUnif = True
        while contUnif:
            unifAdd += testInt
            print('Adding ' + str(unifAdd) + ' for uniform')
            currUnifUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[unifDes], numtests=sampBudget + unifAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currUnifUtil))
            unifUtilList[rep].append(currUnifUtil)
            if currUnifUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if unifCount < 3:
                    unifCount += 1
                else:
                    contUnif = False
        '''22-MAR
        unifUtilList = [[0.7615498336698927, 0.7951028598624803, 0.8194918550241082, 0.8440855363750845, 0.8759452668972023, 0.8898473102422275, 0.9180674279219145, 0.9517935007954614], [0.7181139791377253, 0.7442150364632365, 0.7821150692029217, 0.8126260464037092, 0.8424612620770429, 0.8524463784472176, 0.8840653785678767], [0.7809227818279787, 0.820605644614878, 0.8448793607316922, 0.8792420275428667, 0.904103420014164, 0.929169374792822, 0.9528576154189343], [0.7212088942785417, 0.7437816082396513, 0.7784281998851696, 0.8021146583206442, 0.8293593986639203, 0.8591987336943268, 0.8840533571232152], [0.6997504560036072, 0.7255330218677227, 0.7612888825064243, 0.7832493669920968, 0.812354522829442, 0.8428735923432971, 0.8658625778804936]]
        '''
        # Find the equivalent rudimentary allocation
        currOrigUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[origDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Rudimentary utility: ' + str(currOrigUtil))
        origUtilList.append([currOrigUtil])
        origAdd, contOrig, origCount = 0, False, 0
        if currOrigUtil < currCompUtil:
            contOrig = True
        while contOrig:
            origAdd += testInt * 3
            print('Adding ' + str(origAdd) + ' for rudimentary')
            currOrigUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[origDes], numtests=sampBudget + origAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currOrigUtil))
            origUtilList[rep].append(currOrigUtil)
            if currOrigUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if origCount < 3:
                    origCount += 1
                else:
                    contOrig = False
        '''22-MAR
        origUtilList = [[0.3338775309874147, 0.3893889317973125, 0.43435479751792716, 0.4943190573912868, 0.5368705213369585, 0.5999562125422018, 0.6547912450313023, 0.7280702311177634, 0.7527481455742198, 0.802214667256508, 0.84913974750947, 0.8844089636529899, 0.9405723702627999, 0.9671114503881038], [0.31278006416897197, 0.36097167923364726, 0.4197153026086169, 0.46064345923362326, 0.5113181008220375, 0.5609934166311348, 0.6167769980861526, 0.6751028514398625, 0.7096315671651374, 0.7690744097826818, 0.8107508296695438, 0.8562996733889432, 0.8864389770016392, 0.9277593934856663], [0.36889370430667423, 0.42266395146686353, 0.47434660564073416, 0.5274633267697615, 0.5856602325518581, 0.6246705854691408, 0.6830262039978421, 0.7461102418111629, 0.7857890011584465, 0.8125394825584227, 0.8722294419503509, 0.9204614519837113, 0.9617114115774732, 0.9842752670065673, 1.042885333581999], [0.2980442620336645, 0.34829195556315184, 0.39059194849703704, 0.45138690717995633, 0.4949244007042153, 0.5526880898812125, 0.6033477880294966, 0.6551781978604101, 0.691048164146876, 0.739568771901844, 0.7806538369992242, 0.825551626102905, 0.8748269680061873, 0.9130993446575335, 0.9441112186006548], [0.28559041049127654, 0.33662894671856947, 0.39363766715350224, 0.43237379539857645, 0.49430384129164473, 0.5491132457445249, 0.5970111408693013, 0.6687425566952991, 0.6978351568790879, 0.745950148779968, 0.7955593470786635, 0.8371116426925287, 0.880395719371899, 0.923677911236441]]
        '''
    compAvg = np.average(compUtilList)
    # Locate closest sample point for uniform and rudimentary to compAvg
    minListLen = np.min([len(i) for i in unifUtilList])
    unifUtilArr = np.array([i[:minListLen] for i in unifUtilList])
    unifAvgArr = np.average(unifUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(unifAvgArr.tolist()) if val > compAvg)
    unifSampSaved = round((compAvg - unifAvgArr[kInd - 1]) / (unifAvgArr[kInd] - unifAvgArr[kInd - 1]) * testInt) + (
            kInd - 1) * testInt
    print(unifSampSaved)
    '''21-MAR: 31 saved'''
    # Rudimentary
    minListLen = np.min([len(i) for i in origUtilList])
    origUtilArr = np.array([i[:minListLen] for i in origUtilList])
    origAvgArr = np.average(origUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(origAvgArr.tolist()) if val > compAvg)
    origSampSaved = round(
        (compAvg - origAvgArr[kInd - 1]) / (origAvgArr[kInd] - origAvgArr[kInd - 1]) * testInt * 3) + (
                            kInd - 1) * testInt * 3
    print(origSampSaved)
    '''21-MAR: 298 saved'''

    # Use different loss parameters
    ###########
    # todo: underWt = 1.
    ###########
    paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=1., riskthreshold=0.15, riskslope=0.6,
                                                  marketvec=np.ones(numTN + numSN), candneighnum=1000)


    numtargetdraws, numDataDraws = 5100, 5000

    # Find heuristic allocation first
    utilDict = {'method': 'weightsNodeDraw3linear'}

    numReps = 5
    utilMatList = []
    # set testMax to highest expected allocation for any one node
    testMax, testInt = 100, 10
    for rep in range(numReps):
        # Withdraw a subset of MCMC prior draws
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New loss draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        # Get new data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        # Get marginal utilities at each test node
        currMargUtilMat = GetMargUtilAtNodes(dictTemp.copy(), testMax, testInt, lossDict.copy(), utilDict.copy(),
                                             masterDraws=CSdict3['postSamples'], printUpdate=True)
        print(repr(currMargUtilMat))
        utilMatList.append(currMargUtilMat)
    '''22-MAR run
    utilMatList = [np.array([[0.        , 0.01088367, 0.02366804, 0.03522634, 0.04518001,
        0.05429115, 0.06170353, 0.06987887, 0.07758497, 0.08409471,
        0.0922609 ],
       [0.        , 0.02847574, 0.0513879 , 0.06912239, 0.08496897,
        0.10035419, 0.11355628, 0.12441351, 0.13470912, 0.14625519,
        0.15403402],
       [0.        , 0.01053215, 0.02151598, 0.03260515, 0.04422627,
        0.05547771, 0.06394887, 0.07529553, 0.0832292 , 0.09203173,
        0.0991013 ],
       [0.        , 0.00188317, 0.00548749, 0.00859108, 0.01135168,
        0.01409556, 0.01677248, 0.0195999 , 0.02161954, 0.02594214,
        0.02767769],
       [0.        , 0.03049416, 0.049828  , 0.0631742 , 0.07533541,
        0.08376526, 0.09311442, 0.10062907, 0.107332  , 0.11463545,
        0.11989065],
       [0.        , 0.02400419, 0.03740073, 0.04802847, 0.05587183,
        0.06527252, 0.07056265, 0.0763267 , 0.08197542, 0.08777925,
        0.09148122],
       [0.        , 0.03252395, 0.05365362, 0.06959628, 0.08016599,
        0.08969886, 0.09746171, 0.10527861, 0.11045559, 0.11641219,
        0.12135129],
       [0.        , 0.03136431, 0.05095531, 0.0662625 , 0.080798  ,
        0.09303958, 0.10274926, 0.11246973, 0.12249423, 0.13074985,
        0.13929619]]), 
        np.array([[0.        , 0.01107078, 0.02300029, 0.0353717 , 0.04388474,
        0.05204505, 0.05975575, 0.06891929, 0.07592609, 0.08245542,
        0.08990451],
       [0.        , 0.03050141, 0.05376334, 0.07503077, 0.09171872,
        0.10622402, 0.11751404, 0.13096928, 0.13915777, 0.15069926,
        0.15872302],
       [0.        , 0.00981586, 0.02065059, 0.03101965, 0.04074383,
        0.05165281, 0.0595977 , 0.06952978, 0.07626638, 0.08459924,
        0.09467509],
       [0.        , 0.00214673, 0.00480582, 0.00755866, 0.01111187,
        0.01370206, 0.01586223, 0.01876204, 0.02054649, 0.02307256,
        0.02583611],
       [0.        , 0.03174166, 0.05244254, 0.06575972, 0.07815617,
        0.08719497, 0.0957752 , 0.10338809, 0.11054561, 0.11693476,
        0.12291782],
       [0.        , 0.02488944, 0.03989517, 0.05059822, 0.05879231,
        0.06766035, 0.07483272, 0.08171699, 0.0861234 , 0.09120108,
        0.09633551],
       [0.        , 0.0302982 , 0.0510324 , 0.06588003, 0.07705555,
        0.08644201, 0.09436592, 0.10063807, 0.10678855, 0.1129641 ,
        0.11792721],
       [0.        , 0.02212158, 0.04253936, 0.05790151, 0.07136883,
        0.08418088, 0.09417643, 0.10260827, 0.11313299, 0.12310535,
        0.1300502 ]]),
        np.array([[0.        , 0.00919342, 0.01986591, 0.03190652, 0.04259505,
        0.04991739, 0.05766652, 0.06570681, 0.0719042 , 0.07835804,
        0.08454289],
       [0.        , 0.02440816, 0.04470776, 0.06281298, 0.08046134,
        0.09369081, 0.10632461, 0.11912579, 0.12893504, 0.1372843 ,
        0.14728741],
       [0.        , 0.00569074, 0.01543616, 0.02550775, 0.03485174,
        0.04475004, 0.05449078, 0.06378026, 0.07022076, 0.08111976,
        0.08809707],
       [0.        , 0.00170505, 0.00374647, 0.00659166, 0.01011973,
        0.01265864, 0.01471206, 0.01667299, 0.01990643, 0.02144041,
        0.02341426],
       [0.        , 0.02398414, 0.04117571, 0.05505616, 0.06724893,
        0.07697693, 0.08557795, 0.09478825, 0.10103732, 0.10862509,
        0.11434237],
       [0.        , 0.0228718 , 0.0375308 , 0.04822571, 0.05731266,
        0.063901  , 0.07061442, 0.07549924, 0.08167641, 0.08505161,
        0.09018776],
       [0.        , 0.02513935, 0.04520408, 0.0577177 , 0.0683569 ,
        0.07837852, 0.0847792 , 0.09189106, 0.0993334 , 0.1039948 ,
        0.10879921],
       [0.        , 0.02136997, 0.04054589, 0.05529073, 0.0690352 ,
        0.08078239, 0.09134324, 0.10204092, 0.11115004, 0.12026925,
        0.12848079]]),
        np.array([[0.        , 0.01039543, 0.02256671, 0.03378684, 0.04394314,
        0.05392682, 0.06069337, 0.06859636, 0.07774013, 0.08339359,
        0.08985891],
       [0.        , 0.02914781, 0.05202363, 0.0738285 , 0.08981818,
        0.10410924, 0.11836312, 0.13087746, 0.14027163, 0.15074776,
        0.15993027],
       [0.        , 0.01071109, 0.0215703 , 0.031148  , 0.04163808,
        0.05503134, 0.06205651, 0.07170679, 0.08095795, 0.08876356,
        0.09605018],
       [0.        , 0.00180101, 0.00504892, 0.0081293 , 0.0104461 ,
        0.01301865, 0.01585576, 0.01900689, 0.02141951, 0.02365678,
        0.02593692],
       [0.        , 0.02888871, 0.04795943, 0.06193616, 0.07344362,
        0.08219015, 0.09102029, 0.09927703, 0.10544726, 0.11178248,
        0.11950148],
       [0.        , 0.02485113, 0.04132151, 0.05101368, 0.06034551,
        0.06830007, 0.07563689, 0.07996753, 0.08639545, 0.09084837,
        0.09560016],
       [0.        , 0.02905179, 0.05008968, 0.06386594, 0.07478612,
        0.08452121, 0.09246634, 0.09908938, 0.10537702, 0.11100315,
        0.11661527],
       [0.        , 0.028703  , 0.04959113, 0.06657348, 0.08088227,
        0.09397491, 0.10519616, 0.11411599, 0.12224504, 0.13211056,
        0.14173876]]),
        np.array([[0.        , 0.01814277, 0.03145738, 0.04374345, 0.05372269,
        0.0629357 , 0.07153863, 0.07820898, 0.08637542, 0.09180568,
        0.09895402],
       [0.        , 0.03630651, 0.06131392, 0.08051863, 0.09582164,
        0.11221689, 0.12406721, 0.13602584, 0.14612171, 0.1554203 ,
        0.16321455],
       [0.        , 0.01515037, 0.02813505, 0.03986089, 0.05153592,
        0.06062684, 0.07144059, 0.0805404 , 0.08846696, 0.09598719,
        0.10578462],
       [0.        , 0.00534265, 0.01035942, 0.01432403, 0.0182365 ,
        0.02194153, 0.02551831, 0.02790565, 0.03081279, 0.03350372,
        0.03614253],
       [0.        , 0.04155905, 0.06192909, 0.07619405, 0.08723579,
        0.09634616, 0.10449396, 0.11194988, 0.11927403, 0.12469818,
        0.13243729],
       [0.        , 0.0320574 , 0.0496676 , 0.06283267, 0.07176745,
        0.07853427, 0.08513404, 0.0915232 , 0.09578596, 0.10150947,
        0.1070271 ],
       [0.        , 0.04073478, 0.06195811, 0.07678302, 0.08745174,
        0.09692602, 0.10480846, 0.11128112, 0.11782259, 0.12389517,
        0.12909392],
       [0.        , 0.0361526 , 0.05839154, 0.07592405, 0.08833617,
        0.10106856, 0.11350123, 0.12072861, 0.13080787, 0.13956721,
        0.14896624]])]
    '''
    # Get average utility matrix
    avgUtilMat = np.average(np.array(utilMatList), axis=0)
    '''22-MAR
    avgUtilMat = np.array([[0.        , 0.01193721, 0.02411166, 0.03600697, 0.04586512,
        0.05462322, 0.06227156, 0.07026206, 0.07790616, 0.08402149,
        0.09110425],
       [0.        , 0.02976793, 0.05263931, 0.07226266, 0.08855777,
        0.10331903, 0.11596505, 0.12828238, 0.13783905, 0.14808136,
        0.15663786],
       [0.        , 0.01038004, 0.02146161, 0.03202829, 0.04259917,
        0.05350775, 0.06230689, 0.07217055, 0.07982825, 0.0885003 ,
        0.09674165],
       [0.        , 0.00257572, 0.00588963, 0.00903895, 0.01225318,
        0.01508329, 0.01774417, 0.02038949, 0.02286095, 0.02552312,
        0.0278015 ],
       [0.        , 0.03133355, 0.05066695, 0.06442406, 0.07628398,
        0.08529469, 0.09399636, 0.10200647, 0.10872724, 0.11533519,
        0.12181792],
       [0.        , 0.02573479, 0.04116316, 0.05213975, 0.06081795,
        0.06873364, 0.07535614, 0.08100673, 0.08639133, 0.09127796,
        0.09612635],
       [0.        , 0.03154961, 0.05238758, 0.06676859, 0.07756326,
        0.08719332, 0.09477633, 0.10163565, 0.10795543, 0.11365388,
        0.11875738],
       [0.        , 0.02794229, 0.04840465, 0.06439045, 0.07808409,
        0.09060926, 0.10139326, 0.1103927 , 0.11996603, 0.12916045,
        0.13770644]])
    '''
    # Find allocation for sample budget
    allocArr = forwardAllocateWithBudget(avgUtilMat, int(sampBudget / testInt))
    designArr = allocArr / np.sum(allocArr, axis=0)
    # Get utility for this allocation at the sample budget
    utilDict.update({'method': 'weightsNodeDraw4linear'})
    compUtilList, unifUtilList, origUtilList = [], [], []
    numReps = 5
    for rep in range(numReps):
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New Bayes draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        print('Generating loss matrix...')
        tempLossMat = lf.lossMatSetBayesDraws(dictTemp['postSamples'], lossDict.copy(), lossDict['bayesDraws'])
        tempLossDict = lossDict.copy()
        tempLossDict.update({'lossMat': tempLossMat})
        newBayesDraws, newLossMat = lf.add_cand_neighbors(tempLossDict.copy(), CSdict3['postSamples'],
                                                          dictTemp['postSamples'])
        tempLossDict.update({'bayesDraws': newBayesDraws, 'lossMat': newLossMat})
        baseLoss = (np.sum(newLossMat, axis=1) / newLossMat.shape[1]).min()
        # Get a new set of data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        currCompUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[designArr],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Heuristic utility: ' + str(currCompUtil))
        compUtilList.append(currCompUtil)
        '''22-MAR
        compUtilList = [0.40709412545719825, 0.3881939297825612, 0.40827933747157785, 0.3955408687446065, 0.3923314455965512]
        '''
        # Find the equivalent uniform allocation
        currUnifUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[unifDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Uniform utility: ' + str(currUnifUtil))
        unifUtilList.append([currUnifUtil])
        unifAdd, contUnif, unifCount = 0, False, 0
        if currUnifUtil < currCompUtil:
            contUnif = True
        while contUnif:
            unifAdd += testInt
            print('Adding ' + str(unifAdd) + ' for uniform')
            currUnifUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[unifDes], numtests=sampBudget + unifAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currUnifUtil))
            unifUtilList[rep].append(currUnifUtil)
            if currUnifUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if unifCount < 3:
                    unifCount += 1
                else:
                    contUnif = False
        '''22-MAR
        unifUtilList = [[0.3561362225611773, 0.3713155483638251, 0.3896429448665333, 0.40016099545529715, 0.4192889376567872, 0.4355385114490129, 0.44783680670621084, 0.46138188777150146], [0.3449490689468906, 0.3629529760229606, 0.38232956939168816, 0.388991947218307, 0.40983118963882315, 0.41469313155617415, 0.432837638675879], [0.35495346352171775, 0.36634865856849697, 0.38989440038833667, 0.3993761481612348, 0.41693523295502977, 0.43608388379505403, 0.44165039868313216, 0.45319830413503315], [0.34492786985498425, 0.3593748920540074, 0.3792476024814979, 0.3954143483134216, 0.4125392748625054, 0.41898169104872673, 0.4351282545346542, 0.45400289196088206], [0.3451781756821901, 0.36448343423101615, 0.3786782287761363, 0.39100276019103997, 0.408922220241684, 0.4248373687904574, 0.4384224391302778, 0.45087706702934804]]
        '''
        # Find the equivalent rudimentary allocation
        currOrigUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[origDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Rudimentary utility: ' + str(currOrigUtil))
        origUtilList.append([currOrigUtil])
        origAdd, contOrig, origCount = 0, False, 0
        if currOrigUtil < currCompUtil:
            contOrig = True
        while contOrig:
            origAdd += testInt * 3
            print('Adding ' + str(origAdd) + ' for rudimentary')
            currOrigUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[origDes], numtests=sampBudget + origAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currOrigUtil))
            origUtilList[rep].append(currOrigUtil)
            if currOrigUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if origCount < 3:
                    origCount += 1
                else:
                    contOrig = False
        '''22-MAR
        origUtilList = [[0.19081679420743258, 0.21687746776801098, 0.24696404313893172, 0.27549822154714576, 0.3067589644873667, 0.33269825573657896, 0.35536312483282373, 0.38898721223875277, 0.4076121717358929, 0.4308990323839679, 0.452922178046691, 0.47149870174904707], [0.18086051610033826, 0.2059940333521577, 0.2403221392609829, 0.2677845236831635, 0.29327264384583374, 0.3232873084908916, 0.3459593709655435, 0.3742341083460694, 0.39409723912733075, 0.41501074682035655, 0.43900737540200696, 0.4614278228383144], [0.1911086660388468, 0.21977023882595992, 0.24734547441233912, 0.27312912896551045, 0.30514160503587906, 0.3270478421901879, 0.35791554673033454, 0.38564570395885633, 0.4048247476079052, 0.427397612607779, 0.45461324827753, 0.4676419701481218, 0.49030845182785776], [0.18384727088020947, 0.210851949379665, 0.24228786600236707, 0.2660269848336094, 0.29772674372267227, 0.3242724887182218, 0.3466754793469329, 0.37246501895416473, 0.39479507583811246, 0.4240965165649875, 0.4496017558640131, 0.4647584650248533, 0.4842961526526204], [0.18379618634452943, 0.21081775736059805, 0.23956887814534, 0.26696306279972193, 0.29405854698407463, 0.3190004499387784, 0.34688882611744054, 0.38134566851585006, 0.39202656633310995, 0.42448624398223034, 0.44019231059087827, 0.4637814684420174, 0.4837667079338268]]
        '''
    compAvg = np.average(compUtilList)
    # Locate closest sample point for uniform and rudimentary to compAvg
    minListLen = np.min([len(i) for i in unifUtilList])
    unifUtilArr = np.array([i[:minListLen] for i in unifUtilList])
    unifAvgArr = np.average(unifUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(unifAvgArr.tolist()) if val > compAvg)
    unifSampSaved = round((compAvg - unifAvgArr[kInd - 1]) / (unifAvgArr[kInd] - unifAvgArr[kInd - 1]) * testInt) + (
            kInd - 1) * testInt
    print('Saved vs uniform: ' + str(unifSampSaved))
    '''21-MAR: 32 saved'''
    # Rudimentary
    minListLen = np.min([len(i) for i in origUtilList])
    origUtilArr = np.array([i[:minListLen] for i in origUtilList])
    origAvgArr = np.average(origUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(origAvgArr.tolist()) if val > compAvg)
    origSampSaved = round(
        (compAvg - origAvgArr[kInd - 1]) / (origAvgArr[kInd] - origAvgArr[kInd - 1]) * testInt * 3) + (
                            kInd - 1) * testInt * 3
    print('Saved vs rudimentary: ' + str(origSampSaved))
    '''21-MAR: 239 saved'''

    # Use different loss parameters
    ###########
    # todo: underWt = 10.
    ###########
    paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=10., riskthreshold=0.15, riskslope=0.6,
                                                  marketvec=np.ones(numTN + numSN), candneighnum=1000)

    numtargetdraws, numDataDraws = 5100, 5000

    # Find heuristic allocation first
    utilDict = {'method': 'weightsNodeDraw3linear'}

    numReps = 20
    utilMatList = []
    # set testMax to highest expected allocation for any one node
    testMax, testInt = 100, 10
    for rep in range(numReps):
        # Withdraw a subset of MCMC prior draws
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New loss draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        # Get new data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        # Get marginal utilities at each test node
        currMargUtilMat = GetMargUtilAtNodes(dictTemp.copy(), testMax, testInt, lossDict.copy(), utilDict.copy(),
                                             masterDraws=CSdict3['postSamples'], printUpdate=True)
        print(repr(currMargUtilMat))
        utilMatList.append(currMargUtilMat)
    '''23-MAR run
    utilMatList = [np.array([[ 0.        ,  0.00120468,  0.00656565,  0.01551929,  0.03766449,
         0.05080222,  0.06784914,  0.08344739,  0.10081684,  0.11529759,
         0.13702493],
       [ 0.        ,  0.00462389,  0.02451324,  0.04813466,  0.08802436,
         0.10411307,  0.1389753 ,  0.15649096,  0.18969085,  0.20499551,
         0.23157762],
       [ 0.        ,  0.00271516,  0.00033268, -0.00217927,  0.0085513 ,
         0.02403054,  0.02085072,  0.03616679,  0.03933638,  0.05248315,
         0.05614416],
       [ 0.        , -0.00108056, -0.00362558,  0.00121604,  0.00389859,
         0.00704794, -0.00368827, -0.00040903,  0.00631218,  0.00600609,
         0.01306459],
       [ 0.        ,  0.04756594,  0.07060646,  0.09405848,  0.1129362 ,
         0.12769747,  0.15165531,  0.15996792,  0.18152856,  0.19256876,
         0.21394517],
       [ 0.        ,  0.023096  ,  0.06355345,  0.0973168 ,  0.1157059 ,
         0.14155211,  0.1522775 ,  0.16362234,  0.18334054,  0.19281525,
         0.19723289],
       [ 0.        ,  0.0595386 ,  0.12013513,  0.16613116,  0.19424526,
         0.21451091,  0.2392563 ,  0.26164562,  0.27340608,  0.28751729,
         0.30181918],
       [ 0.        ,  0.06775282,  0.10142666,  0.13179684,  0.14941416,
         0.17097894,  0.19896088,  0.21618913,  0.24161841,  0.25204073,
         0.27661429]]), 
        np.array([[0.        , 0.03313953, 0.07354083, 0.09998127, 0.12749158,
        0.14685468, 0.17277861, 0.19103047, 0.20764294, 0.23145619,
        0.24860821],
       [0.        , 0.03228439, 0.06820891, 0.1103673 , 0.14470328,
        0.17237899, 0.20826166, 0.23377392, 0.26271626, 0.281198  ,
        0.30489594],
       [0.        , 0.02386509, 0.04576465, 0.07650619, 0.08901712,
        0.11192525, 0.13170355, 0.14183803, 0.16298038, 0.17425721,
        0.1890034 ],
       [0.        , 0.02518086, 0.04621282, 0.06150915, 0.07875085,
        0.09698066, 0.10611908, 0.11359236, 0.12638717, 0.13739806,
        0.13766392],
       [0.        , 0.14953149, 0.20230311, 0.23919621, 0.26898233,
        0.2964971 , 0.31403831, 0.33542851, 0.35716539, 0.37418597,
        0.39425657],
       [0.        , 0.07314711, 0.11344224, 0.14533419, 0.16754279,
        0.19123891, 0.20749838, 0.22433617, 0.2360672 , 0.25039213,
        0.26457748],
       [0.        , 0.10447577, 0.1555549 , 0.18822103, 0.22069445,
        0.24385141, 0.26417857, 0.28561584, 0.29813949, 0.30947317,
        0.32725679],
       [0.        , 0.08457057, 0.15300597, 0.19845206, 0.23409496,
        0.26977831, 0.29956478, 0.32352341, 0.34548088, 0.37058166,
        0.38939264]]),
        np.array([[0.        , 0.00139221, 0.01903388, 0.03458285, 0.04724059,
        0.07199024, 0.0879437 , 0.11027449, 0.12787833, 0.14666119,
        0.15992047],
       [0.        , 0.00306777, 0.02986245, 0.06506521, 0.08755597,
        0.11805792, 0.1517383 , 0.17579392, 0.18910241, 0.21729688,
        0.2388932 ],
       [0.        , 0.00064503, 0.00268792, 0.00817402, 0.02932565,
        0.02455109, 0.04337812, 0.05649437, 0.06580987, 0.07819027,
        0.08913803],
       [0.        , 0.00243112, 0.00947748, 0.00427011, 0.01468392,
        0.02908101, 0.02198425, 0.03338106, 0.03183229, 0.04096667,
        0.04172231],
       [0.        , 0.03553166, 0.06038439, 0.08015674, 0.10575054,
        0.12678599, 0.14755386, 0.16286902, 0.17367467, 0.18850434,
        0.21048463],
       [0.        , 0.04460167, 0.08120985, 0.12323763, 0.14193709,
        0.16981131, 0.18892928, 0.20140503, 0.215525  , 0.23639733,
        0.25004333],
       [0.        , 0.05792356, 0.11944427, 0.1646357 , 0.19644711,
        0.22405651, 0.24634944, 0.26239254, 0.27910234, 0.30079058,
        0.31495664],
       [0.        , 0.10642332, 0.17336675, 0.21057376, 0.24135685,
        0.26147421, 0.27443612, 0.296224  , 0.31946734, 0.32496792,
        0.34868003]]),
        np.array([[0.        , 0.05863346, 0.10282637, 0.13231025, 0.17115664,
        0.19181996, 0.20651617, 0.24341473, 0.25685445, 0.27049812,
        0.29599277],
       [0.        , 0.03861315, 0.08361541, 0.13348608, 0.16865482,
        0.2053365 , 0.23085085, 0.26737741, 0.29042047, 0.31981719,
        0.33195722],
       [0.        , 0.03106147, 0.0602737 , 0.0854729 , 0.11202132,
        0.13458499, 0.15847707, 0.16710974, 0.20097755, 0.21773769,
        0.23379661],
       [0.        , 0.02206453, 0.03964952, 0.06066245, 0.0746943 ,
        0.09820347, 0.10419519, 0.1138271 , 0.12835275, 0.1302199 ,
        0.14301024],
       [0.        , 0.20082311, 0.24528342, 0.28148543, 0.30494862,
        0.32518309, 0.34737601, 0.37089341, 0.38939056, 0.40751854,
        0.42395229],
       [0.        , 0.07230572, 0.12927154, 0.16969462, 0.19740849,
        0.22234616, 0.24064166, 0.26277421, 0.28242441, 0.28645109,
        0.30427506],
       [0.        , 0.11351295, 0.1700273 , 0.21155444, 0.24011632,
        0.26639412, 0.28660026, 0.30582847, 0.3257346 , 0.33412628,
        0.35319983],
       [0.        , 0.1134575 , 0.19526986, 0.25027178, 0.29351039,
        0.32677088, 0.35447656, 0.38469364, 0.41507009, 0.43698795,
        0.46141611]]),
        np.array([[0.        , 0.03572665, 0.0783538 , 0.10109161, 0.13337982,
        0.16169645, 0.18474166, 0.19829178, 0.22416227, 0.24111931,
        0.25532316],
       [0.        , 0.04870739, 0.08728784, 0.11537056, 0.14029444,
        0.1756372 , 0.20099288, 0.22917533, 0.25256793, 0.27570327,
        0.2904192 ],
       [0.        , 0.02404566, 0.05292013, 0.07530973, 0.09607215,
        0.11403674, 0.13944292, 0.1520435 , 0.17550348, 0.18903922,
        0.20385438],
       [0.        , 0.03114307, 0.05347605, 0.07668457, 0.08875655,
        0.10454422, 0.11462093, 0.12246253, 0.13558462, 0.14042855,
        0.15133396],
       [0.        , 0.14325183, 0.19888008, 0.23689014, 0.26492813,
        0.29372858, 0.31652841, 0.33804898, 0.3539739 , 0.37585011,
        0.39250157],
       [0.        , 0.09348225, 0.14527721, 0.18056193, 0.21038689,
        0.23103906, 0.24477909, 0.26743163, 0.27926393, 0.29251721,
        0.31375712],
       [0.        , 0.12291258, 0.18337087, 0.22583915, 0.25506331,
        0.27966958, 0.30121127, 0.31623238, 0.32695004, 0.34765429,
        0.35289565],
       [0.        , 0.04835235, 0.09483808, 0.14406501, 0.17599626,
        0.21429704, 0.24475003, 0.27582559, 0.2980499 , 0.32775888,
        0.34581681]]),
        np.array([[0.        , 0.05376667, 0.10311913, 0.13651346, 0.16314932,
        0.19237238, 0.21290491, 0.23284194, 0.25214556, 0.27323666,
        0.28976281],
       [0.        , 0.05219546, 0.10003353, 0.13767871, 0.18044456,
        0.20592443, 0.23760392, 0.26583101, 0.28959246, 0.31537423,
        0.33828965],
       [0.        , 0.03070256, 0.05998629, 0.09078512, 0.11773644,
        0.14045454, 0.16264295, 0.18555271, 0.20283941, 0.21939406,
        0.24226982],
       [0.        , 0.03120702, 0.05890336, 0.07833455, 0.09960008,
        0.1170816 , 0.12939652, 0.14346861, 0.15345403, 0.16556603,
        0.1744767 ],
       [0.        , 0.21315336, 0.27711015, 0.30667501, 0.33630608,
        0.36432197, 0.37866223, 0.40270705, 0.42230374, 0.43168675,
        0.45937293],
       [0.        , 0.09843084, 0.1601419 , 0.19837331, 0.22715259,
        0.25079961, 0.27497146, 0.29124722, 0.30481952, 0.32406521,
        0.33576163],
       [0.        , 0.1563554 , 0.22129522, 0.26520678, 0.29792175,
        0.3201672 , 0.33885261, 0.35541445, 0.37170455, 0.38868968,
        0.40049498],
       [0.        , 0.12944149, 0.21141173, 0.26201055, 0.29997128,
        0.32918388, 0.36179296, 0.38856219, 0.41080412, 0.43771884,
        0.4579797 ]]),
        np.array([[0.        , 0.00770157, 0.03266161, 0.05535978, 0.08179847,
        0.10519675, 0.13237836, 0.14590382, 0.16279405, 0.1879829 ,
        0.20679289],
       [0.        , 0.00325573, 0.037566  , 0.06830022, 0.10717671,
        0.14401101, 0.17206662, 0.2078065 , 0.22677839, 0.26155415,
        0.2834657 ],
       [0.        , 0.00226681, 0.01502894, 0.03186391, 0.04404903,
        0.06308075, 0.08168572, 0.0940591 , 0.11043483, 0.12696395,
        0.13721356],
       [0.        , 0.01345735, 0.02128641, 0.03789736, 0.04677318,
        0.05617818, 0.05799564, 0.0735612 , 0.08020328, 0.09366539,
        0.09280838],
       [0.        , 0.11953525, 0.16417364, 0.18844598, 0.21679802,
        0.24260773, 0.26119381, 0.27291486, 0.29014481, 0.30682068,
        0.31905617],
       [0.        , 0.01352889, 0.04477992, 0.07320886, 0.09499985,
        0.11245572, 0.13191006, 0.15245415, 0.16330565, 0.18220056,
        0.19810975],
       [0.        , 0.01104141, 0.04023908, 0.07169041, 0.10076356,
        0.11639766, 0.13961671, 0.15799805, 0.17442989, 0.1850612 ,
        0.197637  ],
       [0.        , 0.05549393, 0.11823078, 0.15378514, 0.18863343,
        0.22079583, 0.24108493, 0.26920554, 0.29285782, 0.32059195,
        0.32983407]]),
        np.array([[0.        , 0.01702721, 0.04938746, 0.08179545, 0.10121542,
        0.12401175, 0.14677783, 0.16984229, 0.18652802, 0.19394781,
        0.22309352],
       [0.        , 0.00841891, 0.04233425, 0.0843976 , 0.1240208 ,
        0.16375507, 0.19438608, 0.22622229, 0.25094944, 0.27685202,
        0.29818069],
       [0.        , 0.00769767, 0.01448983, 0.03777543, 0.05130851,
        0.06817901, 0.08613603, 0.0960055 , 0.11301918, 0.13529931,
        0.14326072],
       [0.        , 0.01967123, 0.0365611 , 0.05391739, 0.06385921,
        0.07379105, 0.07928046, 0.09059175, 0.10016725, 0.10824891,
        0.10801388],
       [0.        , 0.10071116, 0.13988405, 0.16905553, 0.19960629,
        0.21806099, 0.23689085, 0.25252148, 0.26518859, 0.28988831,
        0.30241526],
       [0.        , 0.02250508, 0.06545604, 0.09818856, 0.11914778,
        0.1428621 , 0.15462625, 0.18190699, 0.19422659, 0.21309155,
        0.22332715],
       [0.        , 0.01411235, 0.04027322, 0.06563583, 0.09107252,
        0.114074  , 0.13905073, 0.15421677, 0.16540505, 0.18995204,
        0.19921775],
       [0.        , 0.06623112, 0.12124336, 0.16145975, 0.18971376,
        0.21734923, 0.24585194, 0.27052218, 0.29654858, 0.30749151,
        0.32954324]]),
        np.array([[0.        , 0.06036062, 0.11635292, 0.1479608 , 0.18132908,
        0.20743647, 0.23578363, 0.25410035, 0.28046125, 0.29630725,
        0.30959561],
       [0.        , 0.05793396, 0.10331497, 0.14334054, 0.18061849,
        0.20835498, 0.24207603, 0.26979677, 0.2906646 , 0.31722512,
        0.33498508],
       [0.        , 0.03395845, 0.07243307, 0.10043778, 0.12558667,
        0.15037627, 0.17289767, 0.19502859, 0.21160592, 0.2319559 ,
        0.24625542],
       [0.        , 0.0326921 , 0.05989376, 0.08156355, 0.09771958,
        0.11247129, 0.12305181, 0.13700933, 0.15860186, 0.16515768,
        0.1745921 ],
       [0.        , 0.20546795, 0.26334283, 0.29633629, 0.32661193,
        0.35664604, 0.37243026, 0.39504945, 0.41346979, 0.43134031,
        0.44591939],
       [0.        , 0.10509544, 0.16694691, 0.20706325, 0.2342641 ,
        0.2623219 , 0.28241504, 0.30390156, 0.31315435, 0.3330749 ,
        0.34975154],
       [0.        , 0.18060561, 0.23777638, 0.27199263, 0.30316308,
        0.32167852, 0.34362373, 0.36223702, 0.37149814, 0.38291811,
        0.39868928],
       [0.        , 0.11869708, 0.20055018, 0.25525819, 0.29356718,
        0.32513762, 0.35632719, 0.38198868, 0.41344002, 0.43201841,
        0.4583221 ]]),
        np.array([[0.        , 0.04474766, 0.08464523, 0.1141524 , 0.14101735,
        0.16208469, 0.18653068, 0.20437233, 0.22153497, 0.23369672,
        0.25398875],
       [0.        , 0.03667804, 0.07240557, 0.10199467, 0.13108398,
        0.1627223 , 0.1888458 , 0.20844786, 0.23555631, 0.25795483,
        0.28153754],
       [0.        , 0.02602991, 0.05442282, 0.0789273 , 0.09861853,
        0.1218746 , 0.13513853, 0.15494553, 0.17725544, 0.1832802 ,
        0.20406198],
       [0.        , 0.02986242, 0.0543273 , 0.07267712, 0.08991887,
        0.1077383 , 0.12155521, 0.12901786, 0.14065439, 0.14721779,
        0.16052934],
       [0.        , 0.14281471, 0.20015847, 0.23409114, 0.25969498,
        0.27889744, 0.29748905, 0.31685641, 0.3312501 , 0.3426454 ,
        0.35264588],
       [0.        , 0.0799959 , 0.10627601, 0.12671243, 0.14747415,
        0.16233389, 0.17979646, 0.19055269, 0.20954689, 0.22048412,
        0.23977668],
       [0.        , 0.09391213, 0.13044488, 0.16183761, 0.18798826,
        0.20561127, 0.22863851, 0.24888509, 0.26244796, 0.27300932,
        0.28679281],
       [0.        , 0.02532229, 0.07634241, 0.11123244, 0.15091152,
        0.18034649, 0.20942599, 0.2348326 , 0.26331566, 0.27969367,
        0.29725997]]),
        np.array([[0.        , 0.03142422, 0.06486296, 0.10291723, 0.14116081,
        0.17551774, 0.19783036, 0.22628256, 0.25495802, 0.2881009 ,
        0.30301989],
       [0.        , 0.10927979, 0.18962621, 0.24590258, 0.29303523,
        0.33180464, 0.36152419, 0.39464595, 0.41594735, 0.44003285,
        0.46492983],
       [0.        , 0.00924678, 0.02821577, 0.04859639, 0.06901854,
        0.08681848, 0.11373008, 0.13446964, 0.15612516, 0.17157582,
        0.19257803],
       [0.        , 0.00886661, 0.02030239, 0.03162162, 0.04285219,
        0.05020692, 0.05696601, 0.07522763, 0.07886471, 0.08940046,
        0.09545104],
       [0.        , 0.09815141, 0.15765332, 0.19864342, 0.23781899,
        0.26720321, 0.29285698, 0.31730411, 0.3463629 , 0.36381531,
        0.38079209],
       [0.        , 0.18830949, 0.24358363, 0.27281776, 0.30067747,
        0.3212423 , 0.33825845, 0.36173765, 0.37199221, 0.39396921,
        0.40520341],
       [0.        , 0.20600525, 0.28659284, 0.33854391, 0.37063065,
        0.3928698 , 0.42325399, 0.44112645, 0.45964889, 0.47606716,
        0.4931827 ],
       [0.        , 0.20242929, 0.2925174 , 0.34752582, 0.38163221,
        0.41656863, 0.44895999, 0.47058623, 0.50056362, 0.52799451,
        0.54814226]]),
        np.array([[0.        , 0.05225314, 0.09806765, 0.14052986, 0.17193756,
        0.19386782, 0.22196125, 0.24386311, 0.2669571 , 0.28066832,
        0.30551034],
       [0.        , 0.04645987, 0.09836442, 0.14611256, 0.18564199,
        0.21062697, 0.24563567, 0.27547419, 0.29782256, 0.32201786,
        0.34395411],
       [0.        , 0.02883084, 0.06059895, 0.08979253, 0.11338253,
        0.14242253, 0.15988901, 0.17911276, 0.20112847, 0.21869236,
        0.23528664],
       [0.        , 0.02606623, 0.04843602, 0.0730944 , 0.09010673,
        0.10942394, 0.11823592, 0.13678685, 0.14405748, 0.15537261,
        0.16559676],
       [0.        , 0.20422929, 0.26106099, 0.29599749, 0.32748012,
        0.35012176, 0.37031403, 0.39423722, 0.4121728 , 0.42588163,
        0.44662261],
       [0.        , 0.09211562, 0.15111817, 0.19511498, 0.2225658 ,
        0.25117175, 0.27547912, 0.28934456, 0.30783317, 0.32644268,
        0.33628961],
       [0.        , 0.15211478, 0.20477672, 0.23896691, 0.26766846,
        0.28646477, 0.30744308, 0.32069842, 0.34057127, 0.35061127,
        0.36186334],
       [0.        , 0.10966179, 0.19277521, 0.23636164, 0.27959567,
        0.31462503, 0.34072509, 0.36584727, 0.39530648, 0.41712954,
        0.45106185]]),
        np.array([[0.        , 0.01796104, 0.06397207, 0.10132702, 0.13841642,
        0.17088235, 0.20677617, 0.23844284, 0.26176382, 0.28000414,
        0.31120991],
       [0.        , 0.06419926, 0.14741114, 0.2102626 , 0.26419057,
        0.31306956, 0.35076908, 0.38470052, 0.41447988, 0.44845973,
        0.48076405],
       [0.        , 0.01822019, 0.04338464, 0.0686211 , 0.10309939,
        0.12061597, 0.15394979, 0.17644259, 0.20117321, 0.22239825,
        0.24184112],
       [0.        , 0.00220347, 0.01453781, 0.0269536 , 0.04528584,
        0.05209016, 0.0726768 , 0.08319845, 0.09331105, 0.10640713,
        0.12082745],
       [0.        , 0.15968181, 0.23020164, 0.27587376, 0.31843527,
        0.35226238, 0.38050398, 0.40735882, 0.43691675, 0.45677722,
        0.47543152],
       [0.        , 0.20787008, 0.27298804, 0.31758101, 0.3529215 ,
        0.36876846, 0.40042284, 0.41613465, 0.44085519, 0.45382621,
        0.46134013],
       [0.        , 0.18050974, 0.27147245, 0.33286789, 0.36855979,
        0.39907918, 0.43410957, 0.45591919, 0.47919596, 0.4914568 ,
        0.50693039],
       [0.        , 0.18728642, 0.27056807, 0.33451912, 0.38059231,
        0.41981962, 0.45167019, 0.48205173, 0.51294312, 0.53833324,
        0.57111647]]),
        np.array([[0.        , 0.04970474, 0.0869798 , 0.11903654, 0.15534592,
        0.17727829, 0.20015053, 0.20688221, 0.23866455, 0.25966229,
        0.27401187],
       [0.        , 0.05254539, 0.09061468, 0.13912354, 0.17743409,
        0.20359755, 0.24222139, 0.26485779, 0.29085169, 0.31732387,
        0.33524825],
       [0.        , 0.03628528, 0.06231944, 0.08789135, 0.11189568,
        0.13338403, 0.15071193, 0.17301166, 0.17995922, 0.20320912,
        0.21754693],
       [0.        , 0.03919001, 0.06256591, 0.08276759, 0.10165094,
        0.11452073, 0.13107034, 0.14103089, 0.14916935, 0.16077829,
        0.1685273 ],
       [0.        , 0.14803767, 0.2062031 , 0.23869462, 0.26899618,
        0.29475712, 0.3106018 , 0.33655331, 0.35287762, 0.37863731,
        0.39232126],
       [0.        , 0.10528444, 0.15721056, 0.19298051, 0.21850618,
        0.23987739, 0.25577525, 0.27292053, 0.29102734, 0.30250714,
        0.31543337],
       [0.        , 0.11906548, 0.16758757, 0.20492093, 0.23026454,
        0.25296962, 0.27082315, 0.28726328, 0.29818494, 0.32045305,
        0.32732069],
       [0.        , 0.10030255, 0.15341737, 0.20138957, 0.23870613,
        0.27370047, 0.29852733, 0.32546612, 0.34780858, 0.37515709,
        0.39102642]]),
        np.array([[0.        , 0.06366433, 0.11206631, 0.14632504, 0.17484113,
        0.20502613, 0.21954086, 0.24744285, 0.26779082, 0.28969772,
        0.30309555],
       [0.        , 0.04786356, 0.0903533 , 0.13942871, 0.16877514,
        0.1964454 , 0.23222649, 0.25908046, 0.2873684 , 0.31231206,
        0.33513576],
       [0.        , 0.0314505 , 0.06373121, 0.09232496, 0.11452551,
        0.14197761, 0.15949579, 0.1831393 , 0.20023212, 0.21817254,
        0.24418183],
       [0.        , 0.02603139, 0.05113609, 0.07233182, 0.09061167,
        0.11202977, 0.12313762, 0.13352904, 0.15400248, 0.15515025,
        0.16595102],
       [0.        , 0.20953398, 0.26870118, 0.29826069, 0.3306852 ,
        0.35712885, 0.37724545, 0.402646  , 0.41550509, 0.43506018,
        0.45001129],
       [0.        , 0.10431084, 0.16015775, 0.19752472, 0.23512803,
        0.25719642, 0.27606703, 0.29817176, 0.3136553 , 0.33135927,
        0.34485313],
       [0.        , 0.16923478, 0.22146758, 0.249     , 0.27420567,
        0.29425078, 0.30975559, 0.32550656, 0.33499784, 0.3478628 ,
        0.36070055],
       [0.        , 0.12688413, 0.20915701, 0.25798898, 0.29847985,
        0.33094302, 0.35741263, 0.38219601, 0.41049851, 0.43263791,
        0.45549266]]),
        np.array([[0.        , 0.03969412, 0.07850853, 0.11734785, 0.14257227,
        0.16499996, 0.20206761, 0.21423998, 0.23886763, 0.25650193,
        0.27674999],
       [0.        , 0.02060213, 0.05570588, 0.07962893, 0.11890793,
        0.15961941, 0.18580423, 0.21334974, 0.23916856, 0.26794735,
        0.28902973],
       [0.        , 0.01777297, 0.04197021, 0.06990373, 0.08761999,
        0.10914268, 0.13090144, 0.14592044, 0.16841107, 0.19327792,
        0.20340821],
       [0.        , 0.01982511, 0.03253966, 0.05443813, 0.06551632,
        0.08189303, 0.09721449, 0.10348077, 0.11643701, 0.12630137,
        0.13665804],
       [0.        , 0.17147022, 0.21459096, 0.23710744, 0.26236403,
        0.28347436, 0.30195278, 0.31782746, 0.33966071, 0.35329636,
        0.37381231],
       [0.        , 0.04386036, 0.09173428, 0.1238076 , 0.15566483,
        0.17560848, 0.19706329, 0.21064607, 0.22823917, 0.24689332,
        0.25502545],
       [0.        , 0.07661683, 0.1200709 , 0.15333289, 0.18388867,
        0.21166108, 0.23389878, 0.25589256, 0.26766092, 0.28611029,
        0.30536344],
       [0.        , 0.10066426, 0.16999533, 0.21342289, 0.25673386,
        0.2836308 , 0.3155915 , 0.34078354, 0.36179174, 0.3878621 ,
        0.40859982]]),
        np.array([[0.        , 0.06234491, 0.11449632, 0.146277  , 0.17829317,
        0.20469275, 0.2325462 , 0.25823104, 0.28234758, 0.29999243,
        0.31710299],
       [0.        , 0.05751459, 0.1122032 , 0.15795412, 0.19721723,
        0.22833867, 0.27057609, 0.29044238, 0.32456734, 0.35715248,
        0.37464227],
       [0.        , 0.03206962, 0.06008573, 0.09304891, 0.11759267,
        0.14233848, 0.16324586, 0.18113216, 0.20390412, 0.22539795,
        0.24062229],
       [0.        , 0.02386904, 0.05094115, 0.07225236, 0.08933108,
        0.10419281, 0.11645811, 0.13068253, 0.14863483, 0.15394327,
        0.17221126],
       [0.        , 0.21216329, 0.26615161, 0.29978541, 0.31906808,
        0.34339112, 0.36239126, 0.37545996, 0.39429183, 0.40026189,
        0.42056349],
       [0.        , 0.08965986, 0.14832649, 0.1812247 , 0.21214827,
        0.23371397, 0.25400709, 0.27379873, 0.29288663, 0.30336285,
        0.31976169],
       [0.        , 0.18246467, 0.25711481, 0.30131489, 0.33944508,
        0.36674534, 0.38609295, 0.40656836, 0.42475179, 0.44532821,
        0.46201847],
       [0.        , 0.09748085, 0.16393618, 0.21559485, 0.25267408,
        0.28590924, 0.31653579, 0.34203142, 0.37085739, 0.38990643,
        0.41423241]]),
        np.array([[0.        , 0.03722358, 0.08016157, 0.10669441, 0.12329107,
        0.14820211, 0.1701387 , 0.18848844, 0.20514525, 0.2233011 ,
        0.23873469],
       [0.        , 0.0324516 , 0.06809985, 0.10084225, 0.12782489,
        0.15036666, 0.18351914, 0.21333176, 0.23700245, 0.248856  ,
        0.28243943],
       [0.        , 0.02897078, 0.05043348, 0.07280051, 0.0938498 ,
        0.11291669, 0.13508687, 0.14560515, 0.16375334, 0.17764456,
        0.18723141],
       [0.        , 0.03155854, 0.0555724 , 0.07443679, 0.09520769,
        0.10885769, 0.12169741, 0.12722876, 0.13489391, 0.14705059,
        0.16073618],
       [0.        , 0.12230639, 0.16998785, 0.20493938, 0.23541831,
        0.25914561, 0.27828442, 0.30240094, 0.31544567, 0.3351258 ,
        0.35742368],
       [0.        , 0.09720627, 0.14593697, 0.18498678, 0.20415049,
        0.23164156, 0.25076975, 0.27066268, 0.28080638, 0.29473547,
        0.30690972],
       [0.        , 0.10901806, 0.15642511, 0.1834394 , 0.20728674,
        0.23008507, 0.24651168, 0.26374135, 0.27391455, 0.28609115,
        0.30096204],
       [0.        , 0.04078417, 0.08972171, 0.12595336, 0.16229071,
        0.19577688, 0.22455539, 0.2501407 , 0.27334951, 0.29736027,
        0.32611513]]),
        np.array([[0.        , 0.04081489, 0.07576069, 0.10459723, 0.1308139 ,
        0.15074586, 0.16956213, 0.19256262, 0.20979363, 0.22444586,
        0.24635572],
       [0.        , 0.0396361 , 0.07199266, 0.11168131, 0.14279642,
        0.17591949, 0.20200489, 0.22948375, 0.25600569, 0.27299716,
        0.30167613],
       [0.        , 0.02125798, 0.04320688, 0.06936076, 0.08945312,
        0.09982913, 0.12111811, 0.13416998, 0.1514787 , 0.16807497,
        0.18677758],
       [0.        , 0.0255833 , 0.04872772, 0.07166794, 0.08287347,
        0.09552021, 0.10776251, 0.12107417, 0.12786814, 0.14058234,
        0.14562226],
       [0.        , 0.13074301, 0.17597215, 0.21176078, 0.2398712 ,
        0.26603139, 0.28591951, 0.3119685 , 0.32899453, 0.34817585,
        0.36143559],
       [0.        , 0.07636434, 0.11285675, 0.14434326, 0.17101376,
        0.18893135, 0.20558204, 0.22561371, 0.23753547, 0.25821193,
        0.26688866],
       [0.        , 0.08855997, 0.1243125 , 0.15396053, 0.17717728,
        0.2018975 , 0.21749864, 0.23338086, 0.25306125, 0.27138678,
        0.27586686],
       [0.        , 0.07360704, 0.13623557, 0.18876764, 0.22123662,
        0.25377408, 0.28230132, 0.29866586, 0.32721525, 0.34958752,
        0.37404261]]),
        np.array([[0.        , 0.06642698, 0.1126966 , 0.15189995, 0.18188019,
        0.21203466, 0.23761848, 0.25672063, 0.27918256, 0.29515504,
        0.31114448],
       [0.        , 0.06301148, 0.10954638, 0.16273775, 0.19582382,
        0.22960034, 0.25977934, 0.28648943, 0.31384086, 0.34309337,
        0.36308334],
       [0.        , 0.03559442, 0.06965872, 0.09877728, 0.12561333,
        0.14635692, 0.17481931, 0.18961401, 0.21067214, 0.23117544,
        0.24754315],
       [0.        , 0.02895487, 0.05318866, 0.07921169, 0.09539065,
        0.11012728, 0.12357473, 0.13549778, 0.14768417, 0.16403758,
        0.17113878],
       [0.        , 0.20895577, 0.26179171, 0.29895428, 0.33325138,
        0.34629755, 0.36701564, 0.38383066, 0.41157286, 0.41913862,
        0.4372556 ],
       [0.        , 0.10746963, 0.16963434, 0.21054548, 0.24163762,
        0.2643124 , 0.28531539, 0.30033478, 0.32029992, 0.33894431,
        0.35136188],
       [0.        , 0.18790219, 0.25973382, 0.30817588, 0.33697458,
        0.36686214, 0.38801777, 0.41178419, 0.42518839, 0.44299389,
        0.45412617],
       [0.        , 0.08751997, 0.14858344, 0.19683696, 0.23883232,
        0.27307756, 0.29903127, 0.33181244, 0.36338814, 0.38202137,
        0.41104407]]),
        np.array([[0.        , 0.03593707, 0.06784239, 0.10877654, 0.13733426,
        0.15264816, 0.18556202, 0.20704747, 0.22541221, 0.24712992,
        0.26274818],
       [0.        , 0.04396279, 0.0885857 , 0.12504217, 0.15698466,
        0.18784943, 0.21860371, 0.2512041 , 0.27445771, 0.30345694,
        0.3219348 ],
       [0.        , 0.02655044, 0.05445105, 0.07715263, 0.10236562,
        0.11928474, 0.13573776, 0.15164273, 0.17740992, 0.19577253,
        0.21049252],
       [0.        , 0.02402553, 0.04882337, 0.06577226, 0.08569004,
        0.10013   , 0.11010692, 0.12403918, 0.13537077, 0.14188628,
        0.14991055],
       [0.        , 0.11517067, 0.17864818, 0.2172413 , 0.24839795,
        0.27822568, 0.30375331, 0.32545305, 0.34256423, 0.36780136,
        0.38010519],
       [0.        , 0.087405  , 0.13797413, 0.17645281, 0.20134546,
        0.22172647, 0.24062281, 0.26242454, 0.27552671, 0.29145343,
        0.30197073],
       [0.        , 0.09874134, 0.14693406, 0.18251505, 0.21063734,
        0.23376327, 0.24780134, 0.27242502, 0.28247593, 0.29713768,
        0.30894214],
       [0.        , 0.05289241, 0.10426139, 0.14653627, 0.18209661,
        0.21584123, 0.24857923, 0.26873011, 0.30118947, 0.3274846 ,
        0.34730771]]),
        np.array([[0.        , 0.06278092, 0.11502388, 0.15198392, 0.18504768,
        0.21297304, 0.23598708, 0.26236858, 0.28371897, 0.3052451 ,
        0.31087664],
       [0.        , 0.05504284, 0.11435937, 0.1534419 , 0.20312679,
        0.23787232, 0.27716745, 0.30216277, 0.33699471, 0.36137493,
        0.38255609],
       [0.        , 0.02824993, 0.06594923, 0.08971839, 0.11630505,
        0.13758822, 0.16367489, 0.18189233, 0.20272819, 0.22310211,
        0.24385654],
       [0.        , 0.03128787, 0.05686137, 0.08253716, 0.09719751,
        0.10988437, 0.1279442 , 0.13965978, 0.1486547 , 0.1662987 ,
        0.17453314],
       [0.        , 0.20656727, 0.26162965, 0.30277574, 0.33357291,
        0.36209645, 0.3786793 , 0.40211512, 0.42109132, 0.43991876,
        0.46294366],
       [0.        , 0.11139267, 0.16750946, 0.2144694 , 0.24168504,
        0.27051676, 0.28415466, 0.3073004 , 0.32477562, 0.33799778,
        0.34795562],
       [0.        , 0.15940642, 0.21659247, 0.25053834, 0.28651985,
        0.30641282, 0.32976667, 0.34864769, 0.36378047, 0.37463433,
        0.38595812],
       [0.        , 0.13366788, 0.22086329, 0.27558284, 0.32130776,
        0.36222546, 0.38942154, 0.42032159, 0.45172189, 0.46872536,
        0.49551464]]),
        np.array([[0.        , 0.02924261, 0.05504948, 0.07613129, 0.09856633,
        0.1262956 , 0.13765071, 0.15764234, 0.17305251, 0.19640761,
        0.21367353],
       [0.        , 0.03760503, 0.07536024, 0.1135612 , 0.15238676,
        0.18642613, 0.2187836 , 0.2560403 , 0.28920734, 0.30660623,
        0.34731379],
       [0.        , 0.01675491, 0.04055408, 0.05367299, 0.07391602,
        0.09375279, 0.10896844, 0.12514837, 0.13467749, 0.15323417,
        0.16771466],
       [0.        , 0.0224432 , 0.04011974, 0.05014071, 0.06061145,
        0.07777322, 0.08829194, 0.09534601, 0.11216376, 0.10878037,
        0.11993066],
       [0.        , 0.11655578, 0.16602204, 0.20605964, 0.22921952,
        0.251824  , 0.2770459 , 0.28749715, 0.3067288 , 0.32005917,
        0.34355373],
       [0.        , 0.08674282, 0.1422218 , 0.17417672, 0.19748817,
        0.21839482, 0.2327423 , 0.25713377, 0.26645243, 0.28391436,
        0.29793536],
       [0.        , 0.09653594, 0.1559815 , 0.20084107, 0.23159873,
        0.25779818, 0.28131297, 0.29928475, 0.31879532, 0.33090285,
        0.34763147],
       [0.        , 0.04749366, 0.09394181, 0.13968869, 0.18092432,
        0.21097466, 0.23884821, 0.27155351, 0.29243863, 0.32318136,
        0.34651883]])]
    '''
    # Get average utility matrix
    avgUtilMat = np.average(np.array(utilMatList), axis=0)
    '''23-MAR
    avgUtilMat = np.array([[0.        , 0.03926838, 0.07791196, 0.10839613, 0.13673667,
        0.16127957, 0.18485203, 0.20581453, 0.22645536, 0.24506592,
        0.26323204],
       [0.        , 0.04156318, 0.08527675, 0.12581979, 0.16246621,
        0.19442731, 0.2267136 , 0.25486866, 0.28068494, 0.30563487,
        0.32856128],
       [0.        , 0.02235837, 0.04621302, 0.06933629, 0.09090974,
        0.110414  , 0.13059489, 0.14698022, 0.16571372, 0.18305777,
        0.19843822],
       [0.        , 0.02245801, 0.04173541, 0.05938949, 0.07395568,
        0.08825078, 0.09781077, 0.10883846, 0.11968096, 0.12829845,
        0.13670912],
       [0.        , 0.1505197 , 0.20177135, 0.23532543, 0.26439749,
        0.28879939, 0.30914706, 0.32921345, 0.34792501, 0.36456342,
        0.38247052],
       [0.        , 0.08800784, 0.13815685, 0.17416162, 0.20047618,
        0.22303752, 0.24148283, 0.2602546 , 0.27537216, 0.29109162,
        0.30380615],
       [0.        , 0.11915504, 0.1742445 , 0.21265924, 0.24227535,
        0.26553351, 0.28711584, 0.30576978, 0.32048025, 0.3356621 ,
        0.34886201],
       [0.        , 0.09462682, 0.16050694, 0.20691627, 0.24401184,
        0.27621648, 0.30429699, 0.33007624, 0.35677066, 0.37857534,
        0.40152495]])
    '''
    # Find allocation for sample budget
    allocArr = forwardAllocateWithBudget(avgUtilMat, int(sampBudget / testInt))
    designArr = allocArr / np.sum(allocArr, axis=0)
    # Get utility for this allocation at the sample budget
    utilDict.update({'method': 'weightsNodeDraw4linear'})
    compUtilList, unifUtilList, origUtilList = [], [], []
    numReps = 20
    for rep in range(numReps):
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New Bayes draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        print('Generating loss matrix...')
        tempLossMat = lf.lossMatSetBayesDraws(dictTemp['postSamples'], lossDict.copy(), lossDict['bayesDraws'])
        tempLossDict = lossDict.copy()
        tempLossDict.update({'lossMat': tempLossMat})
        newBayesDraws, newLossMat = lf.add_cand_neighbors(tempLossDict.copy(), CSdict3['postSamples'],
                                                          dictTemp['postSamples'])
        tempLossDict.update({'bayesDraws': newBayesDraws, 'lossMat': newLossMat})
        baseLoss = (np.sum(newLossMat, axis=1) / newLossMat.shape[1]).min()
        # Get a new set of data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        currCompUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[designArr],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Heuristic utility: ' + str(currCompUtil))
        compUtilList.append(currCompUtil)
        '''23-MAR
        compUtilList = [1.192012268324163, 1.125911805030805, 1.2129202405626964, 1.1386291770898724, 1.1738049489343725]
        '''
        # Find the equivalent uniform allocation
        currUnifUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[unifDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Uniform utility: ' + str(currUnifUtil))
        unifUtilList.append([currUnifUtil])
        unifAdd, contUnif, unifCount = 0, False, 0
        if currUnifUtil < currCompUtil:
            contUnif = True
        while contUnif:
            unifAdd += testInt
            print('Adding ' + str(unifAdd) + ' for uniform')
            currUnifUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[unifDes], numtests=sampBudget + unifAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currUnifUtil))
            unifUtilList[rep].append(currUnifUtil)
            if currUnifUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if unifCount < 3:
                    unifCount += 1
                else:
                    contUnif = False
        '''23-MAR
        unifUtilList = [[1.1959196818826197], [1.1367206366985867], [1.1919032360887725, 1.2510841164942947, 1.2974759051561984, 1.3415844322150319, 1.3938819288770121], [1.1464615472380366], [1.1340475898897822, 1.1891464191210375, 1.2500846264240906, 1.2754686176040666, 1.3241055509383495]]
        [[1.190959386429343, 1.2469465707181469, 1.285159936502886, 1.331782374240376, 1.3802551589089536, 1.4157407322443305, 1.482174008656309], [1.1875281334802974, 1.2340941923095432, 1.2654269836970782, 1.3132558256143065, 1.3645677303015145, 1.40657741862091, 1.435601801985337, 1.4831289250481179], [1.2272088827237102, 1.263797386842322, 1.3184502509149887, 1.3687358603570017, 1.4017726126838062, 1.4448630277903902, 1.4849932205068272], [1.2223955283295274, 1.2446654751808746, 1.2915819001650632, 1.3528435327661237, 1.3747627663459943, 1.418115883886661, 1.4643037078488375], [1.1660417687278155, 1.2293618385680496, 1.3052643896533338, 1.3199829510431336, 1.3679393483992142, 1.4124382110080562, 1.4502061696674993], [1.1972612140890062, 1.251168032349642, 1.2900477965232886, 1.3389693146884953, 1.3697889051547092, 1.41508654314198, 1.4382393426394353], [1.0916848599506288, 1.1473131260702836, 1.1943377963229747, 1.2524851261761993, 1.2773673362449314, 1.3333886133919401, 1.3852597939558633], [1.1236203161207818, 1.1684455430235525, 1.2093304594085676, 1.26250398050571, 1.3160420011142975, 1.3565749328419994, 1.3870064673318225], [1.1306914312878602, 1.182146339264694, 1.2275759310860366, 1.2773198535975494, 1.3232770769909363, 1.376079160891483, 1.4312055270660746], [1.146897896376374, 1.1804977270309767, 1.2447588202106337, 1.3057163975671333, 1.338566856498792, 1.3708651400764706, 1.431343188791116], [1.0541376760208232, 1.092945019116538, 1.1558346900241236, 1.1876938470472815, 1.2291099867802995, 1.2834333055322276, 1.3296943218747375], [1.1313377659961485, 1.1838318670531205, 1.2355405448639436, 1.2612856202622051, 1.315697175463984, 1.352096468373385, 1.3928361904031341], [1.1185626141373781, 1.1771003140468377, 1.2216185188920434, 1.268696398985396, 1.3226661753857352, 1.3591478786139106, 1.4072174039095762], [1.441953375925653, 1.4692823207297314, 1.5322280846274925, 1.5743668797035175, 1.613210365470609, 1.643503373063126, 1.705689200951177], [1.2358252976656559, 1.2838997664385738, 1.3375513543079114, 1.3777928112408304, 1.418347863129167, 1.4580596553320806, 1.5027891761398102], [1.1402670733848899, 1.197751502896546, 1.251057046951555, 1.2737067645973807, 1.3330099688384651, 1.3581843375502283, 1.4180426232047996], [1.0891977531830896, 1.1305008979062654, 1.207229789239622, 1.2465664572505366, 1.2747449741621848, 1.3258503037466323, 1.3800551456213928], [1.4694037320896198, 1.5104555072778192, 1.5787110825588861, 1.626269076624891, 1.670872880687548, 1.7137246705947442, 1.765820875763647], [1.088760804065406, 1.1178439989316358, 1.1819814796988606, 1.2150653310761124, 1.2624952618673149, 1.3097882984435332, 1.3305546042659406, 1.388785229040681], [1.1112150823881883, 1.1748590895439417, 1.2268123292716702, 1.2734940194895614, 1.332433661650673, 1.358679776165336, 1.3940957094778827, 1.4441975594720082]]
        '''
        # Find the equivalent rudimentary allocation
        currOrigUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[origDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Rudimentary utility: ' + str(currOrigUtil))
        origUtilList.append([currOrigUtil])
        origAdd, contOrig, origCount = 0, False, 0
        if currOrigUtil < currCompUtil:
            contOrig = True
        while contOrig:
            origAdd += testInt * 3
            print('Adding ' + str(origAdd) + ' for rudimentary')
            currOrigUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[origDes], numtests=sampBudget + origAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currOrigUtil))
            origUtilList[rep].append(currOrigUtil)
            if currOrigUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if origCount < 3:
                    origCount += 1
                else:
                    contOrig = False
        '''23-MAR
        origUtilList = [[0.4961351874449971, 0.5812406602661708, 0.6639967576389605, 0.7339357802965134, 0.8149768477982109, 0.9134247701935951, 0.9850089802770317, 1.0793159825706065, 1.1395685364333765, 1.2252455013043813, 1.295641076565344, 1.3524004984434868, 1.4668203276199536], [0.4337137700467535, 0.516265520037666, 0.6113375974329172, 0.678551930429526, 0.7466892464655315, 0.8711306854147312, 0.9559825668033106, 1.0234369618868513, 1.0890500487885735, 1.1761309556805006, 1.2234069730414268, 1.3168829036105842, 1.3927573878140258], [0.5126344465573505, 0.5762948308309914, 0.6721674003976492, 0.7477855458238434, 0.8265778760015365, 0.9198946821597733, 0.9789748333088726, 1.091264976883644, 1.1350601321469984, 1.2046955600181093, 1.2842357880855753, 1.3643380526108224, 1.4223389118655563, 1.5109337905178188], [0.4661091402776618, 0.5502888688934116, 0.6176231008974069, 0.7154924819824604, 0.8047512405046886, 0.8916791819290131, 0.9599285050026793, 1.0497582828116565, 1.1312672070134218, 1.1858683468739653, 1.2700224668659956, 1.3251617888629825, 1.407352941097189], [0.46058503715415, 0.5364183018384869, 0.6210138324996164, 0.6905269549309025, 0.7891408039760508, 0.859953673289577, 0.9577629572196598, 1.0559000172854045, 1.0813804949595136, 1.1763147470403439, 1.2579021486914321, 1.3228981049244144, 1.393222503488813]]
        [[0.47820161442619824, 0.5658952041151926, 0.6289187797363756, 0.7168303683334303, 0.8089530978739585, 0.874453885194856, 0.9372741239665707, 1.0431355928678405, 1.1048286564841483, 1.1467004705070964, 1.2558509323621623, 1.3091510916476858, 1.3940368780641732, 1.4673875062018729, 1.510904496433728, 1.5961612336437279], [0.5149325096268234, 0.5840081076704475, 0.6618070875104998, 0.7459004409309795, 0.8472967453504161, 0.90618602586588, 0.9786152920014883, 1.0927264894691717, 1.1440702796761135, 1.197311250865039, 1.2937894605489015, 1.3730946488212652, 1.4377248348457847, 1.494936080155382, 1.5731906415469545], [0.4748405359049297, 0.5525760370025736, 0.6139866453551823, 0.7169661596058292, 0.8008012189289797, 0.889320148721767, 0.9665832747592811, 1.0771255750711761, 1.1039954955040407, 1.1937410551766154, 1.2639479138224008, 1.3562303378154779, 1.4063153434919666, 1.4749594251185112, 1.5392792166940916], [0.500654148639982, 0.5855054935236295, 0.6657984255271172, 0.7349813480735765, 0.8311973379495319, 0.9028204431564228, 1.0087050004094058, 1.0874554361996962, 1.1481468490153395, 1.2226573024285052, 1.3142787711107031, 1.375802678775126, 1.4360959160255522, 1.5079456955281678, 1.5675886951266111], [0.47472942607176893, 0.548260114753722, 0.6369532670687734, 0.7177259230061477, 0.8011122176096235, 0.8942294944399469, 0.9755814543842138, 1.0897381703654663, 1.1391036996534627, 1.1938394468783144, 1.277215621426386, 1.3374646381488526, 1.451095620574118, 1.5043058763313102, 1.5314563527856926], [0.5055606606333187, 0.5833743910117617, 0.669707423291805, 0.7491682641041946, 0.8239238646630662, 0.8947868276452082, 0.968776405728871, 1.0907587168800914, 1.150621553498655, 1.2162703722038488, 1.3024141181187252, 1.3605309245938235, 1.441673645302659, 1.5293234801694053, 1.5410575756385292], [0.4230331198708477, 0.4910120085379983, 0.5917911269657052, 0.6683170226069226, 0.7616823702866045, 0.8299659916875424, 0.9308308365657485, 0.9990065163199899, 1.0728109808743822, 1.1514862173609552, 1.2088983021858963, 1.2781942456562767, 1.3675512140737673, 1.4217840731315103, 1.4899685695903848], [0.4290002871021388, 0.5082277912761208, 0.5844782275433058, 0.6649891221375173, 0.7448824643767384, 0.8227634722953523, 0.9056895758250372, 1.0217439711940681, 1.056288937498989, 1.1607487523317532, 1.2014512165727007, 1.2952181688952464, 1.3583999107743052, 1.4486011517097745, 1.4848852118029585], [0.4690963754961093, 0.5346593871301746, 0.6171772989146165, 0.7009444607159674, 0.7784746757189325, 0.8569712760693218, 0.9507507322845692, 1.0394532685757483, 1.09411594354808, 1.1809821343395326, 1.2434991975631107, 1.3001410492956023, 1.3906086764471608, 1.4686561796612168, 1.517300767232781], [0.4552667610600736, 0.5279235606274648, 0.6232789794343203, 0.6968693254045792, 0.7842409373400319, 0.864823373491137, 0.9610745207229758, 1.0593884278531096, 1.1226393640639456, 1.2011730859075236, 1.269940721919876, 1.337893380665677, 1.4158268624639385, 1.474837352650141, 1.557370021981491], [0.3187610780901702, 0.40655693595556386, 0.4858894694956861, 0.5717733142843402, 0.6385897454182139, 0.7531713648069926, 0.8248794729303004, 0.924687458726904, 0.9909657273774339, 1.0855861752845088, 1.1598007416840224, 1.1985618150969364, 1.3196068695154421, 1.3689710237775508, 1.4452450209295158], [0.4005847961281157, 0.48827598657901916, 0.5644861178533027, 0.6507066263688115, 0.7370284858856548, 0.812573449869709, 0.9090362221493153, 1.0091525372695829, 1.064926357167634, 1.1222875800464083, 1.1987703374629266, 1.2854197298648176, 1.3524709424400152, 1.4139148755046733, 1.4917339606524171], [0.44166502758558934, 0.5122903851833422, 0.5768081449484113, 0.6656000483587938, 0.7485977723145636, 0.8180709907701846, 0.8933253329883124, 1.018547302354058, 1.0689324366564277, 1.1438244656191818, 1.2161375835475292, 1.2792083761784436, 1.3429663766109945, 1.4171957297038151, 1.4834367735498768], [0.6048378553928613, 0.6993221367113094, 0.8056238476381736, 0.8968089326246389, 0.9756134769819571, 1.0495563321263486, 1.176395389111387, 1.264751667620125, 1.3305495521138466, 1.4149659420390508, 1.4973651862161752, 1.5747057664586244, 1.6467876720661847, 1.7076445960920204, 1.7841416236035261], [0.5254785062863121, 0.5953115542014276, 0.6864290117527467, 0.765152099226813, 0.8330166031144088, 0.9121142507153834, 0.9966576255808626, 1.125649756401697, 1.1373380767502352, 1.224959586430546, 1.3097247569968387, 1.3609976176704484, 1.45876431258396, 1.5380083463274143, 1.5820652954425425], [0.4574051900700944, 0.5420721808604041, 0.6157900263419807, 0.6840724569562644, 0.7780526258866436, 0.8491984685512879, 0.9237884708774322, 1.0432662317533365, 1.1069313796853484, 1.1549811813590907, 1.2275442615267504, 1.270517366036815, 1.3750066273034642, 1.434490800272684, 1.4851302590775362], [0.3948326789480445, 0.4841921808546408, 0.5593459498188436, 0.6348115838541979, 0.7031870486182994, 0.8069868776404094, 0.8862964822001613, 0.9760307889515367, 1.0509026489031061, 1.1130884754709323, 1.1853892966082906, 1.2670745753874364, 1.337315158131223, 1.4220300021836039, 1.4939568097369387], [0.6559368143848667, 0.7518232286925981, 0.8551487398923694, 0.9361281127279959, 1.04724443269654, 1.1362719944957815, 1.2059714565634785, 1.3301358025143806, 1.3901958299351511, 1.4724348211273792, 1.5450218466967671, 1.6288561089047828, 1.7070432319129134, 1.7642271222073411, 1.867522400016358], [0.40541842485603663, 0.47255647015378877, 0.5446389258292976, 0.6466245073076902, 0.727795072791249, 0.7962206205389251, 0.8883851915357575, 0.9888420510074587, 1.0518892444682129, 1.1100282585231929, 1.1948029405433314, 1.2701219966983857, 1.3565937538744617, 1.4308160991303662, 1.4924905269270905], [0.4457715481007538, 0.5258298842921505, 0.6072023471562824, 0.6746103541928141, 0.7502895342163489, 0.8461242571386558, 0.9182991226350854, 1.0214188270276452, 1.0829157137735894, 1.158550835506304, 1.2441341284565723, 1.3262733104333373, 1.3825195022695853, 1.4255217187860305, 1.5376338502480156]]
        '''
    compAvg = np.average(compUtilList)
    # Locate closest sample point for uniform and rudimentary to compAvg
    minListLen = np.min([len(i) for i in unifUtilList])
    unifUtilArr = np.array([i[:minListLen] for i in unifUtilList])
    unifAvgArr = np.average(unifUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(unifAvgArr.tolist()) if val > compAvg)
    unifSampSaved = round((compAvg - unifAvgArr[kInd - 1]) / (unifAvgArr[kInd] - unifAvgArr[kInd - 1]) * testInt) + (
            kInd - 1) * testInt
    print('Saved vs uniform: ' + str(unifSampSaved))
    '''23-MAR: 26 saved'''
    # Rudimentary
    minListLen = np.min([len(i) for i in origUtilList])
    origUtilArr = np.array([i[:minListLen] for i in origUtilList])
    origAvgArr = np.average(origUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(origAvgArr.tolist()) if val > compAvg)
    origSampSaved = round(
        (compAvg - origAvgArr[kInd - 1]) / (origAvgArr[kInd] - origAvgArr[kInd - 1]) * testInt * 3) + (
                            kInd - 1) * testInt * 3
    print('Saved vs rudimentary: ' + str(origSampSaved))
    '''23-MAR: 315 saved'''

    ##############################################
    ##############################################
    # Now use a different sampling budget
    ##############################################
    ##############################################
    sampBudget = 90
    unifDes = np.zeros(numTN) + 1 / numTN
    origDes = np.sum(rd3_N, axis=1) / np.sum(rd3_N)

    # Use different loss parameters
    ###########
    # todo: checkSlope = 0.3
    ###########
    paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.3,
                                                  marketvec=np.ones(numTN + numSN), candneighnum=1000)

    numtargetdraws, numDataDraws = 5100, 5000

    # Find heuristic allocation first
    utilDict = {'method': 'weightsNodeDraw3linear'}

    numReps = 5
    utilMatList = []
    # set testMax to highest expected allocation for any one node
    testMax, testInt = 100, 10
    for rep in range(numReps):
        # Withdraw a subset of MCMC prior draws
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New loss draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        # Get new data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        # Get marginal utilities at each test node
        currMargUtilMat = GetMargUtilAtNodes(dictTemp.copy(), testMax, testInt, lossDict.copy(), utilDict.copy(),
                                             masterDraws=CSdict3['postSamples'], printUpdate=True)
        print(repr(currMargUtilMat))
        utilMatList.append(currMargUtilMat)
    '''21-MAR run
    utilMatList =  [np.array([[0.        , 0.02500037, 0.05453001, 0.08209539, 0.10406642,
        0.12635247, 0.14372454, 0.16378819, 0.18127501, 0.19769524,
        0.21806803],
       [0.        , 0.06780398, 0.12588295, 0.17474233, 0.2130504 ,
        0.25103884, 0.27618104, 0.30498385, 0.32892034, 0.35248336,
        0.3769582 ],
       [0.        , 0.0231166 , 0.04776599, 0.0695312 , 0.08821245,
        0.10760009, 0.12111096, 0.14170292, 0.15748052, 0.17533819,
        0.18795814],
       [0.        , 0.01274688, 0.02207516, 0.03094908, 0.03848035,
        0.04623635, 0.05366907, 0.06208322, 0.06854747, 0.07078396,
        0.07963057],
       [0.        , 0.12318974, 0.17220442, 0.20022558, 0.22574309,
        0.24298633, 0.26482079, 0.27670645, 0.29379246, 0.30827047,
        0.32425711],
       [0.        , 0.06763408, 0.11554658, 0.14757023, 0.16712204,
        0.18568884, 0.20369205, 0.21839598, 0.23067107, 0.24024285,
        0.25794557],
       [0.        , 0.09427582, 0.15597956, 0.19570334, 0.22112415,
        0.24196177, 0.26310672, 0.28087477, 0.29439486, 0.30353926,
        0.3170758 ],
       [0.        , 0.08791447, 0.14834522, 0.18427286, 0.22501034,
        0.25045911, 0.2781266 , 0.30292225, 0.33079005, 0.34387117,
        0.36923574]]),
    np.array([[0.        , 0.04263898, 0.0805571 , 0.11234862, 0.1403126 ,
        0.16875213, 0.18675966, 0.20837249, 0.22678839, 0.250339  ,
        0.26368037],
       [0.        , 0.10668159, 0.17626777, 0.23209587, 0.27828299,
        0.31760766, 0.34928672, 0.3738254 , 0.40164507, 0.42471862,
        0.4535201 ],
       [0.        , 0.03326988, 0.06463894, 0.09013261, 0.11849815,
        0.13636318, 0.15945141, 0.18186024, 0.197517  , 0.22064851,
        0.23738432],
       [0.        , 0.02260935, 0.04522903, 0.05965912, 0.070499  ,
        0.08017769, 0.09357638, 0.1044922 , 0.1115661 , 0.12266323,
        0.12643848],
       [0.        , 0.11983738, 0.17577183, 0.21207856, 0.24248467,
        0.26709557, 0.28931028, 0.31106719, 0.32662903, 0.34681699,
        0.36575629],
       [0.        , 0.16976545, 0.22001581, 0.25354531, 0.27824367,
        0.29792475, 0.30811732, 0.32223136, 0.33516808, 0.3471305 ,
        0.35538685],
       [0.        , 0.14437884, 0.20555892, 0.24294385, 0.27550474,
        0.30383918, 0.32267132, 0.34398143, 0.35795046, 0.3729873 ,
        0.39160028],
       [0.        , 0.0947162 , 0.15201215, 0.20010508, 0.23995104,
        0.26819929, 0.30217706, 0.3262525 , 0.35324365, 0.3766628 ,
        0.40135085]]),
    np.array([[0.        , 0.01587516, 0.04593522, 0.07223151, 0.09514553,
        0.11917388, 0.13999842, 0.16216311, 0.1805129 , 0.20201705,
        0.2126472 ],
       [0.        , 0.07058362, 0.13200391, 0.17571572, 0.21648197,
        0.2498789 , 0.27963138, 0.30645987, 0.32903156, 0.34734348,
        0.37093671],
       [0.        , 0.01119609, 0.02935217, 0.05450204, 0.06986139,
        0.08942349, 0.10814692, 0.12211996, 0.14378273, 0.15544419,
        0.17274479],
       [0.        , 0.00779084, 0.02195131, 0.03396341, 0.04757296,
        0.05536284, 0.06179379, 0.07038192, 0.07845354, 0.08556913,
        0.08999636],
       [0.        , 0.07451111, 0.12030676, 0.16170979, 0.19042464,
        0.2129504 , 0.23801667, 0.25476103, 0.27441601, 0.29344206,
        0.3072715 ],
       [0.        , 0.11605207, 0.16142447, 0.18783149, 0.21283935,
        0.22902721, 0.24190524, 0.26048196, 0.27020456, 0.2792825 ,
        0.29289628],
       [0.        , 0.12127702, 0.17564384, 0.21138176, 0.24317561,
        0.26714787, 0.28969989, 0.3025183 , 0.31865848, 0.33093037,
        0.34516775],
       [0.        , 0.08597358, 0.14435954, 0.19249878, 0.22742487,
        0.26077066, 0.28591513, 0.31405436, 0.3391699 , 0.36238359,
        0.38004766]]),
    np.array([[0.        , 0.02987273, 0.06525859, 0.09093212, 0.12042844,
        0.14311871, 0.16312245, 0.18114201, 0.19560633, 0.21208205,
        0.23025551],
       [0.        , 0.05167086, 0.11339466, 0.16032641, 0.20332102,
        0.23676281, 0.26918115, 0.29902935, 0.32377649, 0.34639625,
        0.37086335],
       [0.        , 0.02448826, 0.05228473, 0.07657187, 0.09806873,
        0.11981395, 0.14070131, 0.15774672, 0.17648881, 0.19393561,
        0.21304374],
       [0.        , 0.01636897, 0.02975276, 0.03622167, 0.05152468,
        0.05760275, 0.06538814, 0.07191806, 0.08184431, 0.09189486,
        0.09492135],
       [0.        , 0.11032845, 0.16476965, 0.20177127, 0.23169792,
        0.25892656, 0.28164654, 0.30245426, 0.31937686, 0.33805109,
        0.35663242],
       [0.        , 0.0892706 , 0.14144644, 0.1773953 , 0.20140365,
        0.22173483, 0.23987947, 0.25359025, 0.2649582 , 0.28201554,
        0.29154265],
       [0.        , 0.09411578, 0.15417961, 0.19097058, 0.21931113,
        0.23784049, 0.25735508, 0.27423437, 0.28904966, 0.3022289 ,
        0.31878039],
       [0.        , 0.08576752, 0.14682849, 0.19408626, 0.23024442,
        0.25924388, 0.2881656 , 0.31408096, 0.33713355, 0.35968549, 0.38057977]]), 
    np.array([[0.        , 0.02500037, 0.05453001, 0.08209539, 0.10406642,
        0.12635247, 0.14372454, 0.16378819, 0.18127501, 0.19769524, 0.21806803],
       [0.        , 0.06780398, 0.12588295, 0.17474233, 0.2130504 ,
        0.25103884, 0.27618104, 0.30498385, 0.32892034, 0.35248336, 0.3769582 ],
       [0.        , 0.0231166 , 0.04776599, 0.0695312 , 0.08821245,
        0.10760009, 0.12111096, 0.14170292, 0.15748052, 0.17533819, 0.18795814],
       [0.        , 0.01274688, 0.02207516, 0.03094908, 0.03848035,
        0.04623635, 0.05366907, 0.06208322, 0.06854747, 0.07078396, 0.07963057],
       [0.        , 0.12318974, 0.17220442, 0.20022558, 0.22574309,
        0.24298633, 0.26482079, 0.27670645, 0.29379246, 0.30827047, 0.32425711],
       [0.        , 0.06763408, 0.11554658, 0.14757023, 0.16712204,
        0.18568884, 0.20369205, 0.21839598, 0.23067107, 0.24024285, 0.25794557],
       [0.        , 0.09427582, 0.15597956, 0.19570334, 0.22112415,
        0.24196177, 0.26310672, 0.28087477, 0.29439486, 0.30353926, 0.3170758 ],
       [0.        , 0.08791447, 0.14834522, 0.18427286, 0.22501034,
        0.25045911, 0.2781266 , 0.30292225, 0.33079005, 0.34387117, 0.36923574]]), 
    np.array([[0.        , 0.04263898, 0.0805571 , 0.11234862, 0.1403126 ,
        0.16875213, 0.18675966, 0.20837249, 0.22678839, 0.250339  ,
        0.26368037],
       [0.        , 0.10668159, 0.17626777, 0.23209587, 0.27828299,
        0.31760766, 0.34928672, 0.3738254 , 0.40164507, 0.42471862,
        0.4535201 ],
       [0.        , 0.03326988, 0.06463894, 0.09013261, 0.11849815,
        0.13636318, 0.15945141, 0.18186024, 0.197517  , 0.22064851,
        0.23738432],
       [0.        , 0.02260935, 0.04522903, 0.05965912, 0.070499  ,
        0.08017769, 0.09357638, 0.1044922 , 0.1115661 , 0.12266323,
        0.12643848],
       [0.        , 0.11983738, 0.17577183, 0.21207856, 0.24248467,
        0.26709557, 0.28931028, 0.31106719, 0.32662903, 0.34681699,
        0.36575629],
       [0.        , 0.16976545, 0.22001581, 0.25354531, 0.27824367,
        0.29792475, 0.30811732, 0.32223136, 0.33516808, 0.3471305 ,
        0.35538685],
       [0.        , 0.14437884, 0.20555892, 0.24294385, 0.27550474,
        0.30383918, 0.32267132, 0.34398143, 0.35795046, 0.3729873 ,
        0.39160028],
       [0.        , 0.0947162 , 0.15201215, 0.20010508, 0.23995104,
        0.26819929, 0.30217706, 0.3262525 , 0.35324365, 0.3766628 ,
        0.40135085]]), 
    np.array([[0.        , 0.01587516, 0.04593522, 0.07223151, 0.09514553,
        0.11917388, 0.13999842, 0.16216311, 0.1805129 , 0.20201705,
        0.2126472 ],
       [0.        , 0.07058362, 0.13200391, 0.17571572, 0.21648197,
        0.2498789 , 0.27963138, 0.30645987, 0.32903156, 0.34734348,
        0.37093671],
       [0.        , 0.01119609, 0.02935217, 0.05450204, 0.06986139,
        0.08942349, 0.10814692, 0.12211996, 0.14378273, 0.15544419,
        0.17274479],
       [0.        , 0.00779084, 0.02195131, 0.03396341, 0.04757296,
        0.05536284, 0.06179379, 0.07038192, 0.07845354, 0.08556913,
        0.08999636],
       [0.        , 0.07451111, 0.12030676, 0.16170979, 0.19042464,
        0.2129504 , 0.23801667, 0.25476103, 0.27441601, 0.29344206,
        0.3072715 ],
       [0.        , 0.11605207, 0.16142447, 0.18783149, 0.21283935,
        0.22902721, 0.24190524, 0.26048196, 0.27020456, 0.2792825 ,
        0.29289628],
       [0.        , 0.12127702, 0.17564384, 0.21138176, 0.24317561,
        0.26714787, 0.28969989, 0.3025183 , 0.31865848, 0.33093037,
        0.34516775],
       [0.        , 0.08597358, 0.14435954, 0.19249878, 0.22742487,
        0.26077066, 0.28591513, 0.31405436, 0.3391699 , 0.36238359,
        0.38004766]]), 
    np.array([[0.        , 0.03251971, 0.06843361, 0.09871152, 0.12714824,
        0.14767408, 0.1691276 , 0.18636503, 0.20809668, 0.22473232,
        0.23949598],
       [0.        , 0.05917184, 0.11805847, 0.16443774, 0.20516079,
        0.23630029, 0.27258007, 0.2930291 , 0.31589052, 0.34184075,
        0.36247042],
       [0.        , 0.01805197, 0.03653106, 0.05896414, 0.07931535,
        0.10320268, 0.1190687 , 0.13519066, 0.15517717, 0.17141938,
        0.18448667],
       [0.        , 0.01665995, 0.02790249, 0.04230972, 0.05156995,
        0.06198682, 0.07041216, 0.07866921, 0.08645476, 0.09203822,
        0.09839002],
       [0.        , 0.10347124, 0.14993074, 0.18895728, 0.21806734,
        0.24492117, 0.26628176, 0.28702734, 0.30282592, 0.32122612,
        0.33556131],
       [0.        , 0.11525652, 0.16614162, 0.20308849, 0.22684349,
        0.24589503, 0.26209648, 0.27615994, 0.28847559, 0.29948998,
        0.31038607],
       [0.        , 0.13738655, 0.19818119, 0.23976322, 0.2697537 ,
        0.29209653, 0.31080454, 0.32872838, 0.34258857, 0.35536578,
        0.36757619],
       [0.        , 0.07321535, 0.13626956, 0.17958915, 0.21335394,
        0.24568622, 0.2720279 , 0.29436191, 0.31913059, 0.3384415 ,
        0.35818161]])]
    '''
    # Get average utility matrix
    avgUtilMat = np.average(np.array(utilMatList), axis=0)
    '''21-MAR
    avgUtilMat = np.array([[0.        , 0.02918139, 0.06294291, 0.09126383, 0.11742025,
        0.14101425, 0.16054653, 0.18036617, 0.19845586, 0.21737313,
        0.23282942],
       [0.        , 0.07118238, 0.13312155, 0.18146362, 0.22325944,
        0.2583177 , 0.28937207, 0.31546551, 0.3398528 , 0.36255649,
        0.38694976],
       [0.        , 0.02202456, 0.04611458, 0.06994037, 0.09079121,
        0.11128068, 0.12969586, 0.1477241 , 0.16608925, 0.18335718,
        0.19912353],
       [0.        , 0.0152352 , 0.02938215, 0.0406206 , 0.05192939,
        0.06027329, 0.06896791, 0.07750892, 0.08537324, 0.09258988,
        0.09787536],
       [0.        , 0.10626758, 0.15659668, 0.19294849, 0.22168353,
        0.245376  , 0.26801521, 0.28640325, 0.30340806, 0.32156135,
        0.33789573],
       [0.        , 0.11159575, 0.16091498, 0.19388616, 0.21729044,
        0.23605413, 0.25113811, 0.2661719 , 0.2778955 , 0.28963228,
        0.30163148],
       [0.        , 0.1182868 , 0.17790863, 0.21615255, 0.24577387,
        0.26857717, 0.28872751, 0.30606745, 0.32052841, 0.33301032,
        0.34804008],
       [0.        , 0.08551742, 0.14556299, 0.19011043, 0.22719692,
        0.25687183, 0.28528246, 0.3103344 , 0.33589355, 0.35620891,
        0.37787913]])
    '''
    # Find allocation for sample budget
    allocArr = forwardAllocateWithBudget(avgUtilMat, int(sampBudget / testInt))
    designArr = allocArr / np.sum(allocArr, axis=0)
    # Get utility for this allocation at the sample budget
    utilDict.update({'method': 'weightsNodeDraw4linear'})
    compUtilList, unifUtilList, origUtilList = [], [], []
    numReps = 5
    for rep in range(numReps):
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New Bayes draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        print('Generating loss matrix...')
        tempLossMat = lf.lossMatSetBayesDraws(dictTemp['postSamples'], lossDict.copy(), lossDict['bayesDraws'])
        tempLossDict = lossDict.copy()
        tempLossDict.update({'lossMat': tempLossMat})
        newBayesDraws, newLossMat = lf.add_cand_neighbors(tempLossDict.copy(), CSdict3['postSamples'],
                                                          dictTemp['postSamples'])
        tempLossDict.update({'bayesDraws': newBayesDraws, 'lossMat': newLossMat})
        baseLoss = (np.sum(newLossMat, axis=1) / newLossMat.shape[1]).min()
        # Get a new set of data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        currCompUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[designArr],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Heuristic utility: ' + str(currCompUtil))
        compUtilList.append(currCompUtil)
        '''27-MAR
        compUtilList = [0.7432156084325694, 0.7533995943632834, 0.8083659961914282, 0.8018781170413041, 0.818762643067311]
        '''
        # Find the equivalent uniform allocation
        currUnifUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[unifDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Uniform utility: ' + str(currUnifUtil))
        unifUtilList.append([currUnifUtil])
        unifAdd, contUnif, unifCount = 0, False, 0
        if currUnifUtil < currCompUtil:
            contUnif = True
        while contUnif:
            unifAdd += testInt
            print('Adding ' + str(unifAdd) + ' for uniform')
            currUnifUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[unifDes], numtests=sampBudget + unifAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currUnifUtil))
            unifUtilList[rep].append(currUnifUtil)
            if currUnifUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if unifCount < 3:
                    unifCount += 1
                else:
                    contUnif = False
        '''27-MAR
        unifUtilList = [[0.6401150030171214, 0.680511101667256, 0.7353478324313336, 0.7847287457584207, 0.8249941597549935, 0.8727995429319657, 0.9045773628288303], [0.6473188146061188, 0.6841426118768474, 0.7352355880687798, 0.7857586224773803, 0.8296106131828695, 0.8786752556774124, 0.925118784736505], [0.6885482846574646, 0.7302275139322321, 0.79928483431376, 0.8320331037257636, 0.8855764255098064, 0.9329258049525531, 0.9775937051209684], [0.6881292515000261, 0.7388746657851044, 0.7850269709982571, 0.8322131966492652, 0.8859892139750083, 0.9285124418691355, 0.9646556348935289], [0.7201045314066832, 0.7520023205480868, 0.8070863069990861, 0.8489519587061114, 0.9000556996739659, 0.945466102549644, 0.9923269400602446]]
        '''
        # Find the equivalent rudimentary allocation
        currOrigUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[origDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Rudimentary utility: ' + str(currOrigUtil))
        origUtilList.append([currOrigUtil])
        origAdd, contOrig, origCount = 0, False, 0
        if currOrigUtil < currCompUtil:
            contOrig = True
        while contOrig:
            origAdd += testInt * 3
            print('Adding ' + str(origAdd) + ' for rudimentary')
            currOrigUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[origDes], numtests=sampBudget + origAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currOrigUtil))
            origUtilList[rep].append(currOrigUtil)
            if currOrigUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if origCount < 3:
                    origCount += 1
                else:
                    contOrig = False
        '''27-MAR
        origUtilList = [[0.2351039040184526, 0.3041528437199119, 0.37896079339182354, 0.4435125964015052, 0.5057788798723521, 0.590184139609665, 0.6604162464465051, 0.7221817585425603, 0.7972858141752583, 0.863466805815766, 0.9594325724867501, 0.9990351319769855], [0.23213561171785546, 0.3065048904660861, 0.3646349207851438, 0.43745827446537255, 0.5069613250597715, 0.5858184023750308, 0.646801589369252, 0.7282070385421817, 0.7864717380699302, 0.8621472959806211, 0.9407250761376824, 1.0039883543704216], [0.2708665847956553, 0.3483891999674755, 0.41963618421645243, 0.48828374162167254, 0.565516015645187, 0.6329457720077292, 0.7228201393320326, 0.7767016844497605, 0.8579315699660719, 0.9189385231887157, 1.023019617193031, 1.063123377689033], [0.26626001413303957, 0.3305104762539397, 0.41138135027620315, 0.48118580155369006, 0.5559194135213561, 0.631274002197272, 0.6904137067857476, 0.7742698375743933, 0.8535047581071771, 0.9119396622370459, 1.0050873108896212, 1.059972967164387], [0.2788873547400499, 0.349305554336242, 0.4102006050638538, 0.4806765837952911, 0.5576273088133883, 0.6292663438397579, 0.697749156255485, 0.7855235777573406, 0.8423496315665737, 0.9252075748101749, 1.005851294583234, 1.0469468616541628]]
        '''
    compAvg = np.average(compUtilList)
    # Locate closest sample point for uniform and rudimentary to compAvg
    minListLen = np.min([len(i) for i in unifUtilList])
    unifUtilArr = np.array([i[:minListLen] for i in unifUtilList])
    unifAvgArr = np.average(unifUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(unifAvgArr.tolist()) if val > compAvg)
    unifSampSaved = round((compAvg - unifAvgArr[kInd - 1]) / (unifAvgArr[kInd] - unifAvgArr[kInd - 1]) * testInt) + (
            kInd - 1) * testInt
    print(unifSampSaved)
    '''23-MAR: 222 saved'''
    # Rudimentary
    minListLen = np.min([len(i) for i in origUtilList])
    origUtilArr = np.array([i[:minListLen] for i in origUtilList])
    origAvgArr = np.average(origUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(origAvgArr.tolist()) if val > compAvg)
    origSampSaved = round(
        (compAvg - origAvgArr[kInd - 1]) / (origAvgArr[kInd] - origAvgArr[kInd - 1]) * testInt * 3) + (
                            kInd - 1) * testInt * 3
    print(origSampSaved)
    '''27-MAR: 222 saved'''

    # Use different loss parameters
    ###########
    # todo: checkSlope = 0.9
    ###########
    paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.9,
                                                  marketvec=np.ones(numTN + numSN), candneighnum=1000)

    numtargetdraws, numDataDraws = 5100, 5000

    # Find heuristic allocation first
    utilDict = {'method': 'weightsNodeDraw3linear'}

    numReps = 5
    utilMatList = []
    # set testMax to highest expected allocation for any one node
    testMax, testInt = 100, 10
    for rep in range(numReps):
        # Withdraw a subset of MCMC prior draws
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New loss draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        # Get new data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        # Get marginal utilities at each test node
        currMargUtilMat = GetMargUtilAtNodes(dictTemp.copy(), testMax, testInt, lossDict.copy(), utilDict.copy(),
                                             masterDraws=CSdict3['postSamples'], printUpdate=True)
        print(repr(currMargUtilMat))
        utilMatList.append(currMargUtilMat)
    '''22-MAR run
    utilMatList = [np.array([[0.        , 0.03075288, 0.0565451 , 0.0746321 , 0.09659966,
        0.11144783, 0.12698643, 0.14335713, 0.15641624, 0.1671006 ,
        0.18339319],
       [0.        , 0.06572466, 0.11378502, 0.14692445, 0.17433724,
        0.20018974, 0.22139891, 0.23813687, 0.25383738, 0.26964095,
        0.2851109 ],
       [0.        , 0.00955448, 0.0218536 , 0.0369972 , 0.05127719,
        0.06266118, 0.07858402, 0.09027435, 0.10331226, 0.11345728,
        0.12888646],
       [0.        , 0.00292594, 0.0114615 , 0.01887061, 0.03076254,
        0.03732542, 0.04456915, 0.05187669, 0.05964612, 0.06482972,
        0.06894663],
       [0.        , 0.09000419, 0.12589776, 0.1496807 , 0.16664233,
        0.18527906, 0.19807776, 0.21022732, 0.22143811, 0.23512691,
        0.24555148],
       [0.        , 0.05694585, 0.09415969, 0.12074849, 0.14017877,
        0.1554259 , 0.16750299, 0.17997499, 0.19218658, 0.19868762,
        0.20879519],
       [0.        , 0.07948077, 0.11859158, 0.14214012, 0.16206143,
        0.17784438, 0.19096196, 0.20068877, 0.21200407, 0.22172608,
        0.23045235],
       [0.        , 0.08349351, 0.11998393, 0.14773368, 0.17195863,
        0.19004298, 0.21192951, 0.22522178, 0.24097913, 0.25461054,
        0.26861671]]), np.array([[0.        , 0.0354193 , 0.05683372, 0.0782085 , 0.09669785,
        0.11198916, 0.13005225, 0.14684066, 0.15800112, 0.17123796,
        0.18228022],
       [0.        , 0.08267887, 0.12932618, 0.16144411, 0.191978  ,
        0.21518033, 0.23835322, 0.25391705, 0.26936901, 0.28910589,
        0.30199693],
       [0.        , 0.01891859, 0.03628884, 0.05092801, 0.06371052,
        0.07499908, 0.08887482, 0.09660051, 0.11284919, 0.12393414,
        0.13437873],
       [0.        , 0.02670743, 0.04023619, 0.05012686, 0.05686753,
        0.06523699, 0.07319036, 0.07950473, 0.08326539, 0.09111585,
        0.09381023],
       [0.        , 0.10111285, 0.1400115 , 0.16537856, 0.18529892,
        0.20262119, 0.21805694, 0.23197443, 0.24246257, 0.25466958,
        0.26796038],
       [0.        , 0.08938192, 0.1227288 , 0.14629541, 0.16152294,
        0.17283923, 0.185839  , 0.19607428, 0.20534831, 0.21425957,
        0.22112171],
       [0.        , 0.08959404, 0.12935687, 0.15235996, 0.17156702,
        0.18549643, 0.20298363, 0.21328721, 0.22521348, 0.23644074,
        0.24501391],
       [0.        , 0.09804337, 0.13615147, 0.16147446, 0.18473583,
        0.20338647, 0.21960846, 0.23761779, 0.25168935, 0.26537889,
        0.27647035]]), np.array([[0.        , 0.03513559, 0.06497641, 0.0882293 , 0.10427151,
        0.12455744, 0.13967437, 0.15553761, 0.16846617, 0.17881889,
        0.19385513],
       [0.        , 0.0828142 , 0.13571465, 0.17237   , 0.20211414,
        0.22464037, 0.24597637, 0.26405134, 0.28334982, 0.299907  ,
        0.31328436],
       [0.        , 0.0179283 , 0.03548164, 0.04584786, 0.05990511,
        0.07347835, 0.08506584, 0.09387507, 0.11142229, 0.11924694,
        0.12852292],
       [0.        , 0.02231515, 0.03931223, 0.04985942, 0.05796447,
        0.06589037, 0.0726527 , 0.07639682, 0.08806971, 0.08943918,
        0.09785195],
       [0.        , 0.10918078, 0.14608481, 0.17116112, 0.18998398,
        0.20613658, 0.22084961, 0.23323429, 0.24464794, 0.25703572,
        0.26354536],
       [0.        , 0.08590209, 0.11398255, 0.1344181 , 0.14981653,
        0.16140318, 0.17338802, 0.18165752, 0.18994836, 0.2000064 ,
        0.20690274],
       [0.        , 0.12069917, 0.1658702 , 0.19364535, 0.21143149,
        0.22928947, 0.24181656, 0.25333774, 0.26320408, 0.27221885,
        0.27904696],
       [0.        , 0.09974374, 0.13489977, 0.15909539, 0.18164873,
        0.20194355, 0.21698583, 0.23270839, 0.24835775, 0.26025708,
        0.2773961 ]]), np.array([[0.        , 0.03196404, 0.05760283, 0.08082335, 0.10020703,
        0.11885041, 0.13620186, 0.14771982, 0.16333645, 0.17561741,
        0.18685865],
       [0.        , 0.0680918 , 0.11569436, 0.15115868, 0.18104093,
        0.2073635 , 0.22808381, 0.24730727, 0.26399898, 0.28340489,
        0.2996577 ],
       [0.        , 0.02681898, 0.04852998, 0.06440018, 0.08369152,
        0.09539988, 0.1090851 , 0.12184843, 0.12987958, 0.14316872,
        0.15247667],
       [0.        , 0.01573695, 0.02839274, 0.03708781, 0.04523733,
        0.05410254, 0.06181502, 0.07061853, 0.07495894, 0.07920308,
        0.08275896],
       [0.        , 0.07078331, 0.11629243, 0.14702114, 0.16860698,
        0.19187182, 0.20756216, 0.22173809, 0.2340515 , 0.24491832,
        0.25908863],
       [0.        , 0.09205322, 0.12629978, 0.14458926, 0.16438938,
        0.17519035, 0.18756234, 0.19720792, 0.20605966, 0.21606601,
        0.22231906],
       [0.        , 0.0967696 , 0.13817887, 0.16422165, 0.18711883,
        0.20314808, 0.21525835, 0.23043861, 0.24159869, 0.24965842,
        0.25771903],
       [0.        , 0.08216984, 0.12600731, 0.15360078, 0.17772598,
        0.19768218, 0.21718576, 0.23357722, 0.24831773, 0.26046049,
        0.27608534]]), np.array([[0.        , 0.03471602, 0.05691319, 0.08416546, 0.10539777,
        0.12394867, 0.13983056, 0.15616778, 0.16871067, 0.18467579,
        0.18898689],
       [0.        , 0.06682056, 0.1182276 , 0.15460222, 0.18697697,
        0.21012775, 0.22805276, 0.25022063, 0.27215807, 0.2826246 ,
        0.30061884],
       [0.        , 0.01873002, 0.03693331, 0.05506479, 0.07187386,
        0.08828952, 0.10252277, 0.11629563, 0.12957653, 0.14289278,
        0.15120639],
       [0.        , 0.01001747, 0.02626064, 0.03763449, 0.04872725,
        0.05798881, 0.06516531, 0.06999929, 0.08018826, 0.0858269 ,
        0.09380951],
       [0.        , 0.07485467, 0.11746858, 0.14472599, 0.16704539,
        0.18309145, 0.19973911, 0.2154869 , 0.22485419, 0.23773503,
        0.25144521],
       [0.        , 0.08289615, 0.11494089, 0.13634728, 0.15149236,
        0.16650355, 0.17643583, 0.18629472, 0.19552417, 0.20491242,
        0.2144102 ],
       [0.        , 0.08451949, 0.11908208, 0.14377945, 0.16103633,
        0.17597314, 0.19321247, 0.20513748, 0.21709345, 0.2295761 ,
        0.23638159],
       [0.        , 0.07698847, 0.12337949, 0.15410276, 0.18050144,
        0.2038486 , 0.21785035, 0.23948056, 0.25477226, 0.27003047,
        0.28568715]])]
    '''
    # Get average utility matrix
    avgUtilMat = np.average(np.array(utilMatList), axis=0)
    '''22-MAR
    avgUtilMat = np.array([[0.        , 0.03359756, 0.05857425, 0.08121174, 0.10063477,
        0.1181587 , 0.13454909, 0.1499246 , 0.16298613, 0.17549013,
        0.18707481],
       [0.        , 0.07322602, 0.12254956, 0.15729989, 0.18728946,
        0.21150034, 0.23237301, 0.25072663, 0.26854265, 0.28493667,
        0.30013375],
       [0.        , 0.01839007, 0.03581747, 0.05064761, 0.06609164,
        0.0789656 , 0.09282651, 0.1037788 , 0.11740797, 0.12853997,
        0.13909423],
       [0.        , 0.01554059, 0.02913266, 0.03871584, 0.04791182,
        0.05610883, 0.06347851, 0.06967921, 0.07722568, 0.08208295,
        0.08743546],
       [0.        , 0.08918716, 0.12915102, 0.1555935 , 0.17551552,
        0.19380002, 0.20885712, 0.22253221, 0.23349086, 0.24589711,
        0.25751821],
       [0.        , 0.08143585, 0.11442234, 0.13647971, 0.15347999,
        0.16627244, 0.17814564, 0.18824189, 0.19781342, 0.2067864 ,
        0.21470978],
       [0.        , 0.09421261, 0.13421592, 0.15922931, 0.17864302,
        0.1943503 , 0.2088466 , 0.22057796, 0.23182275, 0.24192404,
        0.24972277],
       [0.        , 0.08808778, 0.12808439, 0.15520141, 0.17931412,
        0.19938076, 0.21671198, 0.23372115, 0.24882324, 0.26214749,
        0.27685113]])
    '''
    # Find allocation for sample budget
    allocArr = forwardAllocateWithBudget(avgUtilMat, int(sampBudget / testInt))
    designArr = allocArr / np.sum(allocArr, axis=0)
    # Get utility for this allocation at the sample budget
    utilDict.update({'method': 'weightsNodeDraw4linear'})
    compUtilList, unifUtilList, origUtilList = [], [], []
    numReps = 5
    for rep in range(numReps):
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New Bayes draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        print('Generating loss matrix...')
        tempLossMat = lf.lossMatSetBayesDraws(dictTemp['postSamples'], lossDict.copy(), lossDict['bayesDraws'])
        tempLossDict = lossDict.copy()
        tempLossDict.update({'lossMat': tempLossMat})
        newBayesDraws, newLossMat = lf.add_cand_neighbors(tempLossDict.copy(), CSdict3['postSamples'],
                                                          dictTemp['postSamples'])
        tempLossDict.update({'bayesDraws': newBayesDraws, 'lossMat': newLossMat})
        baseLoss = (np.sum(newLossMat, axis=1) / newLossMat.shape[1]).min()
        # Get a new set of data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        currCompUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[designArr],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Heuristic utility: ' + str(currCompUtil))
        compUtilList.append(currCompUtil)
        '''27-MAR
        compUtilList = [0.5333803346186698, 0.5260696688490731, 0.49086504396210096, 0.5884214991076298, 0.4990536617645125]
        '''
        # Find the equivalent uniform allocation
        currUnifUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[unifDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Uniform utility: ' + str(currUnifUtil))
        unifUtilList.append([currUnifUtil])
        unifAdd, contUnif, unifCount = 0, False, 0
        if currUnifUtil < currCompUtil:
            contUnif = True
        while contUnif:
            unifAdd += testInt
            print('Adding ' + str(unifAdd) + ' for uniform')
            currUnifUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[unifDes], numtests=sampBudget + unifAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currUnifUtil))
            unifUtilList[rep].append(currUnifUtil)
            if currUnifUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if unifCount < 3:
                    unifCount += 1
                else:
                    contUnif = False
        '''22-MAR
        unifUtilList = [[0.45619981426263223, 0.4826494324506698, 0.5214176876397674, 0.551280935553899, 0.582380766637951, 0.6238850085052783, 0.6373833951886545], [0.44937974805181247, 0.48016015465672135, 0.5191147308410589, 0.5490181250624384, 0.5804806385910184, 0.6188409430101092, 0.6504305220096631], [0.4169213174175366, 0.4425745498533247, 0.48105295595448494, 0.5118634066480952, 0.5456817116639705, 0.5865951903568396, 0.6218579763811234], [0.5109188195346426, 0.5406834579001845, 0.5751536225961216, 0.614688095432018, 0.6453112388507298, 0.6798168280106447, 0.7070988417943371], [0.4231337728621587, 0.4561048851580223, 0.49801053069431633, 0.5255250502708715, 0.5586586487137031, 0.5899428398362883, 0.6220781289591542]]
        '''
        # Find the equivalent rudimentary allocation
        currOrigUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[origDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Rudimentary utility: ' + str(currOrigUtil))
        origUtilList.append([currOrigUtil])
        origAdd, contOrig, origCount = 0, False, 0
        if currOrigUtil < currCompUtil:
            contOrig = True
        while contOrig:
            origAdd += testInt * 3
            print('Adding ' + str(origAdd) + ' for rudimentary')
            currOrigUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[origDes], numtests=sampBudget + origAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currOrigUtil))
            origUtilList[rep].append(currOrigUtil)
            if currOrigUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if origCount < 3:
                    origCount += 1
                else:
                    contOrig = False
        '''22-MAR
        origUtilList = [[0.1771285854745499, 0.22721685654367585, 0.2773520043743769, 0.3235600060398993, 0.3742305734024005, 0.4188874249819188, 0.470602698920374, 0.519161887731185, 0.5793861569520615, 0.6379202346405126, 0.686351859972933, 0.7234584045850534], [0.17845165434936527, 0.22556363066182472, 0.266248458357806, 0.32015020238256664, 0.3687215157191148, 0.4295378347819665, 0.47780126227205333, 0.5293701289169124, 0.584502216026408, 0.6324346795761184, 0.6962716459267169], [0.14894282662777059, 0.19120904571176212, 0.23434535985727267, 0.28363479464687424, 0.3381946022591009, 0.38002260874447824, 0.42605195206557145, 0.4791002081458582, 0.5394273441464841, 0.5892702440953279, 0.651989852053874, 0.6865814038881259], [0.21556947223852063, 0.2680010953473895, 0.3168566040135725, 0.3711198236293032, 0.41526324655170166, 0.4780279363477984, 0.5286459984875913, 0.5860944746212651, 0.6432426107980524, 0.6970847312727364, 0.7585523987773417, 0.7847025644790935], [0.1643876938328952, 0.21746619288897628, 0.2629619673414827, 0.3101738373778997, 0.3601088656843663, 0.41959916938922026, 0.4567484767586758, 0.5128724216072489, 0.5666912419859118, 0.6125458293187296, 0.6684115794133354]]
        '''
    compAvg = np.average(compUtilList)
    # Locate closest sample point for uniform and rudimentary to compAvg
    minListLen = np.min([len(i) for i in unifUtilList])
    unifUtilArr = np.array([i[:minListLen] for i in unifUtilList])
    unifAvgArr = np.average(unifUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(unifAvgArr.tolist()) if val > compAvg)
    unifSampSaved = round((compAvg - unifAvgArr[kInd - 1]) / (unifAvgArr[kInd] - unifAvgArr[kInd - 1]) * testInt) + (
            kInd - 1) * testInt
    print(unifSampSaved)
    '''27-MAR: 23 saved'''
    # Rudimentary
    minListLen = np.min([len(i) for i in origUtilList])
    origUtilArr = np.array([i[:minListLen] for i in origUtilList])
    origAvgArr = np.average(origUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(origAvgArr.tolist()) if val > compAvg)
    origSampSaved = round(
        (compAvg - origAvgArr[kInd - 1]) / (origAvgArr[kInd] - origAvgArr[kInd - 1]) * testInt * 3) + (
                            kInd - 1) * testInt * 3
    print(origSampSaved)
    '''27-MAR: 211 saved'''

    # Use different loss parameters
    ###########
    # todo: underWt = 1.
    ###########
    paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=1., riskthreshold=0.15, riskslope=0.6,
                                                  marketvec=np.ones(numTN + numSN), candneighnum=1000)

    numtargetdraws, numDataDraws = 5100, 5000

    # Find heuristic allocation first
    utilDict = {'method': 'weightsNodeDraw3linear'}

    numReps = 5
    utilMatList = []
    # set testMax to highest expected allocation for any one node
    testMax, testInt = 100, 10
    for rep in range(numReps):
        # Withdraw a subset of MCMC prior draws
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New loss draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        # Get new data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        # Get marginal utilities at each test node
        currMargUtilMat = GetMargUtilAtNodes(dictTemp.copy(), testMax, testInt, lossDict.copy(), utilDict.copy(),
                                             masterDraws=CSdict3['postSamples'], printUpdate=True)
        print(repr(currMargUtilMat))
        utilMatList.append(currMargUtilMat)
    '''22-MAR run
    utilMatList = [np.array([[0.        , 0.01088367, 0.02366804, 0.03522634, 0.04518001,
        0.05429115, 0.06170353, 0.06987887, 0.07758497, 0.08409471,
        0.0922609 ],
       [0.        , 0.02847574, 0.0513879 , 0.06912239, 0.08496897,
        0.10035419, 0.11355628, 0.12441351, 0.13470912, 0.14625519,
        0.15403402],
       [0.        , 0.01053215, 0.02151598, 0.03260515, 0.04422627,
        0.05547771, 0.06394887, 0.07529553, 0.0832292 , 0.09203173,
        0.0991013 ],
       [0.        , 0.00188317, 0.00548749, 0.00859108, 0.01135168,
        0.01409556, 0.01677248, 0.0195999 , 0.02161954, 0.02594214,
        0.02767769],
       [0.        , 0.03049416, 0.049828  , 0.0631742 , 0.07533541,
        0.08376526, 0.09311442, 0.10062907, 0.107332  , 0.11463545,
        0.11989065],
       [0.        , 0.02400419, 0.03740073, 0.04802847, 0.05587183,
        0.06527252, 0.07056265, 0.0763267 , 0.08197542, 0.08777925,
        0.09148122],
       [0.        , 0.03252395, 0.05365362, 0.06959628, 0.08016599,
        0.08969886, 0.09746171, 0.10527861, 0.11045559, 0.11641219,
        0.12135129],
       [0.        , 0.03136431, 0.05095531, 0.0662625 , 0.080798  ,
        0.09303958, 0.10274926, 0.11246973, 0.12249423, 0.13074985,
        0.13929619]]), 
        np.array([[0.        , 0.01107078, 0.02300029, 0.0353717 , 0.04388474,
        0.05204505, 0.05975575, 0.06891929, 0.07592609, 0.08245542,
        0.08990451],
       [0.        , 0.03050141, 0.05376334, 0.07503077, 0.09171872,
        0.10622402, 0.11751404, 0.13096928, 0.13915777, 0.15069926,
        0.15872302],
       [0.        , 0.00981586, 0.02065059, 0.03101965, 0.04074383,
        0.05165281, 0.0595977 , 0.06952978, 0.07626638, 0.08459924,
        0.09467509],
       [0.        , 0.00214673, 0.00480582, 0.00755866, 0.01111187,
        0.01370206, 0.01586223, 0.01876204, 0.02054649, 0.02307256,
        0.02583611],
       [0.        , 0.03174166, 0.05244254, 0.06575972, 0.07815617,
        0.08719497, 0.0957752 , 0.10338809, 0.11054561, 0.11693476,
        0.12291782],
       [0.        , 0.02488944, 0.03989517, 0.05059822, 0.05879231,
        0.06766035, 0.07483272, 0.08171699, 0.0861234 , 0.09120108,
        0.09633551],
       [0.        , 0.0302982 , 0.0510324 , 0.06588003, 0.07705555,
        0.08644201, 0.09436592, 0.10063807, 0.10678855, 0.1129641 ,
        0.11792721],
       [0.        , 0.02212158, 0.04253936, 0.05790151, 0.07136883,
        0.08418088, 0.09417643, 0.10260827, 0.11313299, 0.12310535,
        0.1300502 ]]),
        np.array([[0.        , 0.00919342, 0.01986591, 0.03190652, 0.04259505,
        0.04991739, 0.05766652, 0.06570681, 0.0719042 , 0.07835804,
        0.08454289],
       [0.        , 0.02440816, 0.04470776, 0.06281298, 0.08046134,
        0.09369081, 0.10632461, 0.11912579, 0.12893504, 0.1372843 ,
        0.14728741],
       [0.        , 0.00569074, 0.01543616, 0.02550775, 0.03485174,
        0.04475004, 0.05449078, 0.06378026, 0.07022076, 0.08111976,
        0.08809707],
       [0.        , 0.00170505, 0.00374647, 0.00659166, 0.01011973,
        0.01265864, 0.01471206, 0.01667299, 0.01990643, 0.02144041,
        0.02341426],
       [0.        , 0.02398414, 0.04117571, 0.05505616, 0.06724893,
        0.07697693, 0.08557795, 0.09478825, 0.10103732, 0.10862509,
        0.11434237],
       [0.        , 0.0228718 , 0.0375308 , 0.04822571, 0.05731266,
        0.063901  , 0.07061442, 0.07549924, 0.08167641, 0.08505161,
        0.09018776],
       [0.        , 0.02513935, 0.04520408, 0.0577177 , 0.0683569 ,
        0.07837852, 0.0847792 , 0.09189106, 0.0993334 , 0.1039948 ,
        0.10879921],
       [0.        , 0.02136997, 0.04054589, 0.05529073, 0.0690352 ,
        0.08078239, 0.09134324, 0.10204092, 0.11115004, 0.12026925,
        0.12848079]]),
        np.array([[0.        , 0.01039543, 0.02256671, 0.03378684, 0.04394314,
        0.05392682, 0.06069337, 0.06859636, 0.07774013, 0.08339359,
        0.08985891],
       [0.        , 0.02914781, 0.05202363, 0.0738285 , 0.08981818,
        0.10410924, 0.11836312, 0.13087746, 0.14027163, 0.15074776,
        0.15993027],
       [0.        , 0.01071109, 0.0215703 , 0.031148  , 0.04163808,
        0.05503134, 0.06205651, 0.07170679, 0.08095795, 0.08876356,
        0.09605018],
       [0.        , 0.00180101, 0.00504892, 0.0081293 , 0.0104461 ,
        0.01301865, 0.01585576, 0.01900689, 0.02141951, 0.02365678,
        0.02593692],
       [0.        , 0.02888871, 0.04795943, 0.06193616, 0.07344362,
        0.08219015, 0.09102029, 0.09927703, 0.10544726, 0.11178248,
        0.11950148],
       [0.        , 0.02485113, 0.04132151, 0.05101368, 0.06034551,
        0.06830007, 0.07563689, 0.07996753, 0.08639545, 0.09084837,
        0.09560016],
       [0.        , 0.02905179, 0.05008968, 0.06386594, 0.07478612,
        0.08452121, 0.09246634, 0.09908938, 0.10537702, 0.11100315,
        0.11661527],
       [0.        , 0.028703  , 0.04959113, 0.06657348, 0.08088227,
        0.09397491, 0.10519616, 0.11411599, 0.12224504, 0.13211056,
        0.14173876]]),
        np.array([[0.        , 0.01814277, 0.03145738, 0.04374345, 0.05372269,
        0.0629357 , 0.07153863, 0.07820898, 0.08637542, 0.09180568,
        0.09895402],
       [0.        , 0.03630651, 0.06131392, 0.08051863, 0.09582164,
        0.11221689, 0.12406721, 0.13602584, 0.14612171, 0.1554203 ,
        0.16321455],
       [0.        , 0.01515037, 0.02813505, 0.03986089, 0.05153592,
        0.06062684, 0.07144059, 0.0805404 , 0.08846696, 0.09598719,
        0.10578462],
       [0.        , 0.00534265, 0.01035942, 0.01432403, 0.0182365 ,
        0.02194153, 0.02551831, 0.02790565, 0.03081279, 0.03350372,
        0.03614253],
       [0.        , 0.04155905, 0.06192909, 0.07619405, 0.08723579,
        0.09634616, 0.10449396, 0.11194988, 0.11927403, 0.12469818,
        0.13243729],
       [0.        , 0.0320574 , 0.0496676 , 0.06283267, 0.07176745,
        0.07853427, 0.08513404, 0.0915232 , 0.09578596, 0.10150947,
        0.1070271 ],
       [0.        , 0.04073478, 0.06195811, 0.07678302, 0.08745174,
        0.09692602, 0.10480846, 0.11128112, 0.11782259, 0.12389517,
        0.12909392],
       [0.        , 0.0361526 , 0.05839154, 0.07592405, 0.08833617,
        0.10106856, 0.11350123, 0.12072861, 0.13080787, 0.13956721,
        0.14896624]])]
    '''
    # Get average utility matrix
    avgUtilMat = np.average(np.array(utilMatList), axis=0)
    '''22-MAR
    avgUtilMat = np.array([[0.        , 0.01193721, 0.02411166, 0.03600697, 0.04586512,
        0.05462322, 0.06227156, 0.07026206, 0.07790616, 0.08402149,
        0.09110425],
       [0.        , 0.02976793, 0.05263931, 0.07226266, 0.08855777,
        0.10331903, 0.11596505, 0.12828238, 0.13783905, 0.14808136,
        0.15663786],
       [0.        , 0.01038004, 0.02146161, 0.03202829, 0.04259917,
        0.05350775, 0.06230689, 0.07217055, 0.07982825, 0.0885003 ,
        0.09674165],
       [0.        , 0.00257572, 0.00588963, 0.00903895, 0.01225318,
        0.01508329, 0.01774417, 0.02038949, 0.02286095, 0.02552312,
        0.0278015 ],
       [0.        , 0.03133355, 0.05066695, 0.06442406, 0.07628398,
        0.08529469, 0.09399636, 0.10200647, 0.10872724, 0.11533519,
        0.12181792],
       [0.        , 0.02573479, 0.04116316, 0.05213975, 0.06081795,
        0.06873364, 0.07535614, 0.08100673, 0.08639133, 0.09127796,
        0.09612635],
       [0.        , 0.03154961, 0.05238758, 0.06676859, 0.07756326,
        0.08719332, 0.09477633, 0.10163565, 0.10795543, 0.11365388,
        0.11875738],
       [0.        , 0.02794229, 0.04840465, 0.06439045, 0.07808409,
        0.09060926, 0.10139326, 0.1103927 , 0.11996603, 0.12916045,
        0.13770644]])
    '''
    # Find allocation for sample budget
    allocArr = forwardAllocateWithBudget(avgUtilMat, int(sampBudget / testInt))
    designArr = allocArr / np.sum(allocArr, axis=0)
    # Get utility for this allocation at the sample budget
    utilDict.update({'method': 'weightsNodeDraw4linear'})
    compUtilList, unifUtilList, origUtilList = [], [], []
    numReps = 5
    for rep in range(numReps):
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New Bayes draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        print('Generating loss matrix...')
        tempLossMat = lf.lossMatSetBayesDraws(dictTemp['postSamples'], lossDict.copy(), lossDict['bayesDraws'])
        tempLossDict = lossDict.copy()
        tempLossDict.update({'lossMat': tempLossMat})
        newBayesDraws, newLossMat = lf.add_cand_neighbors(tempLossDict.copy(), CSdict3['postSamples'],
                                                          dictTemp['postSamples'])
        tempLossDict.update({'bayesDraws': newBayesDraws, 'lossMat': newLossMat})
        baseLoss = (np.sum(newLossMat, axis=1) / newLossMat.shape[1]).min()
        # Get a new set of data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        currCompUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[designArr],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Heuristic utility: ' + str(currCompUtil))
        compUtilList.append(currCompUtil)
        '''27-MAR
        compUtilList = [0.22897310553090455, 0.2229216081478005, 0.23048561510424914, 0.23401807909534877, 0.23256665688393174]
        '''
        # Find the equivalent uniform allocation
        currUnifUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[unifDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Uniform utility: ' + str(currUnifUtil))
        unifUtilList.append([currUnifUtil])
        unifAdd, contUnif, unifCount = 0, False, 0
        if currUnifUtil < currCompUtil:
            contUnif = True
        while contUnif:
            unifAdd += testInt
            print('Adding ' + str(unifAdd) + ' for uniform')
            currUnifUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[unifDes], numtests=sampBudget + unifAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currUnifUtil))
            unifUtilList[rep].append(currUnifUtil)
            if currUnifUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if unifCount < 3:
                    unifCount += 1
                else:
                    contUnif = False
        '''27-MAR
        unifUtilList = [[0.19123085194436173, 0.20937143669903335, 0.22822394438238902, 0.24813296921750339, 0.2662768936675388, 0.2822819169209825, 0.2984536857367328], [0.18404158964962858, 0.19882250096100207, 0.22180693072306634, 0.23788166368115626, 0.25365882244520166, 0.2735610620285851, 0.2904462236139993], [0.19072899880696093, 0.210760951415343, 0.2308785547315615, 0.2460543852342847, 0.2680274412099972, 0.2832872143106875], [0.19503974322281903, 0.2114793305416296, 0.23242172093441504, 0.24988197632756348, 0.26757952670908836, 0.2872950080711525, 0.30041310479227845], [0.1974980765939427, 0.21369727768850022, 0.23069340355742352, 0.25196398615460613, 0.26780991538729304, 0.2871342285434624, 0.30511082453983707]]
        '''
        # Find the equivalent rudimentary allocation
        currOrigUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[origDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Rudimentary utility: ' + str(currOrigUtil))
        origUtilList.append([currOrigUtil])
        origAdd, contOrig, origCount = 0, False, 0
        if currOrigUtil < currCompUtil:
            contOrig = True
        while contOrig:
            origAdd += testInt * 3
            print('Adding ' + str(origAdd) + ' for rudimentary')
            currOrigUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[origDes], numtests=sampBudget + origAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currOrigUtil))
            origUtilList[rep].append(currOrigUtil)
            if currOrigUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if origCount < 3:
                    origCount += 1
                else:
                    contOrig = False
        '''27-MAR
        origUtilList = [[0.0915496818226933, 0.119614567943229, 0.14750302797548276, 0.1783429285530127, 0.20440289148262747, 0.2348912479179519, 0.25876851962694025, 0.28870965376580604, 0.3157181607988251], [0.09716052680394127, 0.12612727504052557, 0.15170014502964957, 0.17817065192089943, 0.20948160948699912, 0.23806163906683908, 0.2627645396649796, 0.291999745265098, 0.31804078300771566], [0.10131500661459292, 0.13228040024121612, 0.1575242121657876, 0.1891691343479276, 0.21680139474504223, 0.2476049088606822, 0.27147063296187124, 0.30132027759126956, 0.3292664795906244], [0.09962772455216462, 0.1309994408235562, 0.15996095376279862, 0.18712222108597554, 0.2162885829570491, 0.241261926660842, 0.2704670596006613, 0.30108255182973287, 0.3279530433993063], [0.10389186069930378, 0.13234240011566012, 0.1614195811654271, 0.1900843269474064, 0.2177917677797696, 0.24394937506937509, 0.269961316321333, 0.3007948652389467, 0.32766747593827583]]
        '''
    compAvg = np.average(compUtilList)
    # Locate closest sample point for uniform and rudimentary to compAvg
    minListLen = np.min([len(i) for i in unifUtilList])
    unifUtilArr = np.array([i[:minListLen] for i in unifUtilList])
    unifAvgArr = np.average(unifUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(unifAvgArr.tolist()) if val > compAvg)
    unifSampSaved = round((compAvg - unifAvgArr[kInd - 1]) / (unifAvgArr[kInd] - unifAvgArr[kInd - 1]) * testInt) + (
            kInd - 1) * testInt
    print('Saved vs uniform: ' + str(unifSampSaved))
    '''21-MAR: 21 saved'''
    # Rudimentary
    minListLen = np.min([len(i) for i in origUtilList])
    origUtilArr = np.array([i[:minListLen] for i in origUtilList])
    origAvgArr = np.average(origUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(origAvgArr.tolist()) if val > compAvg)
    origSampSaved = round(
        (compAvg - origAvgArr[kInd - 1]) / (origAvgArr[kInd] - origAvgArr[kInd - 1]) * testInt * 3) + (
                            kInd - 1) * testInt * 3
    print('Saved vs rudimentary: ' + str(origSampSaved))
    '''21-MAR: 138 saved'''

    # Use different loss parameters
    ###########
    # todo: underWt = 10.
    ###########
    paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=10., riskthreshold=0.15, riskslope=0.6,
                                                  marketvec=np.ones(numTN + numSN), candneighnum=1000)

    numtargetdraws, numDataDraws = 5100, 5000

    # Find heuristic allocation first
    utilDict = {'method': 'weightsNodeDraw3linear'}

    numReps = 20
    utilMatList = []
    # set testMax to highest expected allocation for any one node
    testMax, testInt = 100, 10
    for rep in range(numReps):
        # Withdraw a subset of MCMC prior draws
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New loss draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        # Get new data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        # Get marginal utilities at each test node
        currMargUtilMat = GetMargUtilAtNodes(dictTemp.copy(), testMax, testInt, lossDict.copy(), utilDict.copy(),
                                             masterDraws=CSdict3['postSamples'], printUpdate=True)
        print(repr(currMargUtilMat))
        utilMatList.append(currMargUtilMat)
    '''23-MAR run
    utilMatList = [np.array([[ 0.        ,  0.00120468,  0.00656565,  0.01551929,  0.03766449,
         0.05080222,  0.06784914,  0.08344739,  0.10081684,  0.11529759,
         0.13702493],
       [ 0.        ,  0.00462389,  0.02451324,  0.04813466,  0.08802436,
         0.10411307,  0.1389753 ,  0.15649096,  0.18969085,  0.20499551,
         0.23157762],
       [ 0.        ,  0.00271516,  0.00033268, -0.00217927,  0.0085513 ,
         0.02403054,  0.02085072,  0.03616679,  0.03933638,  0.05248315,
         0.05614416],
       [ 0.        , -0.00108056, -0.00362558,  0.00121604,  0.00389859,
         0.00704794, -0.00368827, -0.00040903,  0.00631218,  0.00600609,
         0.01306459],
       [ 0.        ,  0.04756594,  0.07060646,  0.09405848,  0.1129362 ,
         0.12769747,  0.15165531,  0.15996792,  0.18152856,  0.19256876,
         0.21394517],
       [ 0.        ,  0.023096  ,  0.06355345,  0.0973168 ,  0.1157059 ,
         0.14155211,  0.1522775 ,  0.16362234,  0.18334054,  0.19281525,
         0.19723289],
       [ 0.        ,  0.0595386 ,  0.12013513,  0.16613116,  0.19424526,
         0.21451091,  0.2392563 ,  0.26164562,  0.27340608,  0.28751729,
         0.30181918],
       [ 0.        ,  0.06775282,  0.10142666,  0.13179684,  0.14941416,
         0.17097894,  0.19896088,  0.21618913,  0.24161841,  0.25204073,
         0.27661429]]), 
        np.array([[0.        , 0.03313953, 0.07354083, 0.09998127, 0.12749158,
        0.14685468, 0.17277861, 0.19103047, 0.20764294, 0.23145619,
        0.24860821],
       [0.        , 0.03228439, 0.06820891, 0.1103673 , 0.14470328,
        0.17237899, 0.20826166, 0.23377392, 0.26271626, 0.281198  ,
        0.30489594],
       [0.        , 0.02386509, 0.04576465, 0.07650619, 0.08901712,
        0.11192525, 0.13170355, 0.14183803, 0.16298038, 0.17425721,
        0.1890034 ],
       [0.        , 0.02518086, 0.04621282, 0.06150915, 0.07875085,
        0.09698066, 0.10611908, 0.11359236, 0.12638717, 0.13739806,
        0.13766392],
       [0.        , 0.14953149, 0.20230311, 0.23919621, 0.26898233,
        0.2964971 , 0.31403831, 0.33542851, 0.35716539, 0.37418597,
        0.39425657],
       [0.        , 0.07314711, 0.11344224, 0.14533419, 0.16754279,
        0.19123891, 0.20749838, 0.22433617, 0.2360672 , 0.25039213,
        0.26457748],
       [0.        , 0.10447577, 0.1555549 , 0.18822103, 0.22069445,
        0.24385141, 0.26417857, 0.28561584, 0.29813949, 0.30947317,
        0.32725679],
       [0.        , 0.08457057, 0.15300597, 0.19845206, 0.23409496,
        0.26977831, 0.29956478, 0.32352341, 0.34548088, 0.37058166,
        0.38939264]]),
        np.array([[0.        , 0.00139221, 0.01903388, 0.03458285, 0.04724059,
        0.07199024, 0.0879437 , 0.11027449, 0.12787833, 0.14666119,
        0.15992047],
       [0.        , 0.00306777, 0.02986245, 0.06506521, 0.08755597,
        0.11805792, 0.1517383 , 0.17579392, 0.18910241, 0.21729688,
        0.2388932 ],
       [0.        , 0.00064503, 0.00268792, 0.00817402, 0.02932565,
        0.02455109, 0.04337812, 0.05649437, 0.06580987, 0.07819027,
        0.08913803],
       [0.        , 0.00243112, 0.00947748, 0.00427011, 0.01468392,
        0.02908101, 0.02198425, 0.03338106, 0.03183229, 0.04096667,
        0.04172231],
       [0.        , 0.03553166, 0.06038439, 0.08015674, 0.10575054,
        0.12678599, 0.14755386, 0.16286902, 0.17367467, 0.18850434,
        0.21048463],
       [0.        , 0.04460167, 0.08120985, 0.12323763, 0.14193709,
        0.16981131, 0.18892928, 0.20140503, 0.215525  , 0.23639733,
        0.25004333],
       [0.        , 0.05792356, 0.11944427, 0.1646357 , 0.19644711,
        0.22405651, 0.24634944, 0.26239254, 0.27910234, 0.30079058,
        0.31495664],
       [0.        , 0.10642332, 0.17336675, 0.21057376, 0.24135685,
        0.26147421, 0.27443612, 0.296224  , 0.31946734, 0.32496792,
        0.34868003]]),
        np.array([[0.        , 0.05863346, 0.10282637, 0.13231025, 0.17115664,
        0.19181996, 0.20651617, 0.24341473, 0.25685445, 0.27049812,
        0.29599277],
       [0.        , 0.03861315, 0.08361541, 0.13348608, 0.16865482,
        0.2053365 , 0.23085085, 0.26737741, 0.29042047, 0.31981719,
        0.33195722],
       [0.        , 0.03106147, 0.0602737 , 0.0854729 , 0.11202132,
        0.13458499, 0.15847707, 0.16710974, 0.20097755, 0.21773769,
        0.23379661],
       [0.        , 0.02206453, 0.03964952, 0.06066245, 0.0746943 ,
        0.09820347, 0.10419519, 0.1138271 , 0.12835275, 0.1302199 ,
        0.14301024],
       [0.        , 0.20082311, 0.24528342, 0.28148543, 0.30494862,
        0.32518309, 0.34737601, 0.37089341, 0.38939056, 0.40751854,
        0.42395229],
       [0.        , 0.07230572, 0.12927154, 0.16969462, 0.19740849,
        0.22234616, 0.24064166, 0.26277421, 0.28242441, 0.28645109,
        0.30427506],
       [0.        , 0.11351295, 0.1700273 , 0.21155444, 0.24011632,
        0.26639412, 0.28660026, 0.30582847, 0.3257346 , 0.33412628,
        0.35319983],
       [0.        , 0.1134575 , 0.19526986, 0.25027178, 0.29351039,
        0.32677088, 0.35447656, 0.38469364, 0.41507009, 0.43698795,
        0.46141611]]),
        np.array([[0.        , 0.03572665, 0.0783538 , 0.10109161, 0.13337982,
        0.16169645, 0.18474166, 0.19829178, 0.22416227, 0.24111931,
        0.25532316],
       [0.        , 0.04870739, 0.08728784, 0.11537056, 0.14029444,
        0.1756372 , 0.20099288, 0.22917533, 0.25256793, 0.27570327,
        0.2904192 ],
       [0.        , 0.02404566, 0.05292013, 0.07530973, 0.09607215,
        0.11403674, 0.13944292, 0.1520435 , 0.17550348, 0.18903922,
        0.20385438],
       [0.        , 0.03114307, 0.05347605, 0.07668457, 0.08875655,
        0.10454422, 0.11462093, 0.12246253, 0.13558462, 0.14042855,
        0.15133396],
       [0.        , 0.14325183, 0.19888008, 0.23689014, 0.26492813,
        0.29372858, 0.31652841, 0.33804898, 0.3539739 , 0.37585011,
        0.39250157],
       [0.        , 0.09348225, 0.14527721, 0.18056193, 0.21038689,
        0.23103906, 0.24477909, 0.26743163, 0.27926393, 0.29251721,
        0.31375712],
       [0.        , 0.12291258, 0.18337087, 0.22583915, 0.25506331,
        0.27966958, 0.30121127, 0.31623238, 0.32695004, 0.34765429,
        0.35289565],
       [0.        , 0.04835235, 0.09483808, 0.14406501, 0.17599626,
        0.21429704, 0.24475003, 0.27582559, 0.2980499 , 0.32775888,
        0.34581681]]),
        np.array([[0.        , 0.05376667, 0.10311913, 0.13651346, 0.16314932,
        0.19237238, 0.21290491, 0.23284194, 0.25214556, 0.27323666,
        0.28976281],
       [0.        , 0.05219546, 0.10003353, 0.13767871, 0.18044456,
        0.20592443, 0.23760392, 0.26583101, 0.28959246, 0.31537423,
        0.33828965],
       [0.        , 0.03070256, 0.05998629, 0.09078512, 0.11773644,
        0.14045454, 0.16264295, 0.18555271, 0.20283941, 0.21939406,
        0.24226982],
       [0.        , 0.03120702, 0.05890336, 0.07833455, 0.09960008,
        0.1170816 , 0.12939652, 0.14346861, 0.15345403, 0.16556603,
        0.1744767 ],
       [0.        , 0.21315336, 0.27711015, 0.30667501, 0.33630608,
        0.36432197, 0.37866223, 0.40270705, 0.42230374, 0.43168675,
        0.45937293],
       [0.        , 0.09843084, 0.1601419 , 0.19837331, 0.22715259,
        0.25079961, 0.27497146, 0.29124722, 0.30481952, 0.32406521,
        0.33576163],
       [0.        , 0.1563554 , 0.22129522, 0.26520678, 0.29792175,
        0.3201672 , 0.33885261, 0.35541445, 0.37170455, 0.38868968,
        0.40049498],
       [0.        , 0.12944149, 0.21141173, 0.26201055, 0.29997128,
        0.32918388, 0.36179296, 0.38856219, 0.41080412, 0.43771884,
        0.4579797 ]]),
        np.array([[0.        , 0.00770157, 0.03266161, 0.05535978, 0.08179847,
        0.10519675, 0.13237836, 0.14590382, 0.16279405, 0.1879829 ,
        0.20679289],
       [0.        , 0.00325573, 0.037566  , 0.06830022, 0.10717671,
        0.14401101, 0.17206662, 0.2078065 , 0.22677839, 0.26155415,
        0.2834657 ],
       [0.        , 0.00226681, 0.01502894, 0.03186391, 0.04404903,
        0.06308075, 0.08168572, 0.0940591 , 0.11043483, 0.12696395,
        0.13721356],
       [0.        , 0.01345735, 0.02128641, 0.03789736, 0.04677318,
        0.05617818, 0.05799564, 0.0735612 , 0.08020328, 0.09366539,
        0.09280838],
       [0.        , 0.11953525, 0.16417364, 0.18844598, 0.21679802,
        0.24260773, 0.26119381, 0.27291486, 0.29014481, 0.30682068,
        0.31905617],
       [0.        , 0.01352889, 0.04477992, 0.07320886, 0.09499985,
        0.11245572, 0.13191006, 0.15245415, 0.16330565, 0.18220056,
        0.19810975],
       [0.        , 0.01104141, 0.04023908, 0.07169041, 0.10076356,
        0.11639766, 0.13961671, 0.15799805, 0.17442989, 0.1850612 ,
        0.197637  ],
       [0.        , 0.05549393, 0.11823078, 0.15378514, 0.18863343,
        0.22079583, 0.24108493, 0.26920554, 0.29285782, 0.32059195,
        0.32983407]]),
        np.array([[0.        , 0.01702721, 0.04938746, 0.08179545, 0.10121542,
        0.12401175, 0.14677783, 0.16984229, 0.18652802, 0.19394781,
        0.22309352],
       [0.        , 0.00841891, 0.04233425, 0.0843976 , 0.1240208 ,
        0.16375507, 0.19438608, 0.22622229, 0.25094944, 0.27685202,
        0.29818069],
       [0.        , 0.00769767, 0.01448983, 0.03777543, 0.05130851,
        0.06817901, 0.08613603, 0.0960055 , 0.11301918, 0.13529931,
        0.14326072],
       [0.        , 0.01967123, 0.0365611 , 0.05391739, 0.06385921,
        0.07379105, 0.07928046, 0.09059175, 0.10016725, 0.10824891,
        0.10801388],
       [0.        , 0.10071116, 0.13988405, 0.16905553, 0.19960629,
        0.21806099, 0.23689085, 0.25252148, 0.26518859, 0.28988831,
        0.30241526],
       [0.        , 0.02250508, 0.06545604, 0.09818856, 0.11914778,
        0.1428621 , 0.15462625, 0.18190699, 0.19422659, 0.21309155,
        0.22332715],
       [0.        , 0.01411235, 0.04027322, 0.06563583, 0.09107252,
        0.114074  , 0.13905073, 0.15421677, 0.16540505, 0.18995204,
        0.19921775],
       [0.        , 0.06623112, 0.12124336, 0.16145975, 0.18971376,
        0.21734923, 0.24585194, 0.27052218, 0.29654858, 0.30749151,
        0.32954324]]),
        np.array([[0.        , 0.06036062, 0.11635292, 0.1479608 , 0.18132908,
        0.20743647, 0.23578363, 0.25410035, 0.28046125, 0.29630725,
        0.30959561],
       [0.        , 0.05793396, 0.10331497, 0.14334054, 0.18061849,
        0.20835498, 0.24207603, 0.26979677, 0.2906646 , 0.31722512,
        0.33498508],
       [0.        , 0.03395845, 0.07243307, 0.10043778, 0.12558667,
        0.15037627, 0.17289767, 0.19502859, 0.21160592, 0.2319559 ,
        0.24625542],
       [0.        , 0.0326921 , 0.05989376, 0.08156355, 0.09771958,
        0.11247129, 0.12305181, 0.13700933, 0.15860186, 0.16515768,
        0.1745921 ],
       [0.        , 0.20546795, 0.26334283, 0.29633629, 0.32661193,
        0.35664604, 0.37243026, 0.39504945, 0.41346979, 0.43134031,
        0.44591939],
       [0.        , 0.10509544, 0.16694691, 0.20706325, 0.2342641 ,
        0.2623219 , 0.28241504, 0.30390156, 0.31315435, 0.3330749 ,
        0.34975154],
       [0.        , 0.18060561, 0.23777638, 0.27199263, 0.30316308,
        0.32167852, 0.34362373, 0.36223702, 0.37149814, 0.38291811,
        0.39868928],
       [0.        , 0.11869708, 0.20055018, 0.25525819, 0.29356718,
        0.32513762, 0.35632719, 0.38198868, 0.41344002, 0.43201841,
        0.4583221 ]]),
        np.array([[0.        , 0.04474766, 0.08464523, 0.1141524 , 0.14101735,
        0.16208469, 0.18653068, 0.20437233, 0.22153497, 0.23369672,
        0.25398875],
       [0.        , 0.03667804, 0.07240557, 0.10199467, 0.13108398,
        0.1627223 , 0.1888458 , 0.20844786, 0.23555631, 0.25795483,
        0.28153754],
       [0.        , 0.02602991, 0.05442282, 0.0789273 , 0.09861853,
        0.1218746 , 0.13513853, 0.15494553, 0.17725544, 0.1832802 ,
        0.20406198],
       [0.        , 0.02986242, 0.0543273 , 0.07267712, 0.08991887,
        0.1077383 , 0.12155521, 0.12901786, 0.14065439, 0.14721779,
        0.16052934],
       [0.        , 0.14281471, 0.20015847, 0.23409114, 0.25969498,
        0.27889744, 0.29748905, 0.31685641, 0.3312501 , 0.3426454 ,
        0.35264588],
       [0.        , 0.0799959 , 0.10627601, 0.12671243, 0.14747415,
        0.16233389, 0.17979646, 0.19055269, 0.20954689, 0.22048412,
        0.23977668],
       [0.        , 0.09391213, 0.13044488, 0.16183761, 0.18798826,
        0.20561127, 0.22863851, 0.24888509, 0.26244796, 0.27300932,
        0.28679281],
       [0.        , 0.02532229, 0.07634241, 0.11123244, 0.15091152,
        0.18034649, 0.20942599, 0.2348326 , 0.26331566, 0.27969367,
        0.29725997]]),
        np.array([[0.        , 0.03142422, 0.06486296, 0.10291723, 0.14116081,
        0.17551774, 0.19783036, 0.22628256, 0.25495802, 0.2881009 ,
        0.30301989],
       [0.        , 0.10927979, 0.18962621, 0.24590258, 0.29303523,
        0.33180464, 0.36152419, 0.39464595, 0.41594735, 0.44003285,
        0.46492983],
       [0.        , 0.00924678, 0.02821577, 0.04859639, 0.06901854,
        0.08681848, 0.11373008, 0.13446964, 0.15612516, 0.17157582,
        0.19257803],
       [0.        , 0.00886661, 0.02030239, 0.03162162, 0.04285219,
        0.05020692, 0.05696601, 0.07522763, 0.07886471, 0.08940046,
        0.09545104],
       [0.        , 0.09815141, 0.15765332, 0.19864342, 0.23781899,
        0.26720321, 0.29285698, 0.31730411, 0.3463629 , 0.36381531,
        0.38079209],
       [0.        , 0.18830949, 0.24358363, 0.27281776, 0.30067747,
        0.3212423 , 0.33825845, 0.36173765, 0.37199221, 0.39396921,
        0.40520341],
       [0.        , 0.20600525, 0.28659284, 0.33854391, 0.37063065,
        0.3928698 , 0.42325399, 0.44112645, 0.45964889, 0.47606716,
        0.4931827 ],
       [0.        , 0.20242929, 0.2925174 , 0.34752582, 0.38163221,
        0.41656863, 0.44895999, 0.47058623, 0.50056362, 0.52799451,
        0.54814226]]),
        np.array([[0.        , 0.05225314, 0.09806765, 0.14052986, 0.17193756,
        0.19386782, 0.22196125, 0.24386311, 0.2669571 , 0.28066832,
        0.30551034],
       [0.        , 0.04645987, 0.09836442, 0.14611256, 0.18564199,
        0.21062697, 0.24563567, 0.27547419, 0.29782256, 0.32201786,
        0.34395411],
       [0.        , 0.02883084, 0.06059895, 0.08979253, 0.11338253,
        0.14242253, 0.15988901, 0.17911276, 0.20112847, 0.21869236,
        0.23528664],
       [0.        , 0.02606623, 0.04843602, 0.0730944 , 0.09010673,
        0.10942394, 0.11823592, 0.13678685, 0.14405748, 0.15537261,
        0.16559676],
       [0.        , 0.20422929, 0.26106099, 0.29599749, 0.32748012,
        0.35012176, 0.37031403, 0.39423722, 0.4121728 , 0.42588163,
        0.44662261],
       [0.        , 0.09211562, 0.15111817, 0.19511498, 0.2225658 ,
        0.25117175, 0.27547912, 0.28934456, 0.30783317, 0.32644268,
        0.33628961],
       [0.        , 0.15211478, 0.20477672, 0.23896691, 0.26766846,
        0.28646477, 0.30744308, 0.32069842, 0.34057127, 0.35061127,
        0.36186334],
       [0.        , 0.10966179, 0.19277521, 0.23636164, 0.27959567,
        0.31462503, 0.34072509, 0.36584727, 0.39530648, 0.41712954,
        0.45106185]]),
        np.array([[0.        , 0.01796104, 0.06397207, 0.10132702, 0.13841642,
        0.17088235, 0.20677617, 0.23844284, 0.26176382, 0.28000414,
        0.31120991],
       [0.        , 0.06419926, 0.14741114, 0.2102626 , 0.26419057,
        0.31306956, 0.35076908, 0.38470052, 0.41447988, 0.44845973,
        0.48076405],
       [0.        , 0.01822019, 0.04338464, 0.0686211 , 0.10309939,
        0.12061597, 0.15394979, 0.17644259, 0.20117321, 0.22239825,
        0.24184112],
       [0.        , 0.00220347, 0.01453781, 0.0269536 , 0.04528584,
        0.05209016, 0.0726768 , 0.08319845, 0.09331105, 0.10640713,
        0.12082745],
       [0.        , 0.15968181, 0.23020164, 0.27587376, 0.31843527,
        0.35226238, 0.38050398, 0.40735882, 0.43691675, 0.45677722,
        0.47543152],
       [0.        , 0.20787008, 0.27298804, 0.31758101, 0.3529215 ,
        0.36876846, 0.40042284, 0.41613465, 0.44085519, 0.45382621,
        0.46134013],
       [0.        , 0.18050974, 0.27147245, 0.33286789, 0.36855979,
        0.39907918, 0.43410957, 0.45591919, 0.47919596, 0.4914568 ,
        0.50693039],
       [0.        , 0.18728642, 0.27056807, 0.33451912, 0.38059231,
        0.41981962, 0.45167019, 0.48205173, 0.51294312, 0.53833324,
        0.57111647]]),
        np.array([[0.        , 0.04970474, 0.0869798 , 0.11903654, 0.15534592,
        0.17727829, 0.20015053, 0.20688221, 0.23866455, 0.25966229,
        0.27401187],
       [0.        , 0.05254539, 0.09061468, 0.13912354, 0.17743409,
        0.20359755, 0.24222139, 0.26485779, 0.29085169, 0.31732387,
        0.33524825],
       [0.        , 0.03628528, 0.06231944, 0.08789135, 0.11189568,
        0.13338403, 0.15071193, 0.17301166, 0.17995922, 0.20320912,
        0.21754693],
       [0.        , 0.03919001, 0.06256591, 0.08276759, 0.10165094,
        0.11452073, 0.13107034, 0.14103089, 0.14916935, 0.16077829,
        0.1685273 ],
       [0.        , 0.14803767, 0.2062031 , 0.23869462, 0.26899618,
        0.29475712, 0.3106018 , 0.33655331, 0.35287762, 0.37863731,
        0.39232126],
       [0.        , 0.10528444, 0.15721056, 0.19298051, 0.21850618,
        0.23987739, 0.25577525, 0.27292053, 0.29102734, 0.30250714,
        0.31543337],
       [0.        , 0.11906548, 0.16758757, 0.20492093, 0.23026454,
        0.25296962, 0.27082315, 0.28726328, 0.29818494, 0.32045305,
        0.32732069],
       [0.        , 0.10030255, 0.15341737, 0.20138957, 0.23870613,
        0.27370047, 0.29852733, 0.32546612, 0.34780858, 0.37515709,
        0.39102642]]),
        np.array([[0.        , 0.06366433, 0.11206631, 0.14632504, 0.17484113,
        0.20502613, 0.21954086, 0.24744285, 0.26779082, 0.28969772,
        0.30309555],
       [0.        , 0.04786356, 0.0903533 , 0.13942871, 0.16877514,
        0.1964454 , 0.23222649, 0.25908046, 0.2873684 , 0.31231206,
        0.33513576],
       [0.        , 0.0314505 , 0.06373121, 0.09232496, 0.11452551,
        0.14197761, 0.15949579, 0.1831393 , 0.20023212, 0.21817254,
        0.24418183],
       [0.        , 0.02603139, 0.05113609, 0.07233182, 0.09061167,
        0.11202977, 0.12313762, 0.13352904, 0.15400248, 0.15515025,
        0.16595102],
       [0.        , 0.20953398, 0.26870118, 0.29826069, 0.3306852 ,
        0.35712885, 0.37724545, 0.402646  , 0.41550509, 0.43506018,
        0.45001129],
       [0.        , 0.10431084, 0.16015775, 0.19752472, 0.23512803,
        0.25719642, 0.27606703, 0.29817176, 0.3136553 , 0.33135927,
        0.34485313],
       [0.        , 0.16923478, 0.22146758, 0.249     , 0.27420567,
        0.29425078, 0.30975559, 0.32550656, 0.33499784, 0.3478628 ,
        0.36070055],
       [0.        , 0.12688413, 0.20915701, 0.25798898, 0.29847985,
        0.33094302, 0.35741263, 0.38219601, 0.41049851, 0.43263791,
        0.45549266]]),
        np.array([[0.        , 0.03969412, 0.07850853, 0.11734785, 0.14257227,
        0.16499996, 0.20206761, 0.21423998, 0.23886763, 0.25650193,
        0.27674999],
       [0.        , 0.02060213, 0.05570588, 0.07962893, 0.11890793,
        0.15961941, 0.18580423, 0.21334974, 0.23916856, 0.26794735,
        0.28902973],
       [0.        , 0.01777297, 0.04197021, 0.06990373, 0.08761999,
        0.10914268, 0.13090144, 0.14592044, 0.16841107, 0.19327792,
        0.20340821],
       [0.        , 0.01982511, 0.03253966, 0.05443813, 0.06551632,
        0.08189303, 0.09721449, 0.10348077, 0.11643701, 0.12630137,
        0.13665804],
       [0.        , 0.17147022, 0.21459096, 0.23710744, 0.26236403,
        0.28347436, 0.30195278, 0.31782746, 0.33966071, 0.35329636,
        0.37381231],
       [0.        , 0.04386036, 0.09173428, 0.1238076 , 0.15566483,
        0.17560848, 0.19706329, 0.21064607, 0.22823917, 0.24689332,
        0.25502545],
       [0.        , 0.07661683, 0.1200709 , 0.15333289, 0.18388867,
        0.21166108, 0.23389878, 0.25589256, 0.26766092, 0.28611029,
        0.30536344],
       [0.        , 0.10066426, 0.16999533, 0.21342289, 0.25673386,
        0.2836308 , 0.3155915 , 0.34078354, 0.36179174, 0.3878621 ,
        0.40859982]]),
        np.array([[0.        , 0.06234491, 0.11449632, 0.146277  , 0.17829317,
        0.20469275, 0.2325462 , 0.25823104, 0.28234758, 0.29999243,
        0.31710299],
       [0.        , 0.05751459, 0.1122032 , 0.15795412, 0.19721723,
        0.22833867, 0.27057609, 0.29044238, 0.32456734, 0.35715248,
        0.37464227],
       [0.        , 0.03206962, 0.06008573, 0.09304891, 0.11759267,
        0.14233848, 0.16324586, 0.18113216, 0.20390412, 0.22539795,
        0.24062229],
       [0.        , 0.02386904, 0.05094115, 0.07225236, 0.08933108,
        0.10419281, 0.11645811, 0.13068253, 0.14863483, 0.15394327,
        0.17221126],
       [0.        , 0.21216329, 0.26615161, 0.29978541, 0.31906808,
        0.34339112, 0.36239126, 0.37545996, 0.39429183, 0.40026189,
        0.42056349],
       [0.        , 0.08965986, 0.14832649, 0.1812247 , 0.21214827,
        0.23371397, 0.25400709, 0.27379873, 0.29288663, 0.30336285,
        0.31976169],
       [0.        , 0.18246467, 0.25711481, 0.30131489, 0.33944508,
        0.36674534, 0.38609295, 0.40656836, 0.42475179, 0.44532821,
        0.46201847],
       [0.        , 0.09748085, 0.16393618, 0.21559485, 0.25267408,
        0.28590924, 0.31653579, 0.34203142, 0.37085739, 0.38990643,
        0.41423241]]),
        np.array([[0.        , 0.03722358, 0.08016157, 0.10669441, 0.12329107,
        0.14820211, 0.1701387 , 0.18848844, 0.20514525, 0.2233011 ,
        0.23873469],
       [0.        , 0.0324516 , 0.06809985, 0.10084225, 0.12782489,
        0.15036666, 0.18351914, 0.21333176, 0.23700245, 0.248856  ,
        0.28243943],
       [0.        , 0.02897078, 0.05043348, 0.07280051, 0.0938498 ,
        0.11291669, 0.13508687, 0.14560515, 0.16375334, 0.17764456,
        0.18723141],
       [0.        , 0.03155854, 0.0555724 , 0.07443679, 0.09520769,
        0.10885769, 0.12169741, 0.12722876, 0.13489391, 0.14705059,
        0.16073618],
       [0.        , 0.12230639, 0.16998785, 0.20493938, 0.23541831,
        0.25914561, 0.27828442, 0.30240094, 0.31544567, 0.3351258 ,
        0.35742368],
       [0.        , 0.09720627, 0.14593697, 0.18498678, 0.20415049,
        0.23164156, 0.25076975, 0.27066268, 0.28080638, 0.29473547,
        0.30690972],
       [0.        , 0.10901806, 0.15642511, 0.1834394 , 0.20728674,
        0.23008507, 0.24651168, 0.26374135, 0.27391455, 0.28609115,
        0.30096204],
       [0.        , 0.04078417, 0.08972171, 0.12595336, 0.16229071,
        0.19577688, 0.22455539, 0.2501407 , 0.27334951, 0.29736027,
        0.32611513]]),
        np.array([[0.        , 0.04081489, 0.07576069, 0.10459723, 0.1308139 ,
        0.15074586, 0.16956213, 0.19256262, 0.20979363, 0.22444586,
        0.24635572],
       [0.        , 0.0396361 , 0.07199266, 0.11168131, 0.14279642,
        0.17591949, 0.20200489, 0.22948375, 0.25600569, 0.27299716,
        0.30167613],
       [0.        , 0.02125798, 0.04320688, 0.06936076, 0.08945312,
        0.09982913, 0.12111811, 0.13416998, 0.1514787 , 0.16807497,
        0.18677758],
       [0.        , 0.0255833 , 0.04872772, 0.07166794, 0.08287347,
        0.09552021, 0.10776251, 0.12107417, 0.12786814, 0.14058234,
        0.14562226],
       [0.        , 0.13074301, 0.17597215, 0.21176078, 0.2398712 ,
        0.26603139, 0.28591951, 0.3119685 , 0.32899453, 0.34817585,
        0.36143559],
       [0.        , 0.07636434, 0.11285675, 0.14434326, 0.17101376,
        0.18893135, 0.20558204, 0.22561371, 0.23753547, 0.25821193,
        0.26688866],
       [0.        , 0.08855997, 0.1243125 , 0.15396053, 0.17717728,
        0.2018975 , 0.21749864, 0.23338086, 0.25306125, 0.27138678,
        0.27586686],
       [0.        , 0.07360704, 0.13623557, 0.18876764, 0.22123662,
        0.25377408, 0.28230132, 0.29866586, 0.32721525, 0.34958752,
        0.37404261]]),
        np.array([[0.        , 0.06642698, 0.1126966 , 0.15189995, 0.18188019,
        0.21203466, 0.23761848, 0.25672063, 0.27918256, 0.29515504,
        0.31114448],
       [0.        , 0.06301148, 0.10954638, 0.16273775, 0.19582382,
        0.22960034, 0.25977934, 0.28648943, 0.31384086, 0.34309337,
        0.36308334],
       [0.        , 0.03559442, 0.06965872, 0.09877728, 0.12561333,
        0.14635692, 0.17481931, 0.18961401, 0.21067214, 0.23117544,
        0.24754315],
       [0.        , 0.02895487, 0.05318866, 0.07921169, 0.09539065,
        0.11012728, 0.12357473, 0.13549778, 0.14768417, 0.16403758,
        0.17113878],
       [0.        , 0.20895577, 0.26179171, 0.29895428, 0.33325138,
        0.34629755, 0.36701564, 0.38383066, 0.41157286, 0.41913862,
        0.4372556 ],
       [0.        , 0.10746963, 0.16963434, 0.21054548, 0.24163762,
        0.2643124 , 0.28531539, 0.30033478, 0.32029992, 0.33894431,
        0.35136188],
       [0.        , 0.18790219, 0.25973382, 0.30817588, 0.33697458,
        0.36686214, 0.38801777, 0.41178419, 0.42518839, 0.44299389,
        0.45412617],
       [0.        , 0.08751997, 0.14858344, 0.19683696, 0.23883232,
        0.27307756, 0.29903127, 0.33181244, 0.36338814, 0.38202137,
        0.41104407]]),
        np.array([[0.        , 0.03593707, 0.06784239, 0.10877654, 0.13733426,
        0.15264816, 0.18556202, 0.20704747, 0.22541221, 0.24712992,
        0.26274818],
       [0.        , 0.04396279, 0.0885857 , 0.12504217, 0.15698466,
        0.18784943, 0.21860371, 0.2512041 , 0.27445771, 0.30345694,
        0.3219348 ],
       [0.        , 0.02655044, 0.05445105, 0.07715263, 0.10236562,
        0.11928474, 0.13573776, 0.15164273, 0.17740992, 0.19577253,
        0.21049252],
       [0.        , 0.02402553, 0.04882337, 0.06577226, 0.08569004,
        0.10013   , 0.11010692, 0.12403918, 0.13537077, 0.14188628,
        0.14991055],
       [0.        , 0.11517067, 0.17864818, 0.2172413 , 0.24839795,
        0.27822568, 0.30375331, 0.32545305, 0.34256423, 0.36780136,
        0.38010519],
       [0.        , 0.087405  , 0.13797413, 0.17645281, 0.20134546,
        0.22172647, 0.24062281, 0.26242454, 0.27552671, 0.29145343,
        0.30197073],
       [0.        , 0.09874134, 0.14693406, 0.18251505, 0.21063734,
        0.23376327, 0.24780134, 0.27242502, 0.28247593, 0.29713768,
        0.30894214],
       [0.        , 0.05289241, 0.10426139, 0.14653627, 0.18209661,
        0.21584123, 0.24857923, 0.26873011, 0.30118947, 0.3274846 ,
        0.34730771]]),
        np.array([[0.        , 0.06278092, 0.11502388, 0.15198392, 0.18504768,
        0.21297304, 0.23598708, 0.26236858, 0.28371897, 0.3052451 ,
        0.31087664],
       [0.        , 0.05504284, 0.11435937, 0.1534419 , 0.20312679,
        0.23787232, 0.27716745, 0.30216277, 0.33699471, 0.36137493,
        0.38255609],
       [0.        , 0.02824993, 0.06594923, 0.08971839, 0.11630505,
        0.13758822, 0.16367489, 0.18189233, 0.20272819, 0.22310211,
        0.24385654],
       [0.        , 0.03128787, 0.05686137, 0.08253716, 0.09719751,
        0.10988437, 0.1279442 , 0.13965978, 0.1486547 , 0.1662987 ,
        0.17453314],
       [0.        , 0.20656727, 0.26162965, 0.30277574, 0.33357291,
        0.36209645, 0.3786793 , 0.40211512, 0.42109132, 0.43991876,
        0.46294366],
       [0.        , 0.11139267, 0.16750946, 0.2144694 , 0.24168504,
        0.27051676, 0.28415466, 0.3073004 , 0.32477562, 0.33799778,
        0.34795562],
       [0.        , 0.15940642, 0.21659247, 0.25053834, 0.28651985,
        0.30641282, 0.32976667, 0.34864769, 0.36378047, 0.37463433,
        0.38595812],
       [0.        , 0.13366788, 0.22086329, 0.27558284, 0.32130776,
        0.36222546, 0.38942154, 0.42032159, 0.45172189, 0.46872536,
        0.49551464]]),
        np.array([[0.        , 0.02924261, 0.05504948, 0.07613129, 0.09856633,
        0.1262956 , 0.13765071, 0.15764234, 0.17305251, 0.19640761,
        0.21367353],
       [0.        , 0.03760503, 0.07536024, 0.1135612 , 0.15238676,
        0.18642613, 0.2187836 , 0.2560403 , 0.28920734, 0.30660623,
        0.34731379],
       [0.        , 0.01675491, 0.04055408, 0.05367299, 0.07391602,
        0.09375279, 0.10896844, 0.12514837, 0.13467749, 0.15323417,
        0.16771466],
       [0.        , 0.0224432 , 0.04011974, 0.05014071, 0.06061145,
        0.07777322, 0.08829194, 0.09534601, 0.11216376, 0.10878037,
        0.11993066],
       [0.        , 0.11655578, 0.16602204, 0.20605964, 0.22921952,
        0.251824  , 0.2770459 , 0.28749715, 0.3067288 , 0.32005917,
        0.34355373],
       [0.        , 0.08674282, 0.1422218 , 0.17417672, 0.19748817,
        0.21839482, 0.2327423 , 0.25713377, 0.26645243, 0.28391436,
        0.29793536],
       [0.        , 0.09653594, 0.1559815 , 0.20084107, 0.23159873,
        0.25779818, 0.28131297, 0.29928475, 0.31879532, 0.33090285,
        0.34763147],
       [0.        , 0.04749366, 0.09394181, 0.13968869, 0.18092432,
        0.21097466, 0.23884821, 0.27155351, 0.29243863, 0.32318136,
        0.34651883]])]
    '''
    # Get average utility matrix
    avgUtilMat = np.average(np.array(utilMatList), axis=0)
    '''23-MAR
    avgUtilMat = np.array([[0.        , 0.03926838, 0.07791196, 0.10839613, 0.13673667,
        0.16127957, 0.18485203, 0.20581453, 0.22645536, 0.24506592,
        0.26323204],
       [0.        , 0.04156318, 0.08527675, 0.12581979, 0.16246621,
        0.19442731, 0.2267136 , 0.25486866, 0.28068494, 0.30563487,
        0.32856128],
       [0.        , 0.02235837, 0.04621302, 0.06933629, 0.09090974,
        0.110414  , 0.13059489, 0.14698022, 0.16571372, 0.18305777,
        0.19843822],
       [0.        , 0.02245801, 0.04173541, 0.05938949, 0.07395568,
        0.08825078, 0.09781077, 0.10883846, 0.11968096, 0.12829845,
        0.13670912],
       [0.        , 0.1505197 , 0.20177135, 0.23532543, 0.26439749,
        0.28879939, 0.30914706, 0.32921345, 0.34792501, 0.36456342,
        0.38247052],
       [0.        , 0.08800784, 0.13815685, 0.17416162, 0.20047618,
        0.22303752, 0.24148283, 0.2602546 , 0.27537216, 0.29109162,
        0.30380615],
       [0.        , 0.11915504, 0.1742445 , 0.21265924, 0.24227535,
        0.26553351, 0.28711584, 0.30576978, 0.32048025, 0.3356621 ,
        0.34886201],
       [0.        , 0.09462682, 0.16050694, 0.20691627, 0.24401184,
        0.27621648, 0.30429699, 0.33007624, 0.35677066, 0.37857534,
        0.40152495]])
    '''
    # Find allocation for sample budget
    allocArr = forwardAllocateWithBudget(avgUtilMat, int(sampBudget / testInt))
    designArr = allocArr / np.sum(allocArr, axis=0)
    # Get utility for this allocation at the sample budget
    utilDict.update({'method': 'weightsNodeDraw4linear'})
    compUtilList, unifUtilList, origUtilList = [], [], []
    numReps = 20
    for rep in range(numReps):
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New Bayes draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        print('Generating loss matrix...')
        tempLossMat = lf.lossMatSetBayesDraws(dictTemp['postSamples'], lossDict.copy(), lossDict['bayesDraws'])
        tempLossDict = lossDict.copy()
        tempLossDict.update({'lossMat': tempLossMat})
        newBayesDraws, newLossMat = lf.add_cand_neighbors(tempLossDict.copy(), CSdict3['postSamples'],
                                                          dictTemp['postSamples'])
        tempLossDict.update({'bayesDraws': newBayesDraws, 'lossMat': newLossMat})
        baseLoss = (np.sum(newLossMat, axis=1) / newLossMat.shape[1]).min()
        # Get a new set of data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        currCompUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[designArr],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Heuristic utility: ' + str(currCompUtil))
        compUtilList.append(currCompUtil)
        '''27-MAR
        compUtilList = [0.7499004835234393, 1.1377981131790191, 0.8262071433074025, 0.683585995511911, 0.6553806788333283, 0.7281500177080122, 0.6834204707699953, 0.6901160343734754, 0.8674677411826321, 1.0157124876978179, 0.7596451174404839, 1.0136606236456265]
        '''
        # Find the equivalent uniform allocation
        currUnifUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[unifDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Uniform utility: ' + str(currUnifUtil))
        unifUtilList.append([currUnifUtil])
        unifAdd, contUnif, unifCount = 0, False, 0
        if currUnifUtil < currCompUtil:
            contUnif = True
        while contUnif:
            unifAdd += testInt
            print('Adding ' + str(unifAdd) + ' for uniform')
            currUnifUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[unifDes], numtests=sampBudget + unifAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currUnifUtil))
            unifUtilList[rep].append(currUnifUtil)
            if currUnifUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if unifCount < 3:
                    unifCount += 1
                else:
                    contUnif = False
        '''27-MAR
        unifUtilList = [[0.6740215108243994, 0.7292456484104708, 0.7895865886118649, 0.8389448706615354, 0.8785558645678151, 0.9524421892738939], [0.9964878840307048, 1.0691389598964545, 1.1352538320326735, 1.2133758447060723, 1.2632240987090162, 1.3100172899174618, 1.3676023216158182], [0.7512720935776649, 0.8109584976958679, 0.8686730094062938, 0.9111792812613349, 0.9620608282242209, 1.016242187017057], [0.6061826940655397, 0.661885178029503, 0.7232884538733133, 0.7790847343217306, 0.8378033091035615, 0.8969209373701821], [0.570780363863518, 0.6290624041996749, 0.6949740394846531, 0.7485966998757245, 0.8317765842964935, 0.8664356153552006], [0.6432076409294964, 0.7024082559376401, 0.7691966112612105, 0.8267399549376488, 0.874152336425337, 0.9401725975674324], [0.6248717195664364, 0.6826811014517258, 0.7278434605201944, 0.783345003709699, 0.8339254191713756, 0.9064610678878271], [0.5815971285040957, 0.6351454467435085, 0.701498304930702, 0.7553674730704865, 0.8088683675990334, 0.8797830393587818], [0.7756950445481046, 0.840684650013424, 0.888161665909224, 0.9674234253123544, 0.9892640285367094, 1.0488782509186416], [0.9187424379712903, 0.9823611176748992, 1.0417525777921695, 1.1010800571578745, 1.1595473391758029, 1.2148865769395778], [0.6600953860272645, 0.7255315340766577, 0.7961431760934623, 0.8398175049429311, 0.8873151645791655, 0.9511149024808789], [0.934920329725097, 0.988626109314021, 1.067607835040267, 1.1115091986328896, 1.1906774683497652, 1.2427548707871816]]
        '''
        # Find the equivalent rudimentary allocation
        currOrigUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[origDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Rudimentary utility: ' + str(currOrigUtil))
        origUtilList.append([currOrigUtil])
        origAdd, contOrig, origCount = 0, False, 0
        if currOrigUtil < currCompUtil:
            contOrig = True
        while contOrig:
            origAdd += testInt * 3
            print('Adding ' + str(origAdd) + ' for rudimentary')
            currOrigUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[origDes], numtests=sampBudget + origAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currOrigUtil))
            origUtilList[rep].append(currOrigUtil)
            if currOrigUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if origCount < 3:
                    origCount += 1
                else:
                    contOrig = False
        '''27-MAR
        origUtilList = [[0.24261460511487165, 0.30626941529878593, 0.38034349526775824, 0.4518183744400499, 0.5382253961138392, 0.6034104512622989, 0.6881429785285063, 0.7800116575551792, 0.8490476953886059, 0.9389854286611481, 1.0273122585789816], [0.3182757531306031, 0.410677846653714, 0.4941978632692532, 0.6007054322288861, 0.6852898443401738, 0.7922123182742737, 0.8829150480030643, 0.9868195809412805, 1.07000470130264, 1.1642695456457703, 1.274766504696462, 1.3248145805088214, 1.415869472633628], [0.2992250928613256, 0.3759945187984055, 0.44722508926833004, 0.5243066198334922, 0.5888927293279362, 0.678107239504194, 0.7492828077310696, 0.853081693469143, 0.9321156875468981, 0.9866041488167729, 1.0941590723463968], [0.1715929466324413, 0.24122400582668657, 0.31459162592513934, 0.4080860073777517, 0.4606798470308764, 0.5423393682108673, 0.6310373397030018, 0.7118479638446775, 0.7878729104812372, 0.8765366426094401, 0.9831276665811535], [0.11809274528595903, 0.16876559773257505, 0.22478636552586018, 0.3087405022970131, 0.38322420898696663, 0.4743148756747608, 0.5533031055334074, 0.6422233619735316, 0.7393930945235843, 0.8121904979787526, 0.9426084183903276, 0.9894568720922168], [0.20821019830382603, 0.27621505669369206, 0.3477353784426098, 0.4385473476379831, 0.5195783336095907, 0.5943995257125056, 0.6892357169005949, 0.7644855495949585, 0.8274929552463384, 0.9283299770747719, 1.0333708749881598], [0.24579151603303995, 0.31449536791624944, 0.36801205328678055, 0.44488920944245436, 0.509636976493665, 0.6044022964866382, 0.6682676115010828, 0.745433132090108, 0.8373351109471612, 0.9065339981990403, 0.9953602769342584], [0.10248779785779938, 0.15980775722058116, 0.22926174038591363, 0.31014590330079006, 0.38497330578644195, 0.463962223927731, 0.5506450662619065, 0.6394031188997564, 0.7061996243841691, 0.7991210877751254, 0.8964707544390755, 0.943614542126098], [0.2756816942141578, 0.35545034641279116, 0.4292813763395502, 0.5056412547008247, 0.581485080347985, 0.6676767844964138, 0.7491074530442532, 0.8303203747782479, 0.9200248413208145, 0.9931880021529196, 1.1082620680739064, 1.143529267316696], [0.30054754872366374, 0.39132421222649416, 0.4806170329273689, 0.5667330186952082, 0.6804137293594943, 0.7617130458825621, 0.8520870550864377, 0.9493191902246716, 1.0416613928091163, 1.1166123828213488, 1.2324479965440505, 1.2786433438719937], [0.2256169931438201, 0.2894891587435904, 0.3688809327911269, 0.4426495907046126, 0.5066487417077834, 0.5995656211642508, 0.6711571634723015, 0.7642257228031193, 0.8303715607251609, 0.9042049516660899, 1.0175789708204404], [0.3351478675563655, 0.4316769693442142, 0.5157023960284022, 0.6286265157865865, 0.7219649669685513, 0.8042400852351497, 0.8909525769406139, 0.9838665211437183, 1.1038552129959154, 1.198611041023618, 1.2938481009389182, 1.3747106189205436]]
        '''
    compAvg = np.average(compUtilList)
    # Locate closest sample point for uniform and rudimentary to compAvg
    minListLen = np.min([len(i) for i in unifUtilList])
    unifUtilArr = np.array([i[:minListLen] for i in unifUtilList])
    unifAvgArr = np.average(unifUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(unifAvgArr.tolist()) if val > compAvg)
    unifSampSaved = round((compAvg - unifAvgArr[kInd - 1]) / (unifAvgArr[kInd] - unifAvgArr[kInd - 1]) * testInt) + (
            kInd - 1) * testInt
    print('Saved vs uniform: ' + str(unifSampSaved))
    '''27-MAR: XX saved'''
    # Rudimentary
    minListLen = np.min([len(i) for i in origUtilList])
    origUtilArr = np.array([i[:minListLen] for i in origUtilList])
    origAvgArr = np.average(origUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(origAvgArr.tolist()) if val > compAvg)
    origSampSaved = round(
        (compAvg - origAvgArr[kInd - 1]) / (origAvgArr[kInd] - origAvgArr[kInd - 1]) * testInt * 3) + (
                            kInd - 1) * testInt * 3
    print('Saved vs rudimentary: ' + str(origSampSaved))
    '''27-MAR: XX saved'''

    ##############################################
    ##############################################
    # Choose different sourcing matrix (PART 1)
    ##############################################
    ##############################################
    numBoot = 44  # 44 is average across each TN in original data set
    SNprobs = np.sum(CSdict3['N'], axis=0) / np.sum(CSdict3['N'])
    np.random.seed(52)  # Chosen to be "far" from seed 33
    Qvecs = np.random.multinomial(numBoot, SNprobs, size=numTN - 4) / numBoot
    CSdict3['Q'] = np.vstack((CSdict3['N'][:4] / np.sum(CSdict3['N'][:4], axis=1).reshape(4, 1), Qvecs))

    sampBudget = 180
    unifDes = np.zeros(numTN) + 1 / numTN
    origDes = np.sum(rd3_N, axis=1) / np.sum(rd3_N)

    # Use original loss parameters
    paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                                  marketvec=np.ones(numTN + numSN), candneighnum=1000)

    numtargetdraws, numDataDraws = 5100, 5000

    # Find heuristic allocation first
    utilDict = {'method': 'weightsNodeDraw3linear'}

    numReps = 10
    utilMatList = []
    # set testMax to highest expected allocation for any one node
    testMax, testInt = 100, 10
    testArr = np.arange(0, testMax + 1, testInt)
    for rep in range(numReps):
        CSdict3 = methods.GeneratePostSamples(CSdict3)
        # Withdraw a subset of MCMC prior draws
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New loss draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        # Get new data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        # Get marginal utilities at each test node
        currMargUtilMat = GetMargUtilAtNodes(dictTemp.copy(), testMax, testInt, lossDict.copy(), utilDict.copy(),
                                             masterDraws=CSdict3['postSamples'], printUpdate=True)
        print(repr(currMargUtilMat))
        utilMatList.append(currMargUtilMat)
        for mat in utilMatList:
            for i in range(8):
                plt.plot(testArr, mat[i], linewidth=0.2)
        avgUtilMat = np.average(np.array(utilMatList), axis=0)
        for i in range(8):
            plt.plot(testArr, avgUtilMat[i], linewidth=2)
        plt.ylim([0, 0.4])
        # plt.title('Comprehensive utility for allocations via heuristic\nUntested nodes')
        plt.show()
        plt.close()

    '''29-MAR run
    utilMatList = [np.array([[0.        , 0.02275528, 0.04837098, 0.07278004, 0.09910215,
        0.11461911, 0.13422516, 0.15418469, 0.17162021, 0.18504719,
        0.19729222],
       [0.        , 0.09104728, 0.14617577, 0.19210137, 0.22875587,
        0.25756715, 0.28611237, 0.31092822, 0.33587376, 0.35227338,
        0.36983684],
       [0.        , 0.01961832, 0.04017934, 0.06033921, 0.07965188,
        0.09427007, 0.11402248, 0.13491036, 0.14998735, 0.16470116,
        0.17314291],
       [0.        , 0.01785025, 0.03487998, 0.04542012, 0.05776358,
        0.06845461, 0.07337025, 0.08330505, 0.09065133, 0.09804978,
        0.10260606],
       [0.        , 0.10736461, 0.15640483, 0.18866763, 0.21267021,
        0.23131028, 0.24760013, 0.26140689, 0.27937149, 0.28840331,
        0.29971427],
       [0.        , 0.10611122, 0.1473778 , 0.16622364, 0.18824015,
        0.19995745, 0.21483725, 0.22851366, 0.23454736, 0.24817134,
        0.25302033],
       [0.        , 0.13002212, 0.17598799, 0.20686872, 0.23121682,
        0.25342685, 0.26920238, 0.28927316, 0.30292618, 0.31737963,
        0.33421621],
       [0.        , 0.08634447, 0.1275765 , 0.15301091, 0.17491191,
        0.1927294 , 0.21056492, 0.22517469, 0.2376144 , 0.24826687,
        0.26344639]]), np.array([[0.        , 0.0178994 , 0.04146621, 0.06674922, 0.08532571,
        0.10380323, 0.12696794, 0.1406563 , 0.15294824, 0.17123199,
        0.18228932],
       [0.        , 0.05132826, 0.10332586, 0.1430882 , 0.18029472,
        0.20644752, 0.23271443, 0.25608659, 0.27465471, 0.30048873,
        0.3114702 ],
       [0.        , 0.00725251, 0.02046468, 0.03855952, 0.05113885,
        0.06612738, 0.0818477 , 0.09152027, 0.11125677, 0.12298624,
        0.13566298],
       [0.        , 0.0060726 , 0.0124866 , 0.02006972, 0.02590834,
        0.03404262, 0.04073207, 0.04657634, 0.0504507 , 0.06164989,
        0.06458713],
       [0.        , 0.0708288 , 0.11823885, 0.14760624, 0.17250405,
        0.19316336, 0.20725109, 0.22068913, 0.23950787, 0.24721513,
        0.26148293],
       [0.        , 0.08969768, 0.12617136, 0.15007007, 0.16673495,
        0.18120873, 0.19130958, 0.20169659, 0.20993747, 0.22073725,
        0.22813041],
       [0.        , 0.10610797, 0.15488213, 0.18207403, 0.21111539,
        0.23459503, 0.25059226, 0.26693062, 0.28343304, 0.30193164,
        0.31198827],
       [0.        , 0.05694171, 0.09422749, 0.11688037, 0.13711513,
        0.15463993, 0.17018557, 0.18105656, 0.19447084, 0.20530128,
        0.21676963]]), np.array([[0.        , 0.02413365, 0.04880679, 0.06665574, 0.08744179,
        0.10840102, 0.12566595, 0.14242489, 0.154351  , 0.17298308,
        0.19052693],
       [0.        , 0.05143311, 0.1014853 , 0.14369129, 0.17681737,
        0.20588838, 0.22914783, 0.25384861, 0.27601382, 0.29155944,
        0.30843694],
       [0.        , 0.01054531, 0.0243063 , 0.03838198, 0.0547016 ,
        0.06778679, 0.07943512, 0.09349243, 0.10862965, 0.12165822,
        0.13099941],
       [0.        , 0.00879734, 0.01970332, 0.02415635, 0.02998552,
        0.04201   , 0.04184719, 0.04894626, 0.05208047, 0.06059922,
        0.067912  ],
       [0.        , 0.06679476, 0.10788621, 0.13643279, 0.16136224,
        0.18044656, 0.19603321, 0.2129644 , 0.22215808, 0.2352634 ,
        0.25204086],
       [0.        , 0.09072985, 0.12870412, 0.15051426, 0.16748198,
        0.17786778, 0.19019349, 0.2030956 , 0.20993694, 0.21806302,
        0.22733923],
       [0.        , 0.11587129, 0.16088343, 0.18913968, 0.20961563,
        0.23356712, 0.24638812, 0.26555989, 0.2819769 , 0.2960387 ,
        0.3075101 ],
       [0.        , 0.04191082, 0.07146198, 0.09395969, 0.1118059 ,
        0.12883147, 0.14460774, 0.15865881, 0.17413222, 0.18689028,
        0.19801541]]), np.array([[0.        , 0.02173703, 0.04646899, 0.069928  , 0.09044307,
        0.1105231 , 0.13206405, 0.1456937 , 0.15874663, 0.17668529,
        0.19037812],
       [0.        , 0.05264318, 0.09955012, 0.14280263, 0.17572862,
        0.20582441, 0.23190954, 0.25549782, 0.27435674, 0.29900693,
        0.31050781],
       [0.        , 0.01329398, 0.0245389 , 0.04021345, 0.05500323,
        0.06959928, 0.08867972, 0.0991523 , 0.11456602, 0.1324537 ,
        0.14310674],
       [0.        , 0.00658552, 0.01451061, 0.02479891, 0.02801224,
        0.03988201, 0.04387978, 0.04788871, 0.05861033, 0.0586809 ,
        0.06154156],
       [0.        , 0.05353522, 0.09207255, 0.12023465, 0.14331218,
        0.16139082, 0.1769302 , 0.19571498, 0.20943906, 0.21969301,
        0.23050715],
       [0.        , 0.08379559, 0.11898279, 0.13827787, 0.15408999,
        0.16832553, 0.17574014, 0.18884743, 0.19980279, 0.20762076,
        0.21563925],
       [0.        , 0.09794716, 0.14110632, 0.17004506, 0.19545512,
        0.21267105, 0.23090484, 0.25081249, 0.26424793, 0.27580439,
        0.29427288],
       [0.        , 0.04702599, 0.08172637, 0.10430713, 0.1249899 ,
        0.14342032, 0.15598524, 0.1714402 , 0.18329057, 0.19923871,
        0.20668654]]), np.array([[0.        , 0.02207076, 0.03904891, 0.06029145, 0.07560859,
        0.09639835, 0.1069674 , 0.1248965 , 0.13865438, 0.15115093,
        0.16363263],
       [0.        , 0.04475288, 0.08865809, 0.13353436, 0.16856184,
        0.19567051, 0.22460726, 0.25103448, 0.27026719, 0.28845501,
        0.30786997],
       [0.        , 0.01074436, 0.02653691, 0.04247764, 0.06003736,
        0.07478737, 0.08877388, 0.10410549, 0.11863619, 0.13402146,
        0.14590885],
       [0.        , 0.00697531, 0.01347865, 0.02224503, 0.02869429,
        0.03485444, 0.03819802, 0.04282225, 0.04772751, 0.05173113,
        0.05912379],
       [0.        , 0.05882229, 0.09886534, 0.12471637, 0.1422088 ,
        0.16480358, 0.18014636, 0.19388061, 0.20603423, 0.21926714,
        0.22999757],
       [0.        , 0.08652312, 0.12181492, 0.14130329, 0.15591976,
        0.17146359, 0.18075498, 0.18826812, 0.19572762, 0.20767497,
        0.21072704],
       [0.        , 0.09677618, 0.13756467, 0.16821292, 0.19435701,
        0.21314726, 0.23403648, 0.25290906, 0.26593826, 0.28228614,
        0.2973424 ],
       [0.        , 0.05001231, 0.08399632, 0.10919781, 0.12915762,
        0.14768653, 0.16308071, 0.17422591, 0.18867859, 0.20011872,
        0.21098384]])] 
    '''
    '''13-APR run
    utilMatList = [np.array([[0.        , 0.02451592, 0.04855758, 0.07261078, 0.09142814,
        0.11231712, 0.12268141, 0.14118758, 0.15649826, 0.17258595,
        0.18135153],
       [0.        , 0.0655778 , 0.11702967, 0.15557825, 0.19120753,
        0.21937603, 0.2460934 , 0.26942906, 0.28777263, 0.30576801,
        0.32764214],
       [0.        , 0.01395371, 0.02843585, 0.04777044, 0.06274787,
        0.07774682, 0.09496539, 0.10862246, 0.12546814, 0.13929499,
        0.15431139],
       [0.        , 0.00521008, 0.01313866, 0.02074631, 0.02795878,
        0.03604236, 0.0420482 , 0.04601286, 0.05039855, 0.0567085 ,
        0.06208824],
       [0.        , 0.14667013, 0.18892597, 0.21704129, 0.23478878,
        0.24889874, 0.25914828, 0.27279244, 0.28066991, 0.28929819,
        0.29815678],
       [0.        , 0.05107067, 0.08162828, 0.10255872, 0.1165097 ,
        0.12920727, 0.14268343, 0.15585291, 0.16525668, 0.17840945,
        0.18266768],
       [0.        , 0.09346927, 0.13843889, 0.16721363, 0.19044481,
        0.21103031, 0.23259784, 0.24667569, 0.26356087, 0.27718308,
        0.29060785],
       [0.        , 0.12100444, 0.16388659, 0.19173441, 0.21137973,
        0.22787451, 0.24012738, 0.25692266, 0.26379497, 0.27543485,
        0.2852622 ]]),
        np.array([[0.        , 0.00676249, 0.02308257, 0.039151  , 0.05967397,
        0.07769754, 0.09534459, 0.1112172 , 0.12503183, 0.14360623,
        0.15475692],
       [0.        , 0.0346137 , 0.08219858, 0.12369608, 0.15473904,
        0.18892824, 0.21821453, 0.24186506, 0.26576264, 0.28120922,
        0.30322893],
       [0.        , 0.0018109 , 0.01021703, 0.02174241, 0.03680351,
        0.04552145, 0.06302509, 0.07815022, 0.08852497, 0.10427657,
        0.12305748],
       [0.        , 0.00370287, 0.00679118, 0.0111217 , 0.01767454,
        0.01913819, 0.0279442 , 0.03142798, 0.03622548, 0.04361832,
        0.04688333],
       [0.        , 0.10768323, 0.14481648, 0.16980071, 0.18989393,
        0.20543071, 0.22083033, 0.23767198, 0.25202475, 0.26020115,
        0.27477491],
       [0.        , 0.04439424, 0.07675788, 0.10029112, 0.11625655,
        0.13113607, 0.1468082 , 0.15526558, 0.16553691, 0.17438951,
        0.18389694],
       [0.        , 0.04406365, 0.08873329, 0.12490267, 0.15129765,
        0.17359946, 0.19694854, 0.21493065, 0.2326752 , 0.2545836 ,
        0.27274015],
       [0.        , 0.0383171 , 0.06461345, 0.09241397, 0.11379849,
        0.13079087, 0.14576379, 0.16214816, 0.17295371, 0.18549867,
        0.19890409]]),
        np.array([[0.        , 0.05749636, 0.09608722, 0.13029326, 0.15485669,
        0.17692768, 0.19296016, 0.21262851, 0.22886686, 0.24418249,
        0.26050214],
       [0.        , 0.1257779 , 0.19222448, 0.2395383 , 0.27752983,
        0.31018128, 0.33962961, 0.36208096, 0.38185752, 0.40429385,
        0.42102526],
       [0.        , 0.02842965, 0.05681675, 0.08032545, 0.10134849,
        0.11700292, 0.13632777, 0.15409641, 0.16597128, 0.18501697,
        0.19663819],
       [0.        , 0.02720735, 0.0449396 , 0.05798867, 0.07038999,
        0.07948648, 0.08803197, 0.09471818, 0.10174703, 0.10669567,
        0.11344435],
       [0.        , 0.1252845 , 0.17514875, 0.21021125, 0.23329018,
        0.25467111, 0.27114674, 0.2869127 , 0.3003015 , 0.31193395,
        0.32347585],
       [0.        , 0.10857354, 0.15100698, 0.17627265, 0.19548869,
        0.21145851, 0.22671338, 0.23555298, 0.2477294 , 0.25663849,
        0.26664315],
       [0.        , 0.17068713, 0.22417778, 0.26087938, 0.28091857,
        0.30008478, 0.32011375, 0.33597012, 0.34834607, 0.36435549,
        0.3779873 ],
       [0.        , 0.11734849, 0.16431268, 0.19521884, 0.22497543,
        0.24260908, 0.25851014, 0.2811719 , 0.29496294, 0.30423754,
        0.31838349]]),
        np.array([[0.        , 0.01345173, 0.03136005, 0.05660755, 0.07859248,
        0.09639092, 0.11295868, 0.13071614, 0.14827999, 0.16534654,
        0.18174914],
       [0.        , 0.07221144, 0.13878848, 0.18017535, 0.21969484,
        0.24911042, 0.27531689, 0.30023162, 0.31930139, 0.33937108,
        0.36015899],
       [0.        , 0.008002  , 0.02708274, 0.04394301, 0.06352998,
        0.08131683, 0.0980254 , 0.11339686, 0.13039917, 0.14622489,
        0.15684982],
       [0.        , 0.00349879, 0.01013631, 0.01498932, 0.02047488,
        0.02619417, 0.03191725, 0.03569684, 0.04260874, 0.04852905,
        0.05173094],
       [0.        , 0.059602  , 0.10502625, 0.13563956, 0.16142992,
        0.18021392, 0.20175985, 0.21506542, 0.23033948, 0.24530919,
        0.25356911],
       [0.        , 0.08883953, 0.12631464, 0.14627189, 0.16541075,
        0.17677951, 0.18724289, 0.19964313, 0.20627858, 0.21677837,
        0.22701359],
       [0.        , 0.13416521, 0.17427311, 0.20280975, 0.22450047,
        0.24442846, 0.25841944, 0.27636562, 0.29124343, 0.30653524,
        0.31939327],
       [0.        , 0.06871341, 0.09537234, 0.11617885, 0.13536047,
        0.1478486 , 0.16251259, 0.17380859, 0.18937296, 0.1976615 ,
        0.21190252]]),
        np.array([[ 0.00000000e+00,  4.23145488e-03,  1.51264347e-02,
         2.45054095e-02,  4.29940701e-02,  5.08498899e-02,
         6.71761867e-02,  7.46196354e-02,  9.01195534e-02,
         1.01857410e-01,  1.14405441e-01],
       [ 0.00000000e+00,  4.00727840e-02,  8.81030381e-02,
         1.22068004e-01,  1.49450701e-01,  1.77534635e-01,
         2.01722650e-01,  2.23543687e-01,  2.39366952e-01,
         2.56424347e-01,  2.71441093e-01],
       [ 0.00000000e+00,  2.50684148e-03,  7.01300043e-03,
         1.18518074e-02,  2.24697149e-02,  2.67489313e-02,
         3.91415789e-02,  4.64860059e-02,  5.67779341e-02,
         6.50942739e-02,  7.81336044e-02],
       [ 0.00000000e+00, -2.15147090e-04,  1.23774415e-04,
         2.07058389e-03,  1.30248355e-03,  4.54540076e-03,
         4.34345737e-03,  9.34790905e-03,  1.06808691e-02,
         1.65392323e-02,  1.90601751e-02],
       [ 0.00000000e+00,  4.95345578e-02,  8.04655133e-02,
         1.01774369e-01,  1.21416510e-01,  1.36780849e-01,
         1.50481639e-01,  1.64716040e-01,  1.75512424e-01,
         1.83859883e-01,  1.94769153e-01],
       [ 0.00000000e+00,  3.07521460e-02,  5.31325418e-02,
         7.57079463e-02,  9.15199867e-02,  1.02830280e-01,
         1.16579541e-01,  1.22976742e-01,  1.37384085e-01,
         1.38451776e-01,  1.50954030e-01],
       [ 0.00000000e+00,  4.46876651e-02,  7.27381704e-02,
         9.77754782e-02,  1.15578898e-01,  1.36566071e-01,
         1.53768888e-01,  1.70996834e-01,  1.84437278e-01,
         1.99940675e-01,  2.11191984e-01],
       [ 0.00000000e+00,  6.49792297e-02,  9.83533193e-02,
         1.20653771e-01,  1.36898958e-01,  1.49472180e-01,
         1.60407795e-01,  1.67185821e-01,  1.76185866e-01,
         1.85262966e-01,  1.93078103e-01]]),
         np.array([[0.        , 0.0346768 , 0.06621606, 0.09164403, 0.1145677 ,
        0.13315516, 0.15287402, 0.17231226, 0.19186275, 0.20930308,
        0.21777717],
       [0.        , 0.07794463, 0.14559928, 0.19967276, 0.24176276,
        0.27980506, 0.30200435, 0.33175581, 0.34953254, 0.37111317,
        0.38905085],
       [0.        , 0.02434758, 0.04518998, 0.06380172, 0.08224753,
        0.10356801, 0.11630203, 0.13095722, 0.15058368, 0.16073175,
        0.16907954],
       [0.        , 0.01114391, 0.02227675, 0.03172902, 0.04188907,
        0.0491231 , 0.05449954, 0.06392452, 0.06898837, 0.07386585,
        0.07757122],
       [0.        , 0.13939531, 0.17765902, 0.20074625, 0.22025803,
        0.23289398, 0.24869296, 0.25825671, 0.2746029 , 0.28783204,
        0.29470339],
       [0.        , 0.09411192, 0.13132608, 0.15403398, 0.17097161,
        0.18285429, 0.19906   , 0.205608  , 0.21745646, 0.22326376,
        0.23228131],
       [0.        , 0.1043377 , 0.15380167, 0.18570403, 0.2133691 ,
        0.23775624, 0.26010548, 0.27929454, 0.29342104, 0.31421402,
        0.32667931],
       [0.        , 0.05808167, 0.10520707, 0.14058244, 0.16649049,
        0.18313798, 0.20354156, 0.22284478, 0.23510841, 0.24863756,
        0.26258778]])
        ]
    '''
    # Get average utility matrix
    avgUtilMat = np.average(np.array(utilMatList), axis=0)
    '''29-MAR
    avgUtilMat = np.array([[0.        , 0.02171922, 0.04483238, 0.06728089, 0.08758426,
        0.10674896, 0.1251781 , 0.14157122, 0.15526409, 0.1714197 ,
        0.18482385],
       [0.        , 0.05824094, 0.10783903, 0.15104357, 0.18603169,
        0.21427959, 0.24089829, 0.26547914, 0.28623324, 0.3063567 ,
        0.32162435],
       [0.        , 0.01229089, 0.02720522, 0.04399436, 0.06010658,
        0.07451418, 0.09055178, 0.10463617, 0.1206152 , 0.13516416,
        0.14576418],
       [0.        , 0.0092562 , 0.01901183, 0.02733803, 0.03407279,
        0.04384874, 0.04760546, 0.05390772, 0.05990407, 0.06614218,
        0.07115411],
       [0.        , 0.07146914, 0.11469356, 0.14353154, 0.1664115 ,
        0.18622292, 0.2015922 , 0.2169312 , 0.23130215, 0.2419684 ,
        0.25474856],
       [0.        , 0.09137149, 0.1286102 , 0.14927782, 0.16649337,
        0.17976462, 0.19056709, 0.20208428, 0.20999044, 0.22045347,
        0.22697125],
       [0.        , 0.10934494, 0.15408491, 0.18326808, 0.208352  ,
        0.22948146, 0.24622482, 0.26509704, 0.27970446, 0.2946881 ,
        0.30906597],
       [0.        , 0.05644706, 0.09179773, 0.11547118, 0.13559609,
        0.15346153, 0.16888484, 0.18211123, 0.19563732, 0.20796317,
        0.21918036]])
    '''
    # Find allocation for sample budget
    allocArr = forwardAllocateWithBudget(avgUtilMat, int(sampBudget / testInt))
    designArr = allocArr / np.sum(allocArr, axis=0)
    # Get utility for this allocation at the sample budget
    utilDict.update({'method': 'weightsNodeDraw4linear'})
    compUtilList, unifUtilList, origUtilList = [], [], []
    numReps = 5
    for rep in range(numReps):
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New Bayes draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        print('Generating loss matrix...')
        tempLossMat = lf.lossMatSetBayesDraws(dictTemp['postSamples'], lossDict.copy(), lossDict['bayesDraws'])
        tempLossDict = lossDict.copy()
        tempLossDict.update({'lossMat': tempLossMat})
        newBayesDraws, newLossMat = lf.add_cand_neighbors(tempLossDict.copy(), CSdict3['postSamples'],
                                                          dictTemp['postSamples'])
        tempLossDict.update({'bayesDraws': newBayesDraws, 'lossMat': newLossMat})
        baseLoss = (np.sum(newLossMat, axis=1) / newLossMat.shape[1]).min()
        # Get a new set of data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        currCompUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[designArr],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Heuristic utility: ' + str(currCompUtil))
        compUtilList.append(currCompUtil)
        '''29-MAR
        compUtilList = [1.064599746043636, 0.9536545767648335, 0.9865412561096165, 0.9978552443129023, 0.9552351708480038]
        '''
        # Find the equivalent uniform allocation
        currUnifUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[unifDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Uniform utility: ' + str(currUnifUtil))
        unifUtilList.append([currUnifUtil])
        unifAdd, contUnif, unifCount = 0, False, 0
        if currUnifUtil < currCompUtil:
            contUnif = True
        while contUnif:
            unifAdd += testInt
            print('Adding ' + str(unifAdd) + ' for uniform')
            currUnifUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[unifDes], numtests=sampBudget + unifAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currUnifUtil))
            unifUtilList[rep].append(currUnifUtil)
            if currUnifUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if unifCount < 3:
                    unifCount += 1
                else:
                    contUnif = False
        '''29-MAR
        unifUtilList = [[0.9549743557762884, 0.9873513777707177, 1.0227466081244083, 1.0631428216562684, 1.0830365491993126, 1.1084450350859014, 1.14645156537902, 1.1788038702166102], [0.8691320562264298, 0.8953460671990507, 0.9222824838458035, 0.9554848855370683, 0.997959682276413, 1.0030366542207907, 1.0481160529018223], [0.8737240973859528, 0.8949967032607189, 0.9489948206653107, 0.9770327046632632, 1.003670748661889, 1.04314715823682, 1.0709183543181777, 1.1001028024838675], [0.8891646659031762, 0.9231527416342487, 0.9642780133384328, 0.9918191931069571, 1.026738254822006, 1.0565921650310401, 1.0764863382113417, 1.1187187303091464], [0.8389144336375423, 0.8815316156772162, 0.9142323786565725, 0.951583853996596, 0.9937190456377314, 1.0190881352109646, 1.0451843837957924, 1.0668866892556577]]
        '''
        # Find the equivalent rudimentary allocation
        currOrigUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[origDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Rudimentary utility: ' + str(currOrigUtil))
        origUtilList.append([currOrigUtil])
        origAdd, contOrig, origCount = 0, False, 0
        if currOrigUtil < currCompUtil:
            contOrig = True
        while contOrig:
            origAdd += testInt * 3
            print('Adding ' + str(origAdd) + ' for rudimentary')
            currOrigUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[origDes], numtests=sampBudget + origAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currOrigUtil))
            origUtilList[rep].append(currOrigUtil)
            if currOrigUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if origCount < 3:
                    origCount += 1
                else:
                    contOrig = False
        '''29-MAR
        origUtilList = [[0.42419885610644714, 0.47818372469073633, 0.5513359898512133, 0.609865801667699, 0.6839292870705043, 0.7525565927125584, 0.8019130993834653, 0.8785371366440988, 0.9083136130861007, 0.983182433383404, 1.0337317576019047, 1.093994307128984, 1.1343332966936748, 1.2000530111739929, 1.2215792406303079], [0.3806259012400326, 0.4402077288584265, 0.4926193654495701, 0.5579320361401434, 0.6122678207164238, 0.6886073288895203, 0.7490561356740293, 0.8259547966821974, 0.8559815659152994, 0.9191348628531077, 0.9777246126709391, 1.0028624415231557, 1.0701523914382198, 1.119773700287404], [0.387320858716119, 0.4452092229231761, 0.5083386044940639, 0.5610733935417529, 0.6337826621873401, 0.68428718344044, 0.7501916093155012, 0.8311093331673374, 0.857735134164475, 0.9022789874026689, 0.9674172161532781, 1.0196577489054746, 1.068152283378315, 1.1143383198857033, 1.1745902357928757], [0.39365783186895476, 0.46602091052489447, 0.5268575266700868, 0.5837348809606233, 0.6517942892215904, 0.7138114121372454, 0.7792835398581865, 0.8537970800977059, 0.8871749295717222, 0.9506038657360403, 1.008388042535961, 1.060381170199601, 1.106051149981444, 1.1682900859465666], [0.35066290000566624, 0.4140213797577559, 0.47976198710565265, 0.5508317402980891, 0.5992450756710719, 0.6769196049698181, 0.7163650094732628, 0.8152443341294133, 0.8478467616652323, 0.9072768998424752, 0.9584508619926622, 0.9938665714289852, 1.0625497808922817, 1.1053667242947078]]
        '''
    compAvg = np.average(compUtilList)
    # Locate closest sample point for uniform and rudimentary to compAvg
    minListLen = np.min([len(i) for i in unifUtilList])
    unifUtilArr = np.array([i[:minListLen] for i in unifUtilList])
    unifAvgArr = np.average(unifUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(unifAvgArr.tolist()) if val > compAvg)
    unifSampSaved = round((compAvg - unifAvgArr[kInd - 1]) / (unifAvgArr[kInd] - unifAvgArr[kInd - 1]) * testInt) + (
            kInd - 1) * testInt
    print(unifSampSaved)
    '''29-MAR: 31 saved'''
    # Rudimentary
    minListLen = np.min([len(i) for i in origUtilList])
    origUtilArr = np.array([i[:minListLen] for i in origUtilList])
    origAvgArr = np.average(origUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(origAvgArr.tolist()) if val > compAvg)
    origSampSaved = round(
        (compAvg - origAvgArr[kInd - 1]) / (origAvgArr[kInd] - origAvgArr[kInd - 1]) * testInt * 3) + (
                            kInd - 1) * testInt * 3
    print(origSampSaved)
    '''29-MAR: 302 saved'''

    # Do again for different sample budget
    sampBudget = 90
    allocArr = forwardAllocateWithBudget(avgUtilMat, int(sampBudget / testInt))
    designArr = allocArr / np.sum(allocArr, axis=0)
    # Get utility for this allocation at the sample budget
    utilDict.update({'method': 'weightsNodeDraw4linear'})
    compUtilList, unifUtilList, origUtilList = [], [], []
    numReps = 5
    for rep in range(numReps):
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New Bayes draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        print('Generating loss matrix...')
        tempLossMat = lf.lossMatSetBayesDraws(dictTemp['postSamples'], lossDict.copy(), lossDict['bayesDraws'])
        tempLossDict = lossDict.copy()
        tempLossDict.update({'lossMat': tempLossMat})
        newBayesDraws, newLossMat = lf.add_cand_neighbors(tempLossDict.copy(), CSdict3['postSamples'],
                                                          dictTemp['postSamples'])
        tempLossDict.update({'bayesDraws': newBayesDraws, 'lossMat': newLossMat})
        baseLoss = (np.sum(newLossMat, axis=1) / newLossMat.shape[1]).min()
        # Get a new set of data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        currCompUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[designArr],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Heuristic utility: ' + str(currCompUtil))
        compUtilList.append(currCompUtil)
        '''29-MAR
        compUtilList = [0.57517490160261, 0.6021543688185291, 0.6536206889854768, 0.6297070488537075, 0.6206070438104385]
        '''
        # Find the equivalent uniform allocation
        currUnifUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[unifDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Uniform utility: ' + str(currUnifUtil))
        unifUtilList.append([currUnifUtil])
        unifAdd, contUnif, unifCount = 0, False, 0
        if currUnifUtil < currCompUtil:
            contUnif = True
        while contUnif:
            unifAdd += testInt
            print('Adding ' + str(unifAdd) + ' for uniform')
            currUnifUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[unifDes], numtests=sampBudget + unifAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currUnifUtil))
            unifUtilList[rep].append(currUnifUtil)
            if currUnifUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if unifCount < 3:
                    unifCount += 1
                else:
                    contUnif = False
        '''29-MAR
        unifUtilList = [[0.49459387955501377, 0.5299438485418024, 0.564221743030326, 0.6054462042891195, 0.6437224039033316, 0.6867434256507914, 0.7157441951663488], [0.5072273852052187, 0.545124252703101, 0.5828697272426591, 0.6221137495139626, 0.6677118413726126, 0.7053774367603238, 0.729827780253768], [0.5638673349190548, 0.5998321280454473, 0.6399280979808144, 0.6929064644286149, 0.7242141460846354, 0.7648391417738791, 0.7953049073999789], [0.5412109221417545, 0.5780289648132411, 0.6244859569550103, 0.6665020586464325, 0.7037093261308729, 0.7334005518095208, 0.7768367275285688], [0.5357362988569463, 0.5683018099220463, 0.6127040913733399, 0.6619540943328874, 0.6991463239435958, 0.7330231891021532, 0.7664005608579547]]
        '''
        # Find the equivalent rudimentary allocation
        currOrigUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[origDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Rudimentary utility: ' + str(currOrigUtil))
        origUtilList.append([currOrigUtil])
        origAdd, contOrig, origCount = 0, False, 0
        if currOrigUtil < currCompUtil:
            contOrig = True
        while contOrig:
            origAdd += testInt * 3
            print('Adding ' + str(origAdd) + ' for rudimentary')
            currOrigUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[origDes], numtests=sampBudget + origAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currOrigUtil))
            origUtilList[rep].append(currOrigUtil)
            if currOrigUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if origCount < 3:
                    origCount += 1
                else:
                    contOrig = False
        '''29-MAR
        origUtilList = [[0.18014099725961774, 0.23173877810048804, 0.2959391074979547, 0.3534971982748365, 0.41142868028012947, 0.47882312271450633, 0.5322204926342247, 0.6073935509866519, 0.6525870280989903, 0.7199608335720948, 0.7915390659525445], [0.18655314788320165, 0.24548971198625358, 0.2989882782664677, 0.3572840442802545, 0.4151576357439164, 0.4782284715130638, 0.5324452997467288, 0.606320127354893, 0.6582370922660306, 0.7259284321237454, 0.8036629000742961], [0.20892500821260196, 0.268011840968831, 0.32754618759712484, 0.38523154787265357, 0.4429631335021056, 0.5025818905297537, 0.5676755721548101, 0.6333083882641994, 0.6917052511743607, 0.7582080966800597, 0.8380240614721557, 0.8657994499036668], [0.20878932748910328, 0.2716911585898987, 0.3306079475278021, 0.39623080488428064, 0.4590071996093714, 0.5292928326868238, 0.5903958920559229, 0.6486254460513594, 0.70356366251076, 0.7887548817395054, 0.8415162644525926], [0.20759767392712103, 0.26874049937375677, 0.3271129644607145, 0.3954219991837391, 0.452320539462276, 0.5118599769741028, 0.5775656070528239, 0.6473513798591966, 0.7075900539225413, 0.7694519357035681, 0.8474473790044947]]
        '''
    compAvg = np.average(compUtilList)
    # Locate closest sample point for uniform and rudimentary to compAvg
    minListLen = np.min([len(i) for i in unifUtilList])
    unifUtilArr = np.array([i[:minListLen] for i in unifUtilList])
    unifAvgArr = np.average(unifUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(unifAvgArr.tolist()) if val > compAvg)
    unifSampSaved = round((compAvg - unifAvgArr[kInd - 1]) / (unifAvgArr[kInd] - unifAvgArr[kInd - 1]) * testInt) + (
            kInd - 1) * testInt
    print(unifSampSaved)
    '''29-MAR: 23 saved'''
    # Rudimentary
    minListLen = np.min([len(i) for i in origUtilList])
    origUtilArr = np.array([i[:minListLen] for i in origUtilList])
    origAvgArr = np.average(origUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(origAvgArr.tolist()) if val > compAvg)
    origSampSaved = round(
        (compAvg - origAvgArr[kInd - 1]) / (origAvgArr[kInd] - origAvgArr[kInd - 1]) * testInt * 3) + (
                            kInd - 1) * testInt * 3
    print(origSampSaved)
    '''29-MAR: 205 saved'''

    ##############################################
    ##############################################
    # Choose different sourcing matrix (PART 2)
    ##############################################
    ##############################################
    numBoot = 44  # 44 is average across each TN in original data set
    SNprobs = np.sum(CSdict3['N'], axis=0) / np.sum(CSdict3['N'])
    np.random.seed(36)  # Chosen to be "far" from seed 33
    Qvecs = np.random.multinomial(numBoot, SNprobs, size=numTN - 4) / numBoot
    CSdict3['Q'] = np.vstack((CSdict3['N'][:4] / np.sum(CSdict3['N'][:4], axis=1).reshape(4, 1), Qvecs))

    sampBudget = 180
    unifDes = np.zeros(numTN) + 1 / numTN
    origDes = np.sum(rd3_N, axis=1) / np.sum(rd3_N)

    # Use original loss parameters
    paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                                  marketvec=np.ones(numTN + numSN), candneighnum=1000)

    numtargetdraws, numDataDraws = 5100, 5000

    # Find heuristic allocation first
    utilDict = {'method': 'weightsNodeDraw3linear'}

    numReps = 10
    utilMatList = []
    # set testMax to highest expected allocation for any one node
    testMax, testInt = 90, 10
    testArr = np.arange(0, testMax + 1, testInt)
    for rep in range(numReps):
        # New MCMC draws
        CSdict3 = methods.GeneratePostSamples(CSdict3)
        # Withdraw a subset of MCMC prior draws
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New loss draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        # Get new data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        # Get marginal utilities at each test node
        currMargUtilMat = GetMargUtilAtNodes(dictTemp.copy(), testMax, testInt, lossDict.copy(), utilDict.copy(),
                                             masterDraws=CSdict3['postSamples'], printUpdate=True)
        print(repr(currMargUtilMat))
        utilMatList.append(currMargUtilMat)
        # plot
        colors = cm.rainbow(np.linspace(0, 1., numTN))
        for mat in utilMatList:
            for i in range(8):
                plt.plot(testArr, mat[i], color=colors[i], linewidth=0.2)
        avgUtilMat = np.average(np.array(utilMatList), axis=0)
        for i in range(8):
            plt.plot(testArr, avgUtilMat[i], color=colors[i], linewidth=2)
        plt.ylim([0, 0.4])
        plt.title('Comprehensive utility for Rudimentary')
        plt.show()
        plt.close()
        '''14-APR run
        utilMatList = [np.array([[0.        , 0.02444741, 0.04617417, 0.07172455, 0.08809567,
        0.10640526, 0.12366133, 0.13573863, 0.15604136, 0.17068028,
        0.18108176],
       [0.        , 0.04834982, 0.0937102 , 0.13237818, 0.16873718,
        0.19415305, 0.21981798, 0.24330678, 0.26916736, 0.28552992,
        0.30819564],
       [0.        , 0.01205559, 0.0272295 , 0.03830821, 0.05469283,
        0.06533469, 0.08095158, 0.09126618, 0.10626761, 0.11855323,
        0.13183308],
       [0.        , 0.00889473, 0.01812639, 0.02379298, 0.03212082,
        0.03628557, 0.03975221, 0.04636525, 0.05213653, 0.05516324,
        0.06090449],
       [0.        , 0.06774564, 0.09412483, 0.11245624, 0.1295772 ,
        0.13871881, 0.14940352, 0.16106266, 0.16738117, 0.17591085,
        0.1827334 ],
       [0.        , 0.09450995, 0.12544353, 0.14686126, 0.15972911,
        0.17444362, 0.18053802, 0.19119856, 0.19830664, 0.20612721,
        0.21666246],
       [0.        , 0.10812779, 0.13866387, 0.16121653, 0.17872359,
        0.18995057, 0.20162464, 0.2143453 , 0.22490877, 0.23183825,
        0.23978482],
       [0.        , 0.0225108 , 0.05364471, 0.07401074, 0.08804033,
        0.10184499, 0.11381971, 0.12134715, 0.12895082, 0.1392264 ,
        0.14832965]]), 
        np.array([[0.        , 0.01061772, 0.03201993, 0.0476978 , 0.06522253,
        0.08445483, 0.10002001, 0.1130361 , 0.1314298 , 0.14692459,
        0.15758143],
       [0.        , 0.02270379, 0.05805478, 0.09679276, 0.12803155,
        0.15691295, 0.18389047, 0.20513688, 0.22898513, 0.25110721,
        0.26810876],
       [0.        , 0.01228162, 0.02790018, 0.04423467, 0.05821603,
        0.06911803, 0.08560955, 0.09989748, 0.11257363, 0.12876694,
        0.13481379],
       [0.        , 0.00035913, 0.00169294, 0.00579694, 0.00880823,
        0.01714156, 0.02000026, 0.02557278, 0.03122706, 0.03301323,
        0.03981037],
       [0.        , 0.05824016, 0.08573851, 0.10101961, 0.11410206,
        0.12444768, 0.13687999, 0.14131011, 0.15409859, 0.15994816,
        0.1647348 ],
       [0.        , 0.03672482, 0.05983045, 0.07725387, 0.09468524,
        0.10663863, 0.11931555, 0.1288919 , 0.13691828, 0.14544028,
        0.15448448],
       [0.        , 0.05017085, 0.08021415, 0.10054485, 0.11795222,
        0.13324094, 0.14550968, 0.15740122, 0.16794647, 0.17424362,
        0.18580221],
       [0.        , 0.08377163, 0.11383486, 0.13477952, 0.14504161,
        0.1617759 , 0.16763926, 0.17146587, 0.18497091, 0.18961659,
        0.19583548]]), 
        np.array([[0.        , 0.05289022, 0.08776998, 0.12225116, 0.14337044,
        0.16746483, 0.18300634, 0.20319633, 0.22121103, 0.23645388,
        0.25288762],
       [0.        , 0.09949844, 0.16303964, 0.21023837, 0.25285218,
        0.28553042, 0.31828874, 0.33517058, 0.36180306, 0.38057189,
        0.40137325],
       [0.        , 0.02774304, 0.05555077, 0.08177219, 0.10254351,
        0.12142655, 0.14443328, 0.16473343, 0.18135736, 0.19966655,
        0.21823272],
       [0.        , 0.0116689 , 0.02846629, 0.03894043, 0.05353948,
        0.06113148, 0.07123619, 0.0769341 , 0.08715056, 0.09303578,
        0.09951172],
       [0.        , 0.11733075, 0.15430028, 0.17623312, 0.18990536,
        0.20403717, 0.2156951 , 0.2284347 , 0.23739609, 0.24569393,
        0.25454135],
       [0.        , 0.08405904, 0.12345312, 0.14851909, 0.16747762,
        0.18197873, 0.19359453, 0.20492001, 0.21534388, 0.22361092,
        0.23399031],
       [0.        , 0.09027289, 0.13715567, 0.16395737, 0.18549652,
        0.20305894, 0.21722925, 0.22930705, 0.2432413 , 0.25674125,
        0.26186968],
       [0.        , 0.07602299, 0.10753734, 0.131441  , 0.14890382,
        0.16484792, 0.17376019, 0.18356386, 0.19582632, 0.2041181 ,
        0.21100653]]), 
        np.array([[0.        , 0.04280313, 0.07743959, 0.10556424, 0.12466443,
        0.15066002, 0.16781309, 0.18275166, 0.20490615, 0.22078829,
        0.23297169],
       [0.        , 0.09521162, 0.1569858 , 0.20584074, 0.24333427,
        0.27283366, 0.29946565, 0.32765417, 0.35167152, 0.37209175,
        0.38894874],
       [0.        , 0.02686715, 0.05106474, 0.07126627, 0.08877093,
        0.1031878 , 0.11952822, 0.13340527, 0.14928755, 0.16425047,
        0.18184323],
       [0.        , 0.02426568, 0.04074958, 0.05608495, 0.06565289,
        0.07453899, 0.08215042, 0.08985788, 0.09804557, 0.10417867,
        0.107827  ],
       [0.        , 0.12188501, 0.16242993, 0.1906887 , 0.20550559,
        0.21944671, 0.23029986, 0.24028658, 0.25159644, 0.25906787,
        0.26717226],
       [0.        , 0.12862406, 0.17109328, 0.19499282, 0.21154867,
        0.22271788, 0.24070267, 0.24711458, 0.25963004, 0.26591217,
        0.27245365],
       [0.        , 0.14447852, 0.18800221, 0.21547517, 0.23340096,
        0.24737714, 0.26132212, 0.27339058, 0.28408912, 0.29658744,
        0.30412029],
       [0.        , 0.08420287, 0.12093209, 0.14498087, 0.162028  ,
        0.17444314, 0.18940581, 0.19729391, 0.20525447, 0.21334051,
        0.22009713]]), 
        np.array([[0.        , 0.03677073, 0.06687004, 0.095653  , 0.12444647,
        0.14561554, 0.1628917 , 0.18565958, 0.20149133, 0.22313254,
        0.2322078 ],
       [0.        , 0.0921443 , 0.15064145, 0.19596488, 0.24356864,
        0.27816161, 0.31071725, 0.333375  , 0.36150312, 0.37992713,
        0.40054777],
       [0.        , 0.03158609, 0.06114475, 0.08476411, 0.10382951,
        0.12495941, 0.14603506, 0.16075655, 0.17923553, 0.20015804,
        0.21033831],
       [0.        , 0.01119459, 0.02237529, 0.03206565, 0.04215204,
        0.04978128, 0.05579398, 0.06215214, 0.0702405 , 0.07638155,
        0.08086721],
       [0.        , 0.11111867, 0.1549965 , 0.17697585, 0.1972016 ,
        0.20931262, 0.21969137, 0.22900297, 0.24178031, 0.24601579,
        0.25404951],
       [0.        , 0.11530631, 0.15794428, 0.18727605, 0.20385003,
        0.21417796, 0.22583792, 0.23883884, 0.24886706, 0.25156618,
        0.25957483],
       [0.        , 0.14557227, 0.19756024, 0.22939661, 0.24970096,
        0.26761658, 0.28378273, 0.29626806, 0.30821247, 0.32048759,
        0.32567444],
       [0.        , 0.07029971, 0.10043551, 0.11912466, 0.13398989,
        0.14420831, 0.15682328, 0.16601601, 0.17448884, 0.18324531,
        0.19043351]]), 
        np.array([[0.        , 0.02915052, 0.0568921 , 0.08468161, 0.10841281,
        0.1256468 , 0.15082809, 0.16613433, 0.18881289, 0.20107538,
        0.22283628],
       [0.        , 0.06722355, 0.12182064, 0.16474724, 0.20309623,
        0.23478364, 0.26400862, 0.28802007, 0.3070428 , 0.33282133,
        0.34965786],
       [0.        , 0.02421449, 0.04588543, 0.06750901, 0.08666779,
        0.10563033, 0.11836118, 0.13406416, 0.15403105, 0.1707277 ,
        0.18143652],
       [0.        , 0.01659786, 0.02853722, 0.03972463, 0.04595229,
        0.05307529, 0.0632196 , 0.0665243 , 0.07286573, 0.07985776,
        0.08586452],
       [0.        , 0.16011224, 0.20209676, 0.22314093, 0.23641936,
        0.25186789, 0.25763824, 0.26802574, 0.27553782, 0.27996404,
        0.28841742],
       [0.        , 0.15776267, 0.19360888, 0.2209689 , 0.23759046,
        0.24785269, 0.25982594, 0.26423594, 0.2785833 , 0.28487747,
        0.28995016],
       [0.        , 0.14362273, 0.1913843 , 0.21811577, 0.23896475,
        0.25286501, 0.26819642, 0.28062925, 0.29024799, 0.30003273,
        0.31225241],
       [0.        , 0.06856841, 0.10010141, 0.11958507, 0.13581957,
        0.14551666, 0.15676986, 0.16281806, 0.17302035, 0.18118008,
        0.18749527]]), 
        np.array([[0.        , 0.01685809, 0.03712153, 0.06386612, 0.08729323,
        0.10173939, 0.12468433, 0.14379201, 0.155036  , 0.17306744,
        0.1890389 ],
       [0.        , 0.06628199, 0.11629691, 0.15890264, 0.19527346,
        0.22415608, 0.25060362, 0.27294009, 0.28904085, 0.30917314,
        0.33095715],
       [0.        , 0.01241213, 0.02748692, 0.03924473, 0.05663609,
        0.0730707 , 0.09158386, 0.10612843, 0.11712627, 0.12982069,
        0.13590827],
       [0.        , 0.0041729 , 0.00872908, 0.01866805, 0.02462395,
        0.03099282, 0.03625286, 0.04495835, 0.04780783, 0.05156178,
        0.05706089],
       [0.        , 0.11300693, 0.1453673 , 0.16376946, 0.17541776,
        0.18375239, 0.19277467, 0.19876675, 0.20420916, 0.21007576,
        0.21859571],
       [0.        , 0.05616503, 0.08518666, 0.10669408, 0.12662187,
        0.13739913, 0.1503795 , 0.15945393, 0.1699209 , 0.17788696,
        0.18649542],
       [0.        , 0.04563272, 0.07904905, 0.10405482, 0.12060336,
        0.13658139, 0.14915913, 0.16392047, 0.17487578, 0.18559586,
        0.19488231],
       [0.        , 0.07310888, 0.10032254, 0.11892637, 0.13223117,
        0.14410172, 0.15266497, 0.15959014, 0.17010732, 0.17246216,
        0.18312472]]), 
        np.array([[0.        , 0.02333424, 0.038126  , 0.05818243, 0.07652159,
        0.0930885 , 0.1091782 , 0.12290162, 0.13782104, 0.14904138,
        0.16018488],
       [0.        , 0.03753614, 0.0740499 , 0.10038972, 0.13316401,
        0.15476415, 0.17949055, 0.19536641, 0.21502607, 0.23157766,
        0.2469962 ],
       [0.        , 0.00895256, 0.02081164, 0.03044111, 0.04464267,
        0.0561284 , 0.06999362, 0.08211742, 0.0950575 , 0.10494854,
        0.11417126],
       [0.        , 0.00823575, 0.01855312, 0.02363607, 0.0323064 ,
        0.0380835 , 0.04394861, 0.04985392, 0.05145138, 0.05953345,
        0.06464969],
       [0.        , 0.11527287, 0.1568247 , 0.18019577, 0.19212014,
        0.20592982, 0.21249921, 0.21854016, 0.22560234, 0.23050269,
        0.23262888],
       [0.        , 0.05978155, 0.08810741, 0.10831188, 0.1261699 ,
        0.13524031, 0.14512738, 0.15539314, 0.16376151, 0.17180738,
        0.17320792],
       [0.        , 0.04695933, 0.07678728, 0.09652508, 0.11278355,
        0.12640277, 0.13917011, 0.14785176, 0.15689272, 0.16630099,
        0.17723344],
       [0.        , 0.0395691 , 0.06081441, 0.07881922, 0.09178753,
        0.10281127, 0.11002848, 0.11912296, 0.12548132, 0.12947523,
        0.13695145]]), 
        np.array([[0.        , 0.03537446, 0.06434736, 0.0856373 , 0.11161541,
        0.13210294, 0.14932332, 0.16405937, 0.17799343, 0.19429886,
        0.20858928],
       [0.        , 0.07583115, 0.12618876, 0.16624427, 0.19729915,
        0.22647003, 0.2514849 , 0.2716841 , 0.29316011, 0.31267927,
        0.33411131],
       [0.        , 0.01361828, 0.02754478, 0.04397554, 0.05980463,
        0.07744601, 0.09610903, 0.10819553, 0.1236232 , 0.13642089,
        0.15309482],
       [0.        , 0.00632155, 0.01005745, 0.01471325, 0.02291924,
        0.03114394, 0.0372675 , 0.04043668, 0.05063683, 0.05583   ,
        0.06027561],
       [0.        , 0.1503635 , 0.18959642, 0.21300263, 0.22853748,
        0.24320318, 0.25205858, 0.26228675, 0.26869937, 0.27525275,
        0.2836008 ],
       [0.        , 0.08250832, 0.11869987, 0.14112102, 0.15912721,
        0.17229714, 0.18176564, 0.19763742, 0.20712643, 0.21480984,
        0.22325277],
       [0.        , 0.09788981, 0.14107614, 0.16551395, 0.18513062,
        0.1997572 , 0.21383891, 0.22361878, 0.23518644, 0.24617805,
        0.25604735],
       [0.        , 0.0674143 , 0.10782412, 0.13079347, 0.14544611,
        0.16067363, 0.17133179, 0.1843107 , 0.19235671, 0.20081196,
        0.20608015]]), 
        np.array([[0.        , 0.05191602, 0.09051411, 0.11724189, 0.1393711 ,
        0.15867739, 0.17891094, 0.19748465, 0.21374922, 0.22885509,
        0.24299431],
       [0.        , 0.10800944, 0.17665256, 0.2210505 , 0.26385208,
        0.29367264, 0.32463802, 0.34848621, 0.37509772, 0.39464624,
        0.41316588],
       [0.        , 0.03427224, 0.06171548, 0.07909405, 0.10659685,
        0.12436567, 0.14538645, 0.16099835, 0.17953662, 0.19933655,
        0.2152309 ],
       [0.        , 0.02075514, 0.03354546, 0.04442889, 0.05490154,
        0.06429095, 0.07580695, 0.08077498, 0.08569503, 0.09475651,
        0.10079062],
       [0.        , 0.13679852, 0.17078316, 0.19376287, 0.21390631,
        0.22634267, 0.2390888 , 0.24822102, 0.25819282, 0.26382268,
        0.27277994],
       [0.        , 0.10328089, 0.1448251 , 0.16706774, 0.18566444,
        0.19990438, 0.21393025, 0.22114451, 0.2332845 , 0.24145801,
        0.25118244],
       [0.        , 0.17315293, 0.21209315, 0.2331145 , 0.24829703,
        0.26037815, 0.26945285, 0.28150528, 0.29312042, 0.30356294,
        0.30862569],
       [0.        , 0.11491338, 0.15670171, 0.18254758, 0.19958868,
        0.21408513, 0.22369242, 0.23584977, 0.2449036 , 0.25242339,
        0.25671282]])]
        '''
    '''29-MAR run
    utilMatList = [np.array([[0.        , 0.01916123, 0.05181039, 0.07176857, 0.0939536 ,
        0.11665873, 0.13382754, 0.15307864, 0.16597318, 0.18088208,
        0.19074497],
       [0.        , 0.07123407, 0.12863243, 0.17573619, 0.21330149,
        0.24336048, 0.26885429, 0.29347271, 0.31684328, 0.33613847,
        0.35288842],
       [0.        , 0.01293205, 0.02714917, 0.0448176 , 0.0606202 ,
        0.07777586, 0.09121254, 0.11004615, 0.12385605, 0.14222894,
        0.15473014],
       [0.        , 0.00927432, 0.01959722, 0.02707693, 0.03733776,
        0.04786388, 0.05198462, 0.06052032, 0.06952814, 0.07160066,
        0.08211004],
       [0.        , 0.07833316, 0.11676237, 0.14068258, 0.15639652,
        0.17224677, 0.18558647, 0.19247791, 0.20269915, 0.21059857,
        0.21839275],
       [0.        , 0.07954014, 0.11162262, 0.13761391, 0.15862485,
        0.17156601, 0.18074959, 0.19447891, 0.19964158, 0.20810247,
        0.22087342],
       [0.        , 0.05948976, 0.09408648, 0.1169816 , 0.13970638,
        0.1553303 , 0.16983777, 0.18339059, 0.19326457, 0.20825797,
        0.21591629],
       [0.        , 0.04224428, 0.07142509, 0.09397344, 0.11107657,
        0.12399538, 0.13944983, 0.14960728, 0.15965968, 0.16746126,
        0.17492618]]), 
        np.array([[0.        , 0.02444741, 0.04617417, 0.07172455, 0.08809567,
        0.10640526, 0.12366133, 0.13573863, 0.15604136, 0.17068028,
        0.18108176],
       [0.        , 0.04834982, 0.0937102 , 0.13237818, 0.16873718,
        0.19415305, 0.21981798, 0.24330678, 0.26916736, 0.28552992,
        0.30819564],
       [0.        , 0.01205559, 0.0272295 , 0.03830821, 0.05469283,
        0.06533469, 0.08095158, 0.09126618, 0.10626761, 0.11855323,
        0.13183308],
       [0.        , 0.00889473, 0.01812639, 0.02379298, 0.03212082,
        0.03628557, 0.03975221, 0.04636525, 0.05213653, 0.05516324,
        0.06090449],
       [0.        , 0.06774564, 0.09412483, 0.11245624, 0.1295772 ,
        0.13871881, 0.14940352, 0.16106266, 0.16738117, 0.17591085,
        0.1827334 ],
       [0.        , 0.09450995, 0.12544353, 0.14686126, 0.15972911,
        0.17444362, 0.18053802, 0.19119856, 0.19830664, 0.20612721,
        0.21666246],
       [0.        , 0.10812779, 0.13866387, 0.16121653, 0.17872359,
        0.18995057, 0.20162464, 0.2143453 , 0.22490877, 0.23183825,
        0.23978482],
       [0.        , 0.0225108 , 0.05364471, 0.07401074, 0.08804033,
        0.10184499, 0.11381971, 0.12134715, 0.12895082, 0.1392264 ,
        0.14832965]]),
        np.array([[0.        , 0.03688257, 0.06942722, 0.0986451 , 0.12363724,
        0.1444222 , 0.16491802, 0.18163413, 0.19939142, 0.2154648 ,
        0.22861321],
       [0.        , 0.08937856, 0.15160417, 0.19806662, 0.23075377,
        0.26489721, 0.29034409, 0.3166018 , 0.3327706 , 0.36072241,
        0.37969892],
       [0.        , 0.02690672, 0.04967761, 0.06950713, 0.0916044 ,
        0.10889991, 0.124801  , 0.14669005, 0.1615694 , 0.1746577 ,
        0.18871092],
       [0.        , 0.03062442, 0.05155608, 0.07051583, 0.07987877,
        0.08958365, 0.1010316 , 0.11037498, 0.11647938, 0.12390895,
        0.13149895],
       [0.        , 0.09297744, 0.12973616, 0.14960283, 0.16459473,
        0.17741971, 0.18859341, 0.19816849, 0.20518692, 0.21591727,
        0.22236878],
       [0.        , 0.14898993, 0.18603177, 0.20643587, 0.22636166,
        0.23868471, 0.25110112, 0.25921092, 0.26612525, 0.2751434 ,
        0.28523691],
       [0.        , 0.15388162, 0.18914232, 0.2108138 , 0.22730157,
        0.24142528, 0.25602512, 0.26850677, 0.27652058, 0.28651353,
        0.29496382],
       [0.        , 0.08095088, 0.11642536, 0.14006123, 0.15534911,
        0.16619867, 0.17887609, 0.19101872, 0.19787496, 0.20937628,
        0.21415914]]), 
        np.array([[0.        , 0.03243253, 0.06575737, 0.09349498, 0.11953655,
        0.13410331, 0.15868973, 0.17560208, 0.1858635 , 0.20315486,
        0.21967945],
       [0.        , 0.10655272, 0.17512483, 0.22321035, 0.25985149,
        0.29820123, 0.3216317 , 0.3461524 , 0.36876506, 0.38861687,
        0.40573627],
       [0.        , 0.02600881, 0.0433352 , 0.06602951, 0.08514688,
        0.104974  , 0.11777774, 0.13787325, 0.14893703, 0.16589559,
        0.17996964],
       [0.        , 0.0193368 , 0.03309737, 0.04611571, 0.05906813,
        0.06521012, 0.0706446 , 0.08101618, 0.08633629, 0.09430748,
        0.09786254],
       [0.        , 0.11301499, 0.14623482, 0.1638337 , 0.1783922 ,
        0.18872453, 0.19643348, 0.20259243, 0.20882419, 0.21730706,
        0.22295835],
       [0.        , 0.09439459, 0.13548305, 0.16145378, 0.18008592,
        0.19477363, 0.20737885, 0.21761525, 0.22488797, 0.23572667,
        0.24581601],
       [0.        , 0.10064325, 0.14512173, 0.17226591, 0.19312869,
        0.20975346, 0.22509739, 0.23678888, 0.24825325, 0.25801498,
        0.26922433],
       [0.        , 0.09972989, 0.12831599, 0.14740526, 0.16136355,
        0.1725875 , 0.1811033 , 0.19155961, 0.19835089, 0.20594329,
        0.21125719]]), 
        np.array([[0.        , 0.0229097 , 0.05089532, 0.08127111, 0.10208739,
        0.12054716, 0.13868818, 0.15410349, 0.17308676, 0.18652009,
        0.19765686],
       [0.        , 0.06403967, 0.1229784 , 0.1678825 , 0.20732827,
        0.24002676, 0.27113397, 0.29435436, 0.31460252, 0.3330429 ,
        0.35476001],
       [0.        , 0.01377925, 0.02975483, 0.04715424, 0.06688946,
        0.0830543 , 0.09715904, 0.11606282, 0.13220583, 0.14400495,
        0.16309874],
       [0.        , 0.01039683, 0.02014691, 0.02883477, 0.039243  ,
        0.04732814, 0.05667218, 0.06329123, 0.07117819, 0.07903692,
        0.08356246],
       [0.        , 0.07889522, 0.11359422, 0.13338368, 0.14459998,
        0.15994707, 0.17256015, 0.18101375, 0.18964728, 0.19610327,
        0.20386869],
       [0.        , 0.07526335, 0.11403298, 0.1426148 , 0.15869649,
        0.17269007, 0.1872962 , 0.19647352, 0.20657374, 0.21679212,
        0.22413142],
       [0.        , 0.05585806, 0.09845501, 0.12392014, 0.14571524,
        0.16248438, 0.17866193, 0.19024707, 0.20156331, 0.20937352,
        0.22278807],
       [0.        , 0.035187  , 0.06761368, 0.09059733, 0.11182431,
        0.12296252, 0.13625382, 0.14775136, 0.15609829, 0.16745415,
        0.17541087]])]
    '''
    # Get average utility matrix
    avgUtilMat = np.average(np.array(utilMatList), axis=0)
    '''29-MAR
    avgUtilMat = np.array([[0.        , 0.02716669, 0.05681289, 0.08338086, 0.10546209,
        0.12442733, 0.14395696, 0.16003139, 0.17607125, 0.19134042,
        0.20355525],
       [0.        , 0.07591097, 0.13441001, 0.17945477, 0.21599444,
        0.24812775, 0.2743564 , 0.29877761, 0.32042976, 0.34081012,
        0.36025585],
       [0.        , 0.01833648, 0.03542926, 0.05316334, 0.07179075,
        0.08800775, 0.10238038, 0.12038769, 0.13456718, 0.14906808,
        0.1636685 ],
       [0.        , 0.01570542, 0.02850479, 0.03926725, 0.0495297 ,
        0.05725427, 0.06401704, 0.07231359, 0.0791317 , 0.08480345,
        0.0911877 ],
       [0.        , 0.08619329, 0.12009048, 0.13999181, 0.15471213,
        0.16741138, 0.17851541, 0.18706305, 0.19474774, 0.2031674 ,
        0.21006439],
       [0.        , 0.09853959, 0.13452279, 0.15899593, 0.17669961,
        0.19043161, 0.20141275, 0.21179543, 0.21910704, 0.22837837,
        0.23854405],
       [0.        , 0.09560009, 0.13309388, 0.1570396 , 0.17691509,
        0.1917888 , 0.20624937, 0.21865572, 0.2289021 , 0.23879965,
        0.24853547],
       [0.        , 0.05612457, 0.08748497, 0.1092096 , 0.12553077,
        0.13751781, 0.14990055, 0.16025682, 0.16818693, 0.17789228,
        0.18481661]])
    '''
    # Find allocation for sample budget
    allocArr = forwardAllocateWithBudget(avgUtilMat, int(sampBudget / testInt))
    designArr = allocArr / np.sum(allocArr, axis=0)
    # Get utility for this allocation at the sample budget
    utilDict.update({'method': 'weightsNodeDraw4linear'})
    compUtilList, unifUtilList, origUtilList = [], [], []
    numReps = 5
    for rep in range(numReps):
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New Bayes draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        print('Generating loss matrix...')
        tempLossMat = lf.lossMatSetBayesDraws(dictTemp['postSamples'], lossDict.copy(), lossDict['bayesDraws'])
        tempLossDict = lossDict.copy()
        tempLossDict.update({'lossMat': tempLossMat})
        newBayesDraws, newLossMat = lf.add_cand_neighbors(tempLossDict.copy(), CSdict3['postSamples'],
                                                          dictTemp['postSamples'])
        tempLossDict.update({'bayesDraws': newBayesDraws, 'lossMat': newLossMat})
        baseLoss = (np.sum(newLossMat, axis=1) / newLossMat.shape[1]).min()
        # Get a new set of data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        currCompUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[designArr],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Heuristic utility: ' + str(currCompUtil))
        compUtilList.append(currCompUtil)
        '''29-MAR
        compUtilList = [0.89393015878235, 0.9291397682281892, 0.8955522006472445, 0.9700016704092347, 0.9232853470408084]
        '''
        # Find the equivalent uniform allocation
        currUnifUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[unifDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Uniform utility: ' + str(currUnifUtil))
        unifUtilList.append([currUnifUtil])
        unifAdd, contUnif, unifCount = 0, False, 0
        if currUnifUtil < currCompUtil:
            contUnif = True
        while contUnif:
            unifAdd += testInt
            print('Adding ' + str(unifAdd) + ' for uniform')
            currUnifUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[unifDes], numtests=sampBudget + unifAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currUnifUtil))
            unifUtilList[rep].append(currUnifUtil)
            if currUnifUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if unifCount < 3:
                    unifCount += 1
                else:
                    contUnif = False
        '''29-MAR
        unifUtilList = [[0.8286751805851535, 0.8486219845924294, 0.8851138409722776, 0.9142771517311297, 0.9429601252070339, 0.96425883377802, 1.0129102905279863], [0.8469882962575208, 0.8771832333517819, 0.8968624383780615, 0.9508982043623706, 0.9763939799826287, 0.989126868637733, 1.0284100421427915], [0.8182888881802626, 0.852037473719105, 0.8899528670945567, 0.9185433962769829, 0.9426491142086983, 0.9808751950500407, 1.0038122609871536], [0.8775014378230686, 0.9047567542655623, 0.9412297922744899, 0.9824667520104962, 1.0033976393195716, 1.0279699621877096, 1.0695739006412093], [0.8346095720814484, 0.861308967756873, 0.9001448882936467, 0.9380564970849039, 0.9757756753664419, 0.9869479725412971, 1.0044304214287294]]
        '''
        # Find the equivalent rudimentary allocation
        currOrigUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[origDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Rudimentary utility: ' + str(currOrigUtil))
        origUtilList.append([currOrigUtil])
        origAdd, contOrig, origCount = 0, False, 0
        if currOrigUtil < currCompUtil:
            contOrig = True
        while contOrig:
            origAdd += testInt * 3
            print('Adding ' + str(origAdd) + ' for rudimentary')
            currOrigUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[origDes], numtests=sampBudget + origAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currOrigUtil))
            origUtilList[rep].append(currOrigUtil)
            if currOrigUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if origCount < 3:
                    origCount += 1
                else:
                    contOrig = False
        '''29-MAR
        origUtilList = [[0.3753126019491084, 0.4361681926676968, 0.4992170755993044, 0.559300509519717, 0.6220572014590235, 0.6841804034718537, 0.7442804580682632, 0.812016549043332, 0.8593015313220502, 0.9119399325121358, 0.9661697826074711, 1.0221525183029914, 1.065196852501399], [0.39337180800135796, 0.447473052587517, 0.5110019654982714, 0.5723082848632592, 0.6354757857963493, 0.7087005944380875, 0.7782824580481704, 0.8499462501302024, 0.8821863857807206, 0.9493357017777697, 0.9897499605144966, 1.0442294223395368, 1.1135614007524919], [0.36765995558770603, 0.42681966080405553, 0.4826957101814937, 0.5555379172828747, 0.6124760829499203, 0.6795463191809956, 0.7237298202226383, 0.7960624012748871, 0.8521753424702272, 0.896045868285186, 0.9488492911385786, 0.9914786006687777, 1.0647327044690043], [0.40024367176350806, 0.45566951788152865, 0.5254261964709057, 0.5729267542603727, 0.644980234698656, 0.724733860174259, 0.7792770759075243, 0.8561487750542205, 0.8932918708992941, 0.9409855449919728, 0.9910352866485002, 1.0652855195276425, 1.1033442901953712, 1.1642272542096572], [0.3756777399736895, 0.4372745967159486, 0.5000291187797661, 0.5598014845589505, 0.6241646411749158, 0.6784022111841654, 0.7316113422151367, 0.8141646140193468, 0.852187370420801, 0.9143604482120242, 0.9655130667985632, 1.0172369719606813, 1.0742484790496651, 1.1208804948347693]]
        '''
    compAvg = np.average(compUtilList)
    # Locate closest sample point for uniform and rudimentary to compAvg
    minListLen = np.min([len(i) for i in unifUtilList])
    unifUtilArr = np.array([i[:minListLen] for i in unifUtilList])
    unifAvgArr = np.average(unifUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(unifAvgArr.tolist()) if val > compAvg)
    unifSampSaved = round((compAvg - unifAvgArr[kInd - 1]) / (unifAvgArr[kInd] - unifAvgArr[kInd - 1]) * testInt) + (
            kInd - 1) * testInt
    print(unifSampSaved)
    '''29-MAR: 25 saved'''
    # Rudimentary
    minListLen = np.min([len(i) for i in origUtilList])
    origUtilArr = np.array([i[:minListLen] for i in origUtilList])
    origAvgArr = np.average(origUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(origAvgArr.tolist()) if val > compAvg)
    origSampSaved = round(
        (compAvg - origAvgArr[kInd - 1]) / (origAvgArr[kInd] - origAvgArr[kInd - 1]) * testInt * 3) + (
                            kInd - 1) * testInt * 3
    print(origSampSaved)
    '''29-MAR: 270 saved'''

    # Do again for different sample budget
    sampBudget = 90
    allocArr = forwardAllocateWithBudget(avgUtilMat, int(sampBudget / testInt))
    designArr = allocArr / np.sum(allocArr, axis=0)
    # Get utility for this allocation at the sample budget
    utilDict.update({'method': 'weightsNodeDraw4linear'})
    compUtilList, unifUtilList, origUtilList = [], [], []
    numReps = 5
    for rep in range(numReps):
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New Bayes draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        print('Generating loss matrix...')
        tempLossMat = lf.lossMatSetBayesDraws(dictTemp['postSamples'], lossDict.copy(), lossDict['bayesDraws'])
        tempLossDict = lossDict.copy()
        tempLossDict.update({'lossMat': tempLossMat})
        newBayesDraws, newLossMat = lf.add_cand_neighbors(tempLossDict.copy(), CSdict3['postSamples'],
                                                          dictTemp['postSamples'])
        tempLossDict.update({'bayesDraws': newBayesDraws, 'lossMat': newLossMat})
        baseLoss = (np.sum(newLossMat, axis=1) / newLossMat.shape[1]).min()
        # Get a new set of data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        currCompUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[designArr],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Heuristic utility: ' + str(currCompUtil))
        compUtilList.append(currCompUtil)
        '''31-MAR
        compUtilList = [0.6382285807799741, 0.605453522348244, 0.5922968790553096, 0.6645225820278204, 0.6663120137844762]
        '''
        # Find the equivalent uniform allocation
        currUnifUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[unifDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Uniform utility: ' + str(currUnifUtil))
        unifUtilList.append([currUnifUtil])
        unifAdd, contUnif, unifCount = 0, False, 0
        if currUnifUtil < currCompUtil:
            contUnif = True
        while contUnif:
            unifAdd += testInt
            print('Adding ' + str(unifAdd) + ' for uniform')
            currUnifUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[unifDes], numtests=sampBudget + unifAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currUnifUtil))
            unifUtilList[rep].append(currUnifUtil)
            if currUnifUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if unifCount < 3:
                    unifCount += 1
                else:
                    contUnif = False
        '''31-MAR
        unifUtilList = [[0.5631271417334545, 0.5994166813757857, 0.6378429395741949, 0.6835979393553409, 0.7142385316083022, 0.7411487822615221, 0.7801856109691689], [0.5412083518513109, 0.5802342879410776, 0.6161760605946305, 0.6524671498708905, 0.6871180403003354, 0.7079930805456969], [0.5211186983137481, 0.5511774227683426, 0.592568678362734, 0.6269942765438241, 0.6669278414846929, 0.699377437591667], [0.5809162235234466, 0.6193576692846765, 0.6599614279877377, 0.7091574392528006, 0.745832588960953, 0.7778515624054068, 0.8067433428017359], [0.585207096221176, 0.615997158919217, 0.6599667654492407, 0.6939685454768338, 0.7320551895449086, 0.7670579333383305, 0.7926379070056919]]
        '''
        # Find the equivalent rudimentary allocation
        currOrigUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[origDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Rudimentary utility: ' + str(currOrigUtil))
        origUtilList.append([currOrigUtil])
        origAdd, contOrig, origCount = 0, False, 0
        if currOrigUtil < currCompUtil:
            contOrig = True
        while contOrig:
            origAdd += testInt * 3
            print('Adding ' + str(origAdd) + ' for rudimentary')
            currOrigUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[origDes], numtests=sampBudget + origAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currOrigUtil))
            origUtilList[rep].append(currOrigUtil)
            if currOrigUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if origCount < 3:
                    origCount += 1
                else:
                    contOrig = False
        '''31-MAR
        origUtilList = [[0.23049055022113096, 0.2895526438678431, 0.33963460198011886, 0.4000550366197806, 0.4715083799888675, 0.5226152366787868, 0.5879893331034189, 0.6527838792744243, 0.7028075425661329, 0.7762403762992265, 0.8619074792828707], [0.20737143051424356, 0.26668131590771793, 0.32725337174036095, 0.3852652433681998, 0.4391488829825305, 0.5091551506500078, 0.5690837937907518, 0.631596862841127, 0.6944276330070687, 0.764411869566473, 0.8313842268104557], [0.19115607338134888, 0.24817689151821476, 0.302773273965816, 0.3576906746750663, 0.4254981469883168, 0.480833020354662, 0.5389262353897251, 0.5981645575904229, 0.6655694924536766, 0.7274475766066719, 0.8105325492105528], [0.2527203039076764, 0.3168592520560902, 0.37901983528764793, 0.439648650130231, 0.5059541272842392, 0.5698913498132163, 0.63182957650938, 0.6965973010752111, 0.7511492299092892, 0.815602897471333, 0.9123778469463399], [0.2352201964549674, 0.2989608001554642, 0.3584660622368223, 0.42148592408625873, 0.4797782391377976, 0.548082645766665, 0.604773072906827, 0.6727735577340432, 0.7352060089452501, 0.8001844038347796, 0.8727733098137742]]
        '''
    compAvg = np.average(compUtilList)
    # Locate closest sample point for uniform and rudimentary to compAvg
    minListLen = np.min([len(i) for i in unifUtilList])
    unifUtilArr = np.array([i[:minListLen] for i in unifUtilList])
    unifAvgArr = np.average(unifUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(unifAvgArr.tolist()) if val > compAvg)
    unifSampSaved = round((compAvg - unifAvgArr[kInd - 1]) / (unifAvgArr[kInd] - unifAvgArr[kInd - 1]) * testInt) + (
            kInd - 1) * testInt
    print(unifSampSaved)
    '''31-MAR: 20 saved'''
    # Rudimentary
    minListLen = np.min([len(i) for i in origUtilList])
    origUtilArr = np.array([i[:minListLen] for i in origUtilList])
    origAvgArr = np.average(origUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(origAvgArr.tolist()) if val > compAvg)
    origSampSaved = round(
        (compAvg - origAvgArr[kInd - 1]) / (origAvgArr[kInd] - origAvgArr[kInd - 1]) * testInt * 3) + (
                            kInd - 1) * testInt * 3
    print(origSampSaved)
    '''31-MAR: 202 saved'''

    ##############################################
    ##############################################
    # Compare marginal utiliies at tested nodes when using different
    # sourcing matrices for
    ##############################################
    ##############################################
    utilMatList1 = [np.array([[0., 0.02451592, 0.04855758, 0.07261078, 0.09142814,
                               0.11231712, 0.12268141, 0.14118758, 0.15649826, 0.17258595,
                               0.18135153],
                              [0., 0.0655778, 0.11702967, 0.15557825, 0.19120753,
                               0.21937603, 0.2460934, 0.26942906, 0.28777263, 0.30576801,
                               0.32764214],
                              [0., 0.01395371, 0.02843585, 0.04777044, 0.06274787,
                               0.07774682, 0.09496539, 0.10862246, 0.12546814, 0.13929499,
                               0.15431139],
                              [0., 0.00521008, 0.01313866, 0.02074631, 0.02795878,
                               0.03604236, 0.0420482, 0.04601286, 0.05039855, 0.0567085,
                               0.06208824],
                              [0., 0.14667013, 0.18892597, 0.21704129, 0.23478878,
                               0.24889874, 0.25914828, 0.27279244, 0.28066991, 0.28929819,
                               0.29815678],
                              [0., 0.05107067, 0.08162828, 0.10255872, 0.1165097,
                               0.12920727, 0.14268343, 0.15585291, 0.16525668, 0.17840945,
                               0.18266768],
                              [0., 0.09346927, 0.13843889, 0.16721363, 0.19044481,
                               0.21103031, 0.23259784, 0.24667569, 0.26356087, 0.27718308,
                               0.29060785],
                              [0., 0.12100444, 0.16388659, 0.19173441, 0.21137973,
                               0.22787451, 0.24012738, 0.25692266, 0.26379497, 0.27543485,
                               0.2852622]]),
                    np.array([[0., 0.00676249, 0.02308257, 0.039151, 0.05967397,
                               0.07769754, 0.09534459, 0.1112172, 0.12503183, 0.14360623,
                               0.15475692],
                              [0., 0.0346137, 0.08219858, 0.12369608, 0.15473904,
                               0.18892824, 0.21821453, 0.24186506, 0.26576264, 0.28120922,
                               0.30322893],
                              [0., 0.0018109, 0.01021703, 0.02174241, 0.03680351,
                               0.04552145, 0.06302509, 0.07815022, 0.08852497, 0.10427657,
                               0.12305748],
                              [0., 0.00370287, 0.00679118, 0.0111217, 0.01767454,
                               0.01913819, 0.0279442, 0.03142798, 0.03622548, 0.04361832,
                               0.04688333],
                              [0., 0.10768323, 0.14481648, 0.16980071, 0.18989393,
                               0.20543071, 0.22083033, 0.23767198, 0.25202475, 0.26020115,
                               0.27477491],
                              [0., 0.04439424, 0.07675788, 0.10029112, 0.11625655,
                               0.13113607, 0.1468082, 0.15526558, 0.16553691, 0.17438951,
                               0.18389694],
                              [0., 0.04406365, 0.08873329, 0.12490267, 0.15129765,
                               0.17359946, 0.19694854, 0.21493065, 0.2326752, 0.2545836,
                               0.27274015],
                              [0., 0.0383171, 0.06461345, 0.09241397, 0.11379849,
                               0.13079087, 0.14576379, 0.16214816, 0.17295371, 0.18549867,
                               0.19890409]]),
                    np.array([[0., 0.05749636, 0.09608722, 0.13029326, 0.15485669,
                               0.17692768, 0.19296016, 0.21262851, 0.22886686, 0.24418249,
                               0.26050214],
                              [0., 0.1257779, 0.19222448, 0.2395383, 0.27752983,
                               0.31018128, 0.33962961, 0.36208096, 0.38185752, 0.40429385,
                               0.42102526],
                              [0., 0.02842965, 0.05681675, 0.08032545, 0.10134849,
                               0.11700292, 0.13632777, 0.15409641, 0.16597128, 0.18501697,
                               0.19663819],
                              [0., 0.02720735, 0.0449396, 0.05798867, 0.07038999,
                               0.07948648, 0.08803197, 0.09471818, 0.10174703, 0.10669567,
                               0.11344435],
                              [0., 0.1252845, 0.17514875, 0.21021125, 0.23329018,
                               0.25467111, 0.27114674, 0.2869127, 0.3003015, 0.31193395,
                               0.32347585],
                              [0., 0.10857354, 0.15100698, 0.17627265, 0.19548869,
                               0.21145851, 0.22671338, 0.23555298, 0.2477294, 0.25663849,
                               0.26664315],
                              [0., 0.17068713, 0.22417778, 0.26087938, 0.28091857,
                               0.30008478, 0.32011375, 0.33597012, 0.34834607, 0.36435549,
                               0.3779873],
                              [0., 0.11734849, 0.16431268, 0.19521884, 0.22497543,
                               0.24260908, 0.25851014, 0.2811719, 0.29496294, 0.30423754,
                               0.31838349]]),
                    np.array([[0., 0.01345173, 0.03136005, 0.05660755, 0.07859248,
                               0.09639092, 0.11295868, 0.13071614, 0.14827999, 0.16534654,
                               0.18174914],
                              [0., 0.07221144, 0.13878848, 0.18017535, 0.21969484,
                               0.24911042, 0.27531689, 0.30023162, 0.31930139, 0.33937108,
                               0.36015899],
                              [0., 0.008002, 0.02708274, 0.04394301, 0.06352998,
                               0.08131683, 0.0980254, 0.11339686, 0.13039917, 0.14622489,
                               0.15684982],
                              [0., 0.00349879, 0.01013631, 0.01498932, 0.02047488,
                               0.02619417, 0.03191725, 0.03569684, 0.04260874, 0.04852905,
                               0.05173094],
                              [0., 0.059602, 0.10502625, 0.13563956, 0.16142992,
                               0.18021392, 0.20175985, 0.21506542, 0.23033948, 0.24530919,
                               0.25356911],
                              [0., 0.08883953, 0.12631464, 0.14627189, 0.16541075,
                               0.17677951, 0.18724289, 0.19964313, 0.20627858, 0.21677837,
                               0.22701359],
                              [0., 0.13416521, 0.17427311, 0.20280975, 0.22450047,
                               0.24442846, 0.25841944, 0.27636562, 0.29124343, 0.30653524,
                               0.31939327],
                              [0., 0.06871341, 0.09537234, 0.11617885, 0.13536047,
                               0.1478486, 0.16251259, 0.17380859, 0.18937296, 0.1976615,
                               0.21190252]]),
                    np.array([[0.00000000e+00, 4.23145488e-03, 1.51264347e-02,
                               2.45054095e-02, 4.29940701e-02, 5.08498899e-02,
                               6.71761867e-02, 7.46196354e-02, 9.01195534e-02,
                               1.01857410e-01, 1.14405441e-01],
                              [0.00000000e+00, 4.00727840e-02, 8.81030381e-02,
                               1.22068004e-01, 1.49450701e-01, 1.77534635e-01,
                               2.01722650e-01, 2.23543687e-01, 2.39366952e-01,
                               2.56424347e-01, 2.71441093e-01],
                              [0.00000000e+00, 2.50684148e-03, 7.01300043e-03,
                               1.18518074e-02, 2.24697149e-02, 2.67489313e-02,
                               3.91415789e-02, 4.64860059e-02, 5.67779341e-02,
                               6.50942739e-02, 7.81336044e-02],
                              [0.00000000e+00, -2.15147090e-04, 1.23774415e-04,
                               2.07058389e-03, 1.30248355e-03, 4.54540076e-03,
                               4.34345737e-03, 9.34790905e-03, 1.06808691e-02,
                               1.65392323e-02, 1.90601751e-02],
                              [0.00000000e+00, 4.95345578e-02, 8.04655133e-02,
                               1.01774369e-01, 1.21416510e-01, 1.36780849e-01,
                               1.50481639e-01, 1.64716040e-01, 1.75512424e-01,
                               1.83859883e-01, 1.94769153e-01],
                              [0.00000000e+00, 3.07521460e-02, 5.31325418e-02,
                               7.57079463e-02, 9.15199867e-02, 1.02830280e-01,
                               1.16579541e-01, 1.22976742e-01, 1.37384085e-01,
                               1.38451776e-01, 1.50954030e-01],
                              [0.00000000e+00, 4.46876651e-02, 7.27381704e-02,
                               9.77754782e-02, 1.15578898e-01, 1.36566071e-01,
                               1.53768888e-01, 1.70996834e-01, 1.84437278e-01,
                               1.99940675e-01, 2.11191984e-01],
                              [0.00000000e+00, 6.49792297e-02, 9.83533193e-02,
                               1.20653771e-01, 1.36898958e-01, 1.49472180e-01,
                               1.60407795e-01, 1.67185821e-01, 1.76185866e-01,
                               1.85262966e-01, 1.93078103e-01]]),
                    np.array([[0., 0.0346768, 0.06621606, 0.09164403, 0.1145677,
                               0.13315516, 0.15287402, 0.17231226, 0.19186275, 0.20930308,
                               0.21777717],
                              [0., 0.07794463, 0.14559928, 0.19967276, 0.24176276,
                               0.27980506, 0.30200435, 0.33175581, 0.34953254, 0.37111317,
                               0.38905085],
                              [0., 0.02434758, 0.04518998, 0.06380172, 0.08224753,
                               0.10356801, 0.11630203, 0.13095722, 0.15058368, 0.16073175,
                               0.16907954],
                              [0., 0.01114391, 0.02227675, 0.03172902, 0.04188907,
                               0.0491231, 0.05449954, 0.06392452, 0.06898837, 0.07386585,
                               0.07757122],
                              [0., 0.13939531, 0.17765902, 0.20074625, 0.22025803,
                               0.23289398, 0.24869296, 0.25825671, 0.2746029, 0.28783204,
                               0.29470339],
                              [0., 0.09411192, 0.13132608, 0.15403398, 0.17097161,
                               0.18285429, 0.19906, 0.205608, 0.21745646, 0.22326376,
                               0.23228131],
                              [0., 0.1043377, 0.15380167, 0.18570403, 0.2133691,
                               0.23775624, 0.26010548, 0.27929454, 0.29342104, 0.31421402,
                               0.32667931],
                              [0., 0.05808167, 0.10520707, 0.14058244, 0.16649049,
                               0.18313798, 0.20354156, 0.22284478, 0.23510841, 0.24863756,
                               0.26258778]])
                    ]
    utilMatList2 = [np.array([[0., 0.02444741, 0.04617417, 0.07172455, 0.08809567,
                               0.10640526, 0.12366133, 0.13573863, 0.15604136, 0.17068028,
                               0.18108176],
                              [0., 0.04834982, 0.0937102, 0.13237818, 0.16873718,
                               0.19415305, 0.21981798, 0.24330678, 0.26916736, 0.28552992,
                               0.30819564],
                              [0., 0.01205559, 0.0272295, 0.03830821, 0.05469283,
                               0.06533469, 0.08095158, 0.09126618, 0.10626761, 0.11855323,
                               0.13183308],
                              [0., 0.00889473, 0.01812639, 0.02379298, 0.03212082,
                               0.03628557, 0.03975221, 0.04636525, 0.05213653, 0.05516324,
                               0.06090449],
                              [0., 0.06774564, 0.09412483, 0.11245624, 0.1295772,
                               0.13871881, 0.14940352, 0.16106266, 0.16738117, 0.17591085,
                               0.1827334],
                              [0., 0.09450995, 0.12544353, 0.14686126, 0.15972911,
                               0.17444362, 0.18053802, 0.19119856, 0.19830664, 0.20612721,
                               0.21666246],
                              [0., 0.10812779, 0.13866387, 0.16121653, 0.17872359,
                               0.18995057, 0.20162464, 0.2143453, 0.22490877, 0.23183825,
                               0.23978482],
                              [0., 0.0225108, 0.05364471, 0.07401074, 0.08804033,
                               0.10184499, 0.11381971, 0.12134715, 0.12895082, 0.1392264,
                               0.14832965]]),
                    np.array([[0., 0.01061772, 0.03201993, 0.0476978, 0.06522253,
                               0.08445483, 0.10002001, 0.1130361, 0.1314298, 0.14692459,
                               0.15758143],
                              [0., 0.02270379, 0.05805478, 0.09679276, 0.12803155,
                               0.15691295, 0.18389047, 0.20513688, 0.22898513, 0.25110721,
                               0.26810876],
                              [0., 0.01228162, 0.02790018, 0.04423467, 0.05821603,
                               0.06911803, 0.08560955, 0.09989748, 0.11257363, 0.12876694,
                               0.13481379],
                              [0., 0.00035913, 0.00169294, 0.00579694, 0.00880823,
                               0.01714156, 0.02000026, 0.02557278, 0.03122706, 0.03301323,
                               0.03981037],
                              [0., 0.05824016, 0.08573851, 0.10101961, 0.11410206,
                               0.12444768, 0.13687999, 0.14131011, 0.15409859, 0.15994816,
                               0.1647348],
                              [0., 0.03672482, 0.05983045, 0.07725387, 0.09468524,
                               0.10663863, 0.11931555, 0.1288919, 0.13691828, 0.14544028,
                               0.15448448],
                              [0., 0.05017085, 0.08021415, 0.10054485, 0.11795222,
                               0.13324094, 0.14550968, 0.15740122, 0.16794647, 0.17424362,
                               0.18580221],
                              [0., 0.08377163, 0.11383486, 0.13477952, 0.14504161,
                               0.1617759, 0.16763926, 0.17146587, 0.18497091, 0.18961659,
                               0.19583548]]),
                    np.array([[0., 0.05289022, 0.08776998, 0.12225116, 0.14337044,
                               0.16746483, 0.18300634, 0.20319633, 0.22121103, 0.23645388,
                               0.25288762],
                              [0., 0.09949844, 0.16303964, 0.21023837, 0.25285218,
                               0.28553042, 0.31828874, 0.33517058, 0.36180306, 0.38057189,
                               0.40137325],
                              [0., 0.02774304, 0.05555077, 0.08177219, 0.10254351,
                               0.12142655, 0.14443328, 0.16473343, 0.18135736, 0.19966655,
                               0.21823272],
                              [0., 0.0116689, 0.02846629, 0.03894043, 0.05353948,
                               0.06113148, 0.07123619, 0.0769341, 0.08715056, 0.09303578,
                               0.09951172],
                              [0., 0.11733075, 0.15430028, 0.17623312, 0.18990536,
                               0.20403717, 0.2156951, 0.2284347, 0.23739609, 0.24569393,
                               0.25454135],
                              [0., 0.08405904, 0.12345312, 0.14851909, 0.16747762,
                               0.18197873, 0.19359453, 0.20492001, 0.21534388, 0.22361092,
                               0.23399031],
                              [0., 0.09027289, 0.13715567, 0.16395737, 0.18549652,
                               0.20305894, 0.21722925, 0.22930705, 0.2432413, 0.25674125,
                               0.26186968],
                              [0., 0.07602299, 0.10753734, 0.131441, 0.14890382,
                               0.16484792, 0.17376019, 0.18356386, 0.19582632, 0.2041181,
                               0.21100653]]),
                    np.array([[0., 0.04280313, 0.07743959, 0.10556424, 0.12466443,
                               0.15066002, 0.16781309, 0.18275166, 0.20490615, 0.22078829,
                               0.23297169],
                              [0., 0.09521162, 0.1569858, 0.20584074, 0.24333427,
                               0.27283366, 0.29946565, 0.32765417, 0.35167152, 0.37209175,
                               0.38894874],
                              [0., 0.02686715, 0.05106474, 0.07126627, 0.08877093,
                               0.1031878, 0.11952822, 0.13340527, 0.14928755, 0.16425047,
                               0.18184323],
                              [0., 0.02426568, 0.04074958, 0.05608495, 0.06565289,
                               0.07453899, 0.08215042, 0.08985788, 0.09804557, 0.10417867,
                               0.107827],
                              [0., 0.12188501, 0.16242993, 0.1906887, 0.20550559,
                               0.21944671, 0.23029986, 0.24028658, 0.25159644, 0.25906787,
                               0.26717226],
                              [0., 0.12862406, 0.17109328, 0.19499282, 0.21154867,
                               0.22271788, 0.24070267, 0.24711458, 0.25963004, 0.26591217,
                               0.27245365],
                              [0., 0.14447852, 0.18800221, 0.21547517, 0.23340096,
                               0.24737714, 0.26132212, 0.27339058, 0.28408912, 0.29658744,
                               0.30412029],
                              [0., 0.08420287, 0.12093209, 0.14498087, 0.162028,
                               0.17444314, 0.18940581, 0.19729391, 0.20525447, 0.21334051,
                               0.22009713]]),
                    np.array([[0., 0.03677073, 0.06687004, 0.095653, 0.12444647,
                               0.14561554, 0.1628917, 0.18565958, 0.20149133, 0.22313254,
                               0.2322078],
                              [0., 0.0921443, 0.15064145, 0.19596488, 0.24356864,
                               0.27816161, 0.31071725, 0.333375, 0.36150312, 0.37992713,
                               0.40054777],
                              [0., 0.03158609, 0.06114475, 0.08476411, 0.10382951,
                               0.12495941, 0.14603506, 0.16075655, 0.17923553, 0.20015804,
                               0.21033831],
                              [0., 0.01119459, 0.02237529, 0.03206565, 0.04215204,
                               0.04978128, 0.05579398, 0.06215214, 0.0702405, 0.07638155,
                               0.08086721],
                              [0., 0.11111867, 0.1549965, 0.17697585, 0.1972016,
                               0.20931262, 0.21969137, 0.22900297, 0.24178031, 0.24601579,
                               0.25404951],
                              [0., 0.11530631, 0.15794428, 0.18727605, 0.20385003,
                               0.21417796, 0.22583792, 0.23883884, 0.24886706, 0.25156618,
                               0.25957483],
                              [0., 0.14557227, 0.19756024, 0.22939661, 0.24970096,
                               0.26761658, 0.28378273, 0.29626806, 0.30821247, 0.32048759,
                               0.32567444],
                              [0., 0.07029971, 0.10043551, 0.11912466, 0.13398989,
                               0.14420831, 0.15682328, 0.16601601, 0.17448884, 0.18324531,
                               0.19043351]]),
                    np.array([[0., 0.02915052, 0.0568921, 0.08468161, 0.10841281,
                               0.1256468, 0.15082809, 0.16613433, 0.18881289, 0.20107538,
                               0.22283628],
                              [0., 0.06722355, 0.12182064, 0.16474724, 0.20309623,
                               0.23478364, 0.26400862, 0.28802007, 0.3070428, 0.33282133,
                               0.34965786],
                              [0., 0.02421449, 0.04588543, 0.06750901, 0.08666779,
                               0.10563033, 0.11836118, 0.13406416, 0.15403105, 0.1707277,
                               0.18143652],
                              [0., 0.01659786, 0.02853722, 0.03972463, 0.04595229,
                               0.05307529, 0.0632196, 0.0665243, 0.07286573, 0.07985776,
                               0.08586452],
                              [0., 0.16011224, 0.20209676, 0.22314093, 0.23641936,
                               0.25186789, 0.25763824, 0.26802574, 0.27553782, 0.27996404,
                               0.28841742],
                              [0., 0.15776267, 0.19360888, 0.2209689, 0.23759046,
                               0.24785269, 0.25982594, 0.26423594, 0.2785833, 0.28487747,
                               0.28995016],
                              [0., 0.14362273, 0.1913843, 0.21811577, 0.23896475,
                               0.25286501, 0.26819642, 0.28062925, 0.29024799, 0.30003273,
                               0.31225241],
                              [0., 0.06856841, 0.10010141, 0.11958507, 0.13581957,
                               0.14551666, 0.15676986, 0.16281806, 0.17302035, 0.18118008,
                               0.18749527]]),
                    np.array([[0., 0.01685809, 0.03712153, 0.06386612, 0.08729323,
                               0.10173939, 0.12468433, 0.14379201, 0.155036, 0.17306744,
                               0.1890389],
                              [0., 0.06628199, 0.11629691, 0.15890264, 0.19527346,
                               0.22415608, 0.25060362, 0.27294009, 0.28904085, 0.30917314,
                               0.33095715],
                              [0., 0.01241213, 0.02748692, 0.03924473, 0.05663609,
                               0.0730707, 0.09158386, 0.10612843, 0.11712627, 0.12982069,
                               0.13590827],
                              [0., 0.0041729, 0.00872908, 0.01866805, 0.02462395,
                               0.03099282, 0.03625286, 0.04495835, 0.04780783, 0.05156178,
                               0.05706089],
                              [0., 0.11300693, 0.1453673, 0.16376946, 0.17541776,
                               0.18375239, 0.19277467, 0.19876675, 0.20420916, 0.21007576,
                               0.21859571],
                              [0., 0.05616503, 0.08518666, 0.10669408, 0.12662187,
                               0.13739913, 0.1503795, 0.15945393, 0.1699209, 0.17788696,
                               0.18649542],
                              [0., 0.04563272, 0.07904905, 0.10405482, 0.12060336,
                               0.13658139, 0.14915913, 0.16392047, 0.17487578, 0.18559586,
                               0.19488231],
                              [0., 0.07310888, 0.10032254, 0.11892637, 0.13223117,
                               0.14410172, 0.15266497, 0.15959014, 0.17010732, 0.17246216,
                               0.18312472]]),
                    np.array([[0., 0.02333424, 0.038126, 0.05818243, 0.07652159,
                               0.0930885, 0.1091782, 0.12290162, 0.13782104, 0.14904138,
                               0.16018488],
                              [0., 0.03753614, 0.0740499, 0.10038972, 0.13316401,
                               0.15476415, 0.17949055, 0.19536641, 0.21502607, 0.23157766,
                               0.2469962],
                              [0., 0.00895256, 0.02081164, 0.03044111, 0.04464267,
                               0.0561284, 0.06999362, 0.08211742, 0.0950575, 0.10494854,
                               0.11417126],
                              [0., 0.00823575, 0.01855312, 0.02363607, 0.0323064,
                               0.0380835, 0.04394861, 0.04985392, 0.05145138, 0.05953345,
                               0.06464969],
                              [0., 0.11527287, 0.1568247, 0.18019577, 0.19212014,
                               0.20592982, 0.21249921, 0.21854016, 0.22560234, 0.23050269,
                               0.23262888],
                              [0., 0.05978155, 0.08810741, 0.10831188, 0.1261699,
                               0.13524031, 0.14512738, 0.15539314, 0.16376151, 0.17180738,
                               0.17320792],
                              [0., 0.04695933, 0.07678728, 0.09652508, 0.11278355,
                               0.12640277, 0.13917011, 0.14785176, 0.15689272, 0.16630099,
                               0.17723344],
                              [0., 0.0395691, 0.06081441, 0.07881922, 0.09178753,
                               0.10281127, 0.11002848, 0.11912296, 0.12548132, 0.12947523,
                               0.13695145]]),
                    np.array([[0., 0.03537446, 0.06434736, 0.0856373, 0.11161541,
                               0.13210294, 0.14932332, 0.16405937, 0.17799343, 0.19429886,
                               0.20858928],
                              [0., 0.07583115, 0.12618876, 0.16624427, 0.19729915,
                               0.22647003, 0.2514849, 0.2716841, 0.29316011, 0.31267927,
                               0.33411131],
                              [0., 0.01361828, 0.02754478, 0.04397554, 0.05980463,
                               0.07744601, 0.09610903, 0.10819553, 0.1236232, 0.13642089,
                               0.15309482],
                              [0., 0.00632155, 0.01005745, 0.01471325, 0.02291924,
                               0.03114394, 0.0372675, 0.04043668, 0.05063683, 0.05583,
                               0.06027561],
                              [0., 0.1503635, 0.18959642, 0.21300263, 0.22853748,
                               0.24320318, 0.25205858, 0.26228675, 0.26869937, 0.27525275,
                               0.2836008],
                              [0., 0.08250832, 0.11869987, 0.14112102, 0.15912721,
                               0.17229714, 0.18176564, 0.19763742, 0.20712643, 0.21480984,
                               0.22325277],
                              [0., 0.09788981, 0.14107614, 0.16551395, 0.18513062,
                               0.1997572, 0.21383891, 0.22361878, 0.23518644, 0.24617805,
                               0.25604735],
                              [0., 0.0674143, 0.10782412, 0.13079347, 0.14544611,
                               0.16067363, 0.17133179, 0.1843107, 0.19235671, 0.20081196,
                               0.20608015]]),
                    np.array([[0., 0.05191602, 0.09051411, 0.11724189, 0.1393711,
                               0.15867739, 0.17891094, 0.19748465, 0.21374922, 0.22885509,
                               0.24299431],
                              [0., 0.10800944, 0.17665256, 0.2210505, 0.26385208,
                               0.29367264, 0.32463802, 0.34848621, 0.37509772, 0.39464624,
                               0.41316588],
                              [0., 0.03427224, 0.06171548, 0.07909405, 0.10659685,
                               0.12436567, 0.14538645, 0.16099835, 0.17953662, 0.19933655,
                               0.2152309],
                              [0., 0.02075514, 0.03354546, 0.04442889, 0.05490154,
                               0.06429095, 0.07580695, 0.08077498, 0.08569503, 0.09475651,
                               0.10079062],
                              [0., 0.13679852, 0.17078316, 0.19376287, 0.21390631,
                               0.22634267, 0.2390888, 0.24822102, 0.25819282, 0.26382268,
                               0.27277994],
                              [0., 0.10328089, 0.1448251, 0.16706774, 0.18566444,
                               0.19990438, 0.21393025, 0.22114451, 0.2332845, 0.24145801,
                               0.25118244],
                              [0., 0.17315293, 0.21209315, 0.2331145, 0.24829703,
                               0.26037815, 0.26945285, 0.28150528, 0.29312042, 0.30356294,
                               0.30862569],
                              [0., 0.11491338, 0.15670171, 0.18254758, 0.19958868,
                               0.21408513, 0.22369242, 0.23584977, 0.2449036, 0.25242339,
                               0.25671282]])]
    avgUtilMat1 = np.average(np.array(utilMatList1), axis=0)
    avgUtilMat2 = np.average(np.array(utilMatList2), axis=0)

    colors = cm.rainbow(np.linspace(0, 1., 4))
    for i in range(4):
        plt.plot(testArr, avgUtilMat1[i], color=colors[i], linewidth=2, alpha=0.2, label='TN ' + str(i))
        plt.plot(testArr, avgUtilMat2[i], color=colors[i], linewidth=0.5, alpha=1.)
    plt.ylim([0, 0.4])
    plt.legend()
    plt.title('Utility at tested nodes under different sourcing matrices for untested nodes')
    plt.show()
    plt.close()

    ##############################################
    ##############################################
    # Use different variance for untested nodes (TNvar = 4)
    ##############################################
    ##############################################
    # Go back to orginal sourcing matrix
    numBoot = 44  # 44 is average across each TN in original data set
    SNprobs = np.sum(CSdict3['N'], axis=0) / np.sum(CSdict3['N'])
    np.random.seed(33)
    Qvecs = np.random.multinomial(numBoot, SNprobs, size=numTN - 4) / numBoot
    CSdict3['Q'] = np.vstack((CSdict3['N'][:4] / np.sum(CSdict3['N'][:4], axis=1).reshape(4, 1), Qvecs))

    SNpriorMean = np.repeat(sps.logit(0.1), numSN)
    # Establish test nodes according to assessment by regulators
    # REMOVE LATER
    # ASHANTI: Moderate; BRONG AHAFO: Moderate; CENTRAL: Moderately High; EASTERN REGION: Moderately High
    # GREATER ACCRA: Moderately High; NORTHERN SECTOR: Moderate; VOLTA: Moderately High; WESTERN: Moderate
    TNpriorMean = sps.logit(np.array([0.1, 0.1, 0.15, 0.15, 0.15, 0.1, 0.15, 0.1]))
    TNvar, SNvar = 4., 4.
    CSdict3['prior'] = prior_normal_assort(np.concatenate((SNpriorMean, TNpriorMean)),
                                           np.diag(np.concatenate((np.repeat(SNvar, numSN), np.repeat(TNvar, numTN)))))

    sampBudget = 180
    unifDes = np.zeros(numTN) + 1 / numTN
    origDes = np.sum(rd3_N, axis=1) / np.sum(rd3_N)

    # Use original loss parameters
    paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                                  marketvec=np.ones(numTN + numSN), candneighnum=1000)

    numtargetdraws, numDataDraws = 5100, 5000

    # Find heuristic allocation first
    utilDict = {'method': 'weightsNodeDraw3linear'}

    numReps = 10
    utilMatList = []
    # set testMax to highest expected allocation for any one node
    testMax, testInt = 100, 10
    testArr = np.arange(0, testMax + 1, testInt)
    for rep in range(numReps):
        CSdict3 = methods.GeneratePostSamples(CSdict3)
        # Withdraw a subset of MCMC prior draws
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New loss draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        # Get new data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        # Get marginal utilities at each test node
        currMargUtilMat = GetMargUtilAtNodes(dictTemp.copy(), testMax, testInt, lossDict.copy(), utilDict.copy(),
                                             masterDraws=CSdict3['postSamples'], printUpdate=True)
        print(repr(currMargUtilMat))
        utilMatList.append(currMargUtilMat)
        for mat in utilMatList:
            for i in range(8):
                plt.plot(testArr, mat[i], linewidth=0.2)
        avgUtilMat = np.average(np.array(utilMatList), axis=0)
        for i in range(8):
            plt.plot(testArr, avgUtilMat[i], linewidth=2)
        plt.ylim([0, 0.4])
        # plt.title('Comprehensive utility for allocations via heuristic\nUntested nodes')
        plt.show()
        plt.close()
    '''13-APR run
    utilMatList = [np.array([[0.        , 0.02335924, 0.03861796, 0.04280477, 0.05339048,
        0.06127308, 0.06693074, 0.07969466, 0.08544896, 0.09493892,
        0.10727059],
       [0.        , 0.02815689, 0.05579091, 0.08276633, 0.10334095,
        0.1233647 , 0.14180044, 0.16193034, 0.17831966, 0.19336932,
        0.21288853],
       [0.        , 0.00927649, 0.02051898, 0.02946207, 0.03139826,
        0.04320537, 0.04353494, 0.05790448, 0.06469835, 0.07783886,
        0.08444377],
       [0.        , 0.00825998, 0.0138634 , 0.01990932, 0.02255103,
        0.02990845, 0.02990995, 0.03042242, 0.03113957, 0.03613893,
        0.03743424],
       [0.        , 0.07877112, 0.11753183, 0.1457735 , 0.16270069,
        0.18089024, 0.19340589, 0.20941242, 0.2225792 , 0.23446819,
        0.24392023],
       [0.        , 0.09446803, 0.12218001, 0.14581639, 0.16665164,
        0.17648817, 0.1846348 , 0.19647273, 0.20564934, 0.21117153,
        0.22166313],
       [0.        , 0.08752024, 0.12916104, 0.15214483, 0.17458407,
        0.18546354, 0.20029281, 0.20953506, 0.22108129, 0.23017263,
        0.23969322],
       [0.        , 0.12147485, 0.1508586 , 0.16921808, 0.19018483,
        0.20871369, 0.22430447, 0.23642113, 0.25532338, 0.26752123,
        0.27830945]]),
        np.array([[0.        , 0.01313641, 0.03879579, 0.05780749, 0.07768714,
        0.09502801, 0.11757174, 0.13432393, 0.14955095, 0.16908302,
        0.17693678],
       [0.        , 0.03825391, 0.0870787 , 0.13234703, 0.16896417,
        0.201411  , 0.22476124, 0.25086174, 0.272942  , 0.29667388,
        0.3125742 ],
       [0.        , 0.00313873, 0.01109433, 0.02408647, 0.03610584,
        0.05046271, 0.0620441 , 0.07725854, 0.09032929, 0.10943644,
        0.12095651],
       [0.        , 0.0059182 , 0.01239423, 0.0189296 , 0.02554282,
        0.03103827, 0.03747601, 0.04183832, 0.04404031, 0.04974331,
        0.05245414],
       [0.        , 0.1968052 , 0.24827233, 0.28205878, 0.2964339 ,
        0.31413549, 0.32517856, 0.343812  , 0.35226114, 0.36111409,
        0.37124424],
       [0.        , 0.08609729, 0.13678752, 0.1605573 , 0.18355412,
        0.19750327, 0.21023251, 0.22382062, 0.23594022, 0.2441725 ,
        0.25581126],
       [0.        , 0.12886253, 0.18001636, 0.21505893, 0.23841141,
        0.25555011, 0.2706609 , 0.28348255, 0.29641934, 0.31169007,
        0.32012373],
       [0.        , 0.12929826, 0.18449621, 0.21881006, 0.24492898,
        0.26784707, 0.29127322, 0.30473657, 0.32488016, 0.34292929,
        0.35790622]]),
        np.array([[0.        , 0.0724605 , 0.12321651, 0.15765861, 0.18945108,
        0.2129031 , 0.23912336, 0.25283583, 0.27146731, 0.29040547,
        0.31094233],
       [0.        , 0.1420571 , 0.20602456, 0.25341205, 0.29060661,
        0.32095715, 0.3497573 , 0.36859205, 0.39513923, 0.41424388,
        0.43323954],
       [0.        , 0.04693731, 0.08274115, 0.11039294, 0.14148206,
        0.16435348, 0.18284637, 0.20770426, 0.21985381, 0.23989875,
        0.25957077],
       [0.        , 0.0274807 , 0.045725  , 0.06116423, 0.07482432,
        0.08914297, 0.09462181, 0.10731935, 0.11374316, 0.12129406,
        0.12509106],
       [0.        , 0.26923745, 0.32901556, 0.36386646, 0.38698114,
        0.41088984, 0.42469666, 0.43835604, 0.45058461, 0.46715038,
        0.47492912],
       [0.        , 0.19432089, 0.25089287, 0.28148201, 0.3078486 ,
        0.32606932, 0.3402429 , 0.35328745, 0.36645158, 0.37670504,
        0.38900418],
       [0.        , 0.23502171, 0.29412122, 0.33113948, 0.35571361,
        0.37929503, 0.39515445, 0.40916021, 0.42209867, 0.43127236,
        0.44610332],
       [0.        , 0.25797309, 0.31410005, 0.35099135, 0.377541  ,
        0.40054145, 0.42830989, 0.44272018, 0.46809086, 0.48615302,
        0.50364982]]),
        np.array([[ 0.        ,  0.00051282, -0.00333193, -0.00128179, -0.00042802,
         0.00653205,  0.01388496,  0.02083696,  0.0255557 ,  0.03258295,
         0.04276305],
       [ 0.        ,  0.01551811,  0.032674  ,  0.05244179,  0.0757514 ,
         0.08651373,  0.09773892,  0.11775339,  0.12887556,  0.14785346,
         0.16113826],
       [ 0.        ,  0.00062617,  0.00045008,  0.00245989,  0.00588881,
         0.0048134 ,  0.01183442,  0.01718119,  0.02124045,  0.02996608,
         0.03237112],
       [ 0.        , -0.00023562, -0.00051108,  0.00306422,  0.00102083,
         0.00257224,  0.00347028,  0.00653368,  0.0085236 ,  0.00964147,
         0.01265182],
       [ 0.        ,  0.08850035,  0.11866416,  0.13370459,  0.14501016,
         0.1582489 ,  0.16491038,  0.17491739,  0.1849575 ,  0.19166118,
         0.20943711],
       [ 0.        ,  0.05229265,  0.06974241,  0.08777222,  0.09655546,
         0.10550518,  0.11298178,  0.12254045,  0.13248032,  0.13906877,
         0.14521765],
       [ 0.        ,  0.04291219,  0.06266118,  0.07502517,  0.08737665,
         0.09651529,  0.10663129,  0.114961  ,  0.12249835,  0.13026516,
         0.14078565],
       [ 0.        ,  0.06133131,  0.09216766,  0.10792486,  0.120146  ,
         0.13214806,  0.14697203,  0.15478819,  0.16756046,  0.18393852,
         0.19161231]]),
         np.array([[0.        , 0.03604698, 0.06124874, 0.07909296, 0.10593425,
        0.12410504, 0.14351776, 0.15981859, 0.17704147, 0.19421741,
        0.20717063],
       [0.        , 0.10398979, 0.16639846, 0.21571108, 0.25896983,
        0.29147228, 0.31806939, 0.34199244, 0.36444756, 0.3929467 ,
        0.40920814],
       [0.        , 0.02545722, 0.04229068, 0.05912463, 0.07932238,
        0.09096039, 0.11291457, 0.12584531, 0.1403978 , 0.15787267,
        0.17202508],
       [0.        , 0.01660264, 0.02445057, 0.03327082, 0.04178312,
        0.04575886, 0.05319729, 0.05944349, 0.0670189 , 0.07156327,
        0.07559263],
       [0.        , 0.2689693 , 0.32013567, 0.35519777, 0.36852812,
        0.39077497, 0.40042864, 0.41714303, 0.43330491, 0.44072049,
        0.45081501],
       [0.        , 0.19267052, 0.23829126, 0.26878741, 0.28718309,
        0.30246672, 0.31314981, 0.3253894 , 0.33299175, 0.34552505,
        0.35610432],
       [0.        , 0.2288398 , 0.29229292, 0.32554174, 0.35382566,
        0.37410864, 0.39132707, 0.40278577, 0.41265978, 0.42779487,
        0.43751455],
       [0.        , 0.17633857, 0.23519707, 0.27089781, 0.30341203,
        0.32952843, 0.35338109, 0.37336481, 0.39296577, 0.41650776,
        0.43237105]]),
        np.array([[0.        , 0.02784605, 0.05345839, 0.07886641, 0.10193742,
        0.12019679, 0.14123806, 0.15729607, 0.17419212, 0.19158907,
        0.20757892],
       [0.        , 0.07451378, 0.12386646, 0.16388943, 0.19836006,
        0.23106551, 0.25293106, 0.27448223, 0.301085  , 0.32014248,
        0.34421704],
       [0.        , 0.01993609, 0.03570052, 0.05289389, 0.07253586,
        0.09011384, 0.10071144, 0.11969461, 0.13275741, 0.15384272,
        0.16423415],
       [0.        , 0.00264344, 0.00871149, 0.01523333, 0.02069179,
        0.02633273, 0.03241627, 0.04033698, 0.04349895, 0.0499519 ,
        0.05403492],
       [0.        , 0.17521184, 0.22315544, 0.25304815, 0.27450426,
        0.28964217, 0.30971851, 0.32450682, 0.33970044, 0.35271104,
        0.36531411],
       [0.        , 0.11221549, 0.16258548, 0.19533784, 0.21691498,
        0.23890525, 0.25412464, 0.26716562, 0.28136881, 0.29056291,
        0.30040951],
       [0.        , 0.17254707, 0.23041003, 0.26184614, 0.28721201,
        0.30560261, 0.32034438, 0.33318716, 0.3466079 , 0.3575821 ,
        0.3661502 ],
       [0.        , 0.2241903 , 0.26553166, 0.29595327, 0.31838107,
        0.33828304, 0.35770887, 0.37614223, 0.39208601, 0.4053388 ,
        0.42010761]]),
        np.array([[0.        , 0.05322584, 0.09311057, 0.12680125, 0.15144792,
        0.17351804, 0.19582128, 0.21433814, 0.23287411, 0.25116803,
        0.26821642],
       [0.        , 0.09173617, 0.15156922, 0.19853064, 0.2393253 ,
        0.27395221, 0.30969448, 0.33626144, 0.35667048, 0.38218351,
        0.407124  ],
       [0.        , 0.03689837, 0.06731727, 0.09483826, 0.1171848 ,
        0.14306001, 0.15776438, 0.17862992, 0.19812776, 0.2143086 ,
        0.23417192],
       [0.        , 0.02490174, 0.04438694, 0.05837288, 0.07427645,
        0.08575395, 0.09145322, 0.10372036, 0.11191377, 0.11886475,
        0.12829947],
       [0.        , 0.2453042 , 0.29453758, 0.32271279, 0.34996765,
        0.36775924, 0.38759269, 0.4017696 , 0.4172603 , 0.43400815,
        0.4456867 ],
       [0.        , 0.18876643, 0.25203653, 0.28756442, 0.31022392,
        0.33478891, 0.35030482, 0.36415549, 0.37318131, 0.3860648 ,
        0.39836325],
       [0.        , 0.18428247, 0.23954835, 0.27369867, 0.29910796,
        0.3188296 , 0.33734832, 0.35239537, 0.36767112, 0.37986454,
        0.39228509],
       [0.        , 0.22889457, 0.28484479, 0.32360374, 0.35416369,
        0.37631059, 0.40353778, 0.42179668, 0.44313334, 0.46256045,
        0.48082342]]),
        np.array([[0.        , 0.03926217, 0.06105648, 0.08696859, 0.10646605,
        0.12371542, 0.14279779, 0.15889692, 0.17916405, 0.19647485,
        0.20685789],
       [0.        , 0.07631337, 0.12951012, 0.17509282, 0.20885753,
        0.24266373, 0.26532549, 0.29022159, 0.31382966, 0.33218426,
        0.35658628],
       [0.        , 0.03928058, 0.05994509, 0.07997601, 0.09890487,
        0.11583592, 0.12486724, 0.14043531, 0.15620746, 0.1740782 ,
        0.18496775],
       [0.        , 0.02773681, 0.03829393, 0.04675977, 0.0566586 ,
        0.06444664, 0.0703148 , 0.07797869, 0.08133403, 0.08780952,
        0.09322089],
       [0.        , 0.21191099, 0.26041086, 0.28736002, 0.30978824,
        0.32451789, 0.34155524, 0.35492304, 0.36751812, 0.37552758,
        0.39187233],
       [0.        , 0.181568  , 0.21714905, 0.2413642 , 0.25864144,
        0.27059148, 0.28283766, 0.29360551, 0.30060587, 0.31024283,
        0.32009064],
       [0.        , 0.20296487, 0.26297268, 0.30012591, 0.32628588,
        0.34766876, 0.36634537, 0.37966913, 0.39426264, 0.40140973,
        0.41390823],
       [0.        , 0.23246617, 0.28467182, 0.31638935, 0.34038828,
        0.36496301, 0.3801429 , 0.40525425, 0.4154783 , 0.43092652,
        0.44205349]]),
        np.array([[0.        , 0.01559575, 0.04102979, 0.06231824, 0.08150125,
        0.10013828, 0.11774539, 0.13412896, 0.15162169, 0.16181404,
        0.18205169],
       [0.        , 0.05608372, 0.1017004 , 0.13666913, 0.16880313,
        0.20132634, 0.22359562, 0.25124959, 0.27112698, 0.29221217,
        0.30749224],
       [0.        , 0.01380102, 0.03178813, 0.0440616 , 0.06121605,
        0.07418234, 0.09209776, 0.10771142, 0.121165  , 0.13536933,
        0.14720476],
       [0.        , 0.00245392, 0.00922994, 0.0164785 , 0.02192651,
        0.03106678, 0.03555291, 0.03942155, 0.04494583, 0.05142055,
        0.05832692],
       [0.        , 0.21764327, 0.26171415, 0.2866695 , 0.30538924,
        0.31623715, 0.32835561, 0.34292345, 0.35337831, 0.36473836,
        0.37582501],
       [0.        , 0.10145477, 0.14590169, 0.17781662, 0.19823773,
        0.21217532, 0.23116127, 0.24351041, 0.25304913, 0.26447217,
        0.27475707],
       [0.        , 0.13908624, 0.18047228, 0.2034137 , 0.22309882,
        0.23862318, 0.25060208, 0.26278474, 0.27427351, 0.28182271,
        0.29101621],
       [0.        , 0.10411697, 0.14479834, 0.17462902, 0.20043621,
        0.22263656, 0.24558851, 0.26702352, 0.28605839, 0.30236089,
        0.31803809]]),
        np.array([[0.        , 0.0434926 , 0.08047958, 0.10661438, 0.13543525,
        0.1587469 , 0.18232249, 0.19831337, 0.21774672, 0.23270063,
        0.2510882 ],
       [0.        , 0.09805197, 0.1572192 , 0.19533384, 0.22897455,
        0.25974203, 0.28889655, 0.31400649, 0.33693688, 0.35651703,
        0.37093279],
       [0.        , 0.03380573, 0.05724588, 0.07738889, 0.10034521,
        0.11692331, 0.13506226, 0.15386911, 0.17141128, 0.18550409,
        0.1992434 ],
       [0.        , 0.01598864, 0.02861248, 0.03684698, 0.04625967,
        0.05205383, 0.0565812 , 0.0640237 , 0.0716219 , 0.07239335,
        0.08429488],
       [0.        , 0.1802517 , 0.22856411, 0.25904428, 0.28496633,
        0.30996747, 0.32422326, 0.34098693, 0.35681305, 0.37367307,
        0.38616576],
       [0.        , 0.16454108, 0.20554301, 0.23188978, 0.25224649,
        0.26967566, 0.28419412, 0.29738684, 0.30628339, 0.32097637,
        0.32682249],
       [0.        , 0.16990144, 0.21950203, 0.25277342, 0.27675605,
        0.29715494, 0.31363672, 0.32861441, 0.34180278, 0.35573085,
        0.36887001],
       [0.        , 0.1613246 , 0.22100756, 0.26274309, 0.29132978,
        0.31701594, 0.34013894, 0.36233513, 0.37847148, 0.39719674,
        0.41769997]]),
        np.array([[0.        , 0.03464655, 0.05355682, 0.07626264, 0.09416007,
        0.110277  , 0.12725298, 0.14522284, 0.15844632, 0.17296847,
        0.18623038],
       [0.        , 0.08696136, 0.1424967 , 0.17598449, 0.20740439,
        0.24144053, 0.26185737, 0.27983827, 0.30594899, 0.32353889,
        0.33896208],
       [0.        , 0.02964357, 0.04795795, 0.06507512, 0.08094079,
        0.09839205, 0.11173943, 0.12909594, 0.14191668, 0.15980276,
        0.17312901],
       [0.        , 0.02418718, 0.03736538, 0.04709419, 0.05282461,
        0.06054368, 0.06537159, 0.06684568, 0.07149068, 0.07635434,
        0.0811673 ],
       [0.        , 0.22130535, 0.26259727, 0.28464496, 0.30406404,
        0.31789396, 0.33253746, 0.34593679, 0.35598333, 0.37156334,
        0.38055773],
       [0.        , 0.20017683, 0.2483455 , 0.27151293, 0.28561628,
        0.29617493, 0.30818631, 0.31769147, 0.32603008, 0.33504372,
        0.34499083],
       [0.        , 0.1532302 , 0.19571892, 0.22414439, 0.24690155,
        0.26266726, 0.27611619, 0.2929157 , 0.3007093 , 0.31409601,
        0.3244995 ],
       [0.        , 0.19757765, 0.24524196, 0.27195093, 0.29294809,
        0.31321551, 0.32585712, 0.34581936, 0.35993539, 0.37226715,
        0.38985361]])
        ]
    '''
    # Get average utility matrix
    avgUtilMat = np.average(np.array(utilMatList), axis=0)
    '''29-MAR
    avgUtilMat = np.array([[0.        , 0.03268954, 0.05829443, 0.07944669, 0.09972572,
        0.11694852, 0.1352915 , 0.15051875, 0.16573722, 0.18072208,
        0.19519153],
       [0.        , 0.07378511, 0.12312079, 0.16201624, 0.19539617,
        0.22490084, 0.24858435, 0.27156269, 0.29321109, 0.31380596,
        0.33221483],
       [0.        , 0.02352739, 0.04155001, 0.05815998, 0.07502954,
        0.09020935, 0.10321972, 0.11957546, 0.13255503, 0.14890168,
        0.16111984],
       [0.        , 0.01417615, 0.02386566, 0.0324658 , 0.03985089,
        0.04714713, 0.05185139, 0.05798947, 0.06266097, 0.06774322,
        0.07296075],
       [0.        , 0.19581007, 0.24223627, 0.27037098, 0.28984852,
        0.30735976, 0.32114572, 0.33588068, 0.34857645, 0.3606669 ,
        0.37234249],
       [0.        , 0.14259745, 0.18631412, 0.21362738, 0.23306125,
        0.24821311, 0.26109551, 0.27318418, 0.2830938 , 0.29309143,
        0.3030213 ],
       [0.        , 0.1586517 , 0.20789791, 0.23771931, 0.26084306,
        0.27831627, 0.29349633, 0.30631737, 0.31818952, 0.32924555,
        0.34008634],
       [0.        , 0.17227148, 0.22026507, 0.25119196, 0.27580545,
        0.29738212, 0.31792862, 0.3354911 , 0.35308941, 0.36979094,
        0.38476591]])
    '''
    # Find allocation for sample budget
    allocArr = forwardAllocateWithBudget(avgUtilMat, int(sampBudget / testInt))
    designArr = allocArr / np.sum(allocArr, axis=0)
    # Get utility for this allocation at the sample budget
    utilDict.update({'method': 'weightsNodeDraw4linear'})
    compUtilList, unifUtilList, origUtilList = [], [], []
    numReps = 10
    for rep in range(numReps):
        CSdict3 = methods.GeneratePostSamples(CSdict3)
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New Bayes draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        print('Generating loss matrix...')
        tempLossMat = lf.lossMatSetBayesDraws(dictTemp['postSamples'], lossDict.copy(), lossDict['bayesDraws'])
        tempLossDict = lossDict.copy()
        tempLossDict.update({'lossMat': tempLossMat})
        newBayesDraws, newLossMat = lf.add_cand_neighbors(tempLossDict.copy(), CSdict3['postSamples'],
                                                          dictTemp['postSamples'])
        tempLossDict.update({'bayesDraws': newBayesDraws, 'lossMat': newLossMat})
        baseLoss = (np.sum(newLossMat, axis=1) / newLossMat.shape[1]).min()
        # Get a new set of data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        currCompUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[designArr],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Heuristic utility: ' + str(currCompUtil))
        compUtilList.append(currCompUtil)
        '''29-MAR
        compUtilList = [1.3294820556463969, 1.3322611116716248, 1.370423216484164, 1.532186935536589, 1.3693257074244394, 1.6353967001347849, 1.2341287083770722, 1.426133341998029, 1.4654687057582558, 1.3143393038648186]
        '''
        # Find the equivalent uniform allocation
        currUnifUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[unifDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Uniform utility: ' + str(currUnifUtil))
        unifUtilList.append([currUnifUtil])
        unifAdd, contUnif, unifCount = 0, False, 0
        if currUnifUtil < currCompUtil:
            contUnif = True
        while contUnif:
            unifAdd += testInt
            print('Adding ' + str(unifAdd) + ' for uniform')
            currUnifUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[unifDes], numtests=sampBudget + unifAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currUnifUtil))
            unifUtilList[rep].append(currUnifUtil)
            if currUnifUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if unifCount < 3:
                    unifCount += 1
                else:
                    contUnif = False
        '''29-MAR
        unifUtilList = [[1.2118351543022126, 1.2432796389699439, 1.2844993314987811, 1.3228866921529985, 1.3541260194039313, 1.3953921851865685, 1.4171599610817247, 1.4538281346843323], [1.2161717669151089, 1.2569309840419507, 1.3088269208738081, 1.3251430471218675, 1.3733637532758065, 1.4090277568345484, 1.4487814756713835, 1.4705198588836144], [1.255623503947764, 1.2836056418895585, 1.329379764399877, 1.3536197807940678, 1.3757114718813495, 1.4325869348738673, 1.4744718446628542, 1.497464870603976], [1.3899021183268778, 1.421954682517089, 1.4826072679049576, 1.5204403051651663, 1.5463170773152646, 1.5827526828390748, 1.6202754612484802, 1.6412713709401832], [1.2341067619594592, 1.268589975733319, 1.310335707490585, 1.350720406274022, 1.3931107109637315, 1.4282159569909956, 1.4597044569058797, 1.4799469134372218], [1.5911197105557884, 1.6027880932492269, 1.6173104988077487, 1.6336420165193704, 1.6424973123346538, 1.6649350769715694, 1.6690230251426055, 1.6796375291221137], [1.1198485607654884, 1.1478631328791158, 1.1935391806147306, 1.2280048152811052, 1.2733360876880222, 1.3025945938316972, 1.3308008347074916, 1.3702381663267778], [1.2998138637921444, 1.3442816976122822, 1.3923565151407131, 1.4164302555246806, 1.452700215013607, 1.4826447596702823, 1.529898294766903, 1.5598409077010347], [1.3593119575307404, 1.3787786698459898, 1.427575305523221, 1.4688845555621493, 1.4968825892036102, 1.5371117335786502, 1.5708307310006875], [1.1929958129787819, 1.2255327047092384, 1.2741586977453947, 1.3188752936991137, 1.329872378864275, 1.370085765489394, 1.4135368079054316]]
        '''
        # Find the equivalent rudimentary allocation
        currOrigUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[origDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Rudimentary utility: ' + str(currOrigUtil))
        origUtilList.append([currOrigUtil])
        origAdd, contOrig, origCount = 0, False, 0
        if currOrigUtil < currCompUtil:
            contOrig = True
        while contOrig:
            origAdd += testInt * 3
            print('Adding ' + str(origAdd) + ' for rudimentary')
            currOrigUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[origDes], numtests=sampBudget + origAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currOrigUtil))
            origUtilList[rep].append(currOrigUtil)
            if currOrigUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if origCount < 3:
                    origCount += 1
                else:
                    contOrig = False
        '''29-MAR
        origUtilList = [[0.36922533763690435, 0.43533528289705226, 0.514500110655872, 0.5851835556603677, 0.6636598814940466, 0.7272434161568135, 0.8052589004957467, 0.9146108684884284, 0.9257998602028357, 0.995250712143688, 1.042184779490925, 1.1073084138011016, 1.1424380618921353, 1.2426962508299493, 1.2800002855598827, 1.3143951198681774, 1.4082577525444693, 1.4405513972739477, 1.4533161414794784, 1.5181142256514892], [0.41618347662532873, 0.4748613852087362, 0.556156732579935, 0.62230232275946, 0.6881995549557165, 0.761210550762597, 0.8257442245802622, 0.9328049501929754, 0.9686389930761132, 1.0125321926145476, 1.1042878735731856, 1.1430658062301684, 1.2108896371339752, 1.2643634307630136, 1.303176668292667, 1.365727806059069, 1.422017248301688, 1.445581395634556, 1.4885045785524618], [0.40272973996166694, 0.46029870893338387, 0.5225407019418835, 0.5987897040764185, 0.6644016109053221, 0.7446920038050959, 0.8064619019987669, 0.8923347454832795, 0.929553175483135, 1.0130663957313781, 1.070798414535457, 1.1217801109896617, 1.1844576464189163, 1.2419216819082424, 1.3037126776042531, 1.3525587230621263, 1.4096924940434827, 1.4421881656641822, 1.4632292928613828, 1.5254423809521853], [0.5050009976769028, 0.5839979838885658, 0.6551006161845216, 0.7256851631149153, 0.7970254820629621, 0.8786429649450058, 0.9531170697983238, 1.0657677795679046, 1.0901202380019543, 1.1658549350387029, 1.2161628376768903, 1.275403561251347, 1.3340962893135901, 1.3953544542587726, 1.4499164800602244, 1.4988062174409462, 1.5834276492866315, 1.6463815686006518, 1.6505408279271414, 1.6935828388119596], [0.37946755952557254, 0.44874735122749865, 0.5106623851241587, 0.595762861802009, 0.6532126187694236, 0.7372144753151879, 0.8138368458472574, 0.9040512342662939, 0.9205038998865, 1.000253265519126, 1.0560820575064596, 1.111561203338879, 1.1619487789026062, 1.2201696458695426, 1.2798332421093321, 1.330798079762872, 1.4082456594671209, 1.4581116431242016, 1.450173312078289, 1.5143443220072146], [1.2998683192031606, 1.328493332439292, 1.3614958587198747, 1.3939134270890656, 1.4245136425468585, 1.454172757563698, 1.4819501914131459, 1.5113234967730922, 1.525338240181693, 1.5481773976897995, 1.5580791572675219, 1.5830426781610991, 1.5992029591491561, 1.63347497431895, 1.6297025902733933, 1.6532879441795632, 1.6669199755445994, 1.6851184949034943, 1.6954434778104772], [0.29525680960479805, 0.3649238711087186, 0.42722526753190104, 0.5088146503035946, 0.5744730549876169, 0.6405011251559989, 0.7055062684417046, 0.8239313221175188, 0.8506504946229128, 0.9262605572183595, 0.977464123084423, 1.0324068131218804, 1.0822977651786596, 1.1561277704113797, 1.203909279106333, 1.2359115615241105, 1.305209001693997, 1.3318421870074837, 1.3491679441957865], [0.4463840261672667, 0.5126682372262734, 0.589008575684705, 0.6580931668752679, 0.7272981359544062, 0.800380918763381, 0.8710629936628012, 0.9815748514327356, 0.9997051309699132, 1.0738903798146548, 1.1353187275213665, 1.2050472819379046, 1.2506061813298777, 1.309690967518899, 1.3733376806792807, 1.406815979582368, 1.4881980480632975, 1.5418270385300432, 1.5630771536930994, 1.5918219797955993], [0.45037120080075344, 0.5211654476467373, 0.5986352320064716, 0.6670162625647862, 0.7474827642881965, 0.8176085810853428, 0.8763331897401132, 1.0007018690210199, 1.0331947540951223, 1.0875842881358606, 1.1495490497923089, 1.2110398512269391, 1.278374323935087, 1.3384241807582384, 1.3703722452326441, 1.4309154401302582, 1.4978386961144046, 1.5514612668014078, 1.5659568284217174, 1.6004370103566363], [0.35796240018706893, 0.4181315720437322, 0.4949729522870463, 0.5642743641920704, 0.6399796428955802, 0.7000679647039303, 0.7700302374054346, 0.8943943700473911, 0.8985600119938795, 0.9631258732747101, 1.03085052033259, 1.1021486362325703, 1.1508789550281544, 1.2193553840089209, 1.2473040442748125, 1.3044641336547125, 1.3851070395683993, 1.423823445000692, 1.4286030374766323, 1.4947260326440155]]
        '''
    compAvg = np.average(compUtilList)
    # Locate closest sample point for uniform and rudimentary to compAvg
    minListLen = np.min([len(i) for i in unifUtilList])
    unifUtilArr = np.array([i[:minListLen] for i in unifUtilList])
    unifAvgArr = np.average(unifUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(unifAvgArr.tolist()) if val > compAvg)
    unifSampSaved = round((compAvg - unifAvgArr[kInd - 1]) / (unifAvgArr[kInd] - unifAvgArr[kInd - 1]) * testInt) + (
            kInd - 1) * testInt
    print(unifSampSaved)
    '''29-MAR: 32 saved'''
    # Rudimentary
    minListLen = np.min([len(i) for i in origUtilList])
    origUtilArr = np.array([i[:minListLen] for i in origUtilList])
    origAvgArr = np.average(origUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(origAvgArr.tolist()) if val > compAvg)
    origSampSaved = round(
        (compAvg - origAvgArr[kInd - 1]) / (origAvgArr[kInd] - origAvgArr[kInd - 1]) * testInt * 3) + (
                            kInd - 1) * testInt * 3
    print(origSampSaved)
    '''29-MAR: 455 saved'''

    # Do again for different sample budget
    sampBudget = 90
    allocArr = forwardAllocateWithBudget(avgUtilMat, int(sampBudget / testInt))
    designArr = allocArr / np.sum(allocArr, axis=0)
    # Get utility for this allocation at the sample budget
    utilDict.update({'method': 'weightsNodeDraw4linear'})
    compUtilList, unifUtilList, origUtilList = [], [], []
    numReps = 5
    for rep in range(numReps):
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New Bayes draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        print('Generating loss matrix...')
        tempLossMat = lf.lossMatSetBayesDraws(dictTemp['postSamples'], lossDict.copy(), lossDict['bayesDraws'])
        tempLossDict = lossDict.copy()
        tempLossDict.update({'lossMat': tempLossMat})
        newBayesDraws, newLossMat = lf.add_cand_neighbors(tempLossDict.copy(), CSdict3['postSamples'],
                                                          dictTemp['postSamples'])
        tempLossDict.update({'bayesDraws': newBayesDraws, 'lossMat': newLossMat})
        baseLoss = (np.sum(newLossMat, axis=1) / newLossMat.shape[1]).min()
        # Get a new set of data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        currCompUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[designArr],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Heuristic utility: ' + str(currCompUtil))
        compUtilList.append(currCompUtil)
        '''29-MAR
        compUtilList = [0.9559255628855037, 0.9460268133645227, 0.9481317666072409, 0.9246563279775266, 0.9248569886735063]
        '''
        # Find the equivalent uniform allocation
        currUnifUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[unifDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Uniform utility: ' + str(currUnifUtil))
        unifUtilList.append([currUnifUtil])
        unifAdd, contUnif, unifCount = 0, False, 0
        if currUnifUtil < currCompUtil:
            contUnif = True
        while contUnif:
            unifAdd += testInt
            print('Adding ' + str(unifAdd) + ' for uniform')
            currUnifUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[unifDes], numtests=sampBudget + unifAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currUnifUtil))
            unifUtilList[rep].append(currUnifUtil)
            if currUnifUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if unifCount < 3:
                    unifCount += 1
                else:
                    contUnif = False
        '''29-MAR
        unifUtilList = [[0.8403940327153139, 0.8907220753973908, 0.9407541169826046, 0.9907791828940407, 1.0311205068084504, 1.0841063868442693, 1.119546381727003], [0.8411018920974969, 0.8866382154608856, 0.9385584445925894, 0.9942119854794726, 1.0281939419641302, 1.0843655861460855, 1.12789528135598], [0.8342950637408433, 0.8734629640294598, 0.9415724023722398, 0.9848054261432737, 1.0314764773932659, 1.0868446487378352, 1.1204096946385351], [0.816311262339747, 0.8706450321598034, 0.9283706978127322, 0.9656433866640173, 1.0136364562897837, 1.0605360433424544], [0.8151854763902429, 0.8678418182960876, 0.9181005180555886, 0.9692268130826842, 1.0203037530927808, 1.069376245842566, 1.1121887906612535]]
        '''
        # Find the equivalent rudimentary allocation
        currOrigUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[origDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Rudimentary utility: ' + str(currOrigUtil))
        origUtilList.append([currOrigUtil])
        origAdd, contOrig, origCount = 0, False, 0
        if currOrigUtil < currCompUtil:
            contOrig = True
        while contOrig:
            origAdd += testInt * 3
            print('Adding ' + str(origAdd) + ' for rudimentary')
            currOrigUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[origDes], numtests=sampBudget + origAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currOrigUtil))
            origUtilList[rep].append(currOrigUtil)
            if currOrigUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if origCount < 3:
                    origCount += 1
                else:
                    contOrig = False
        '''29-MAR
        origUtilList = [[0.22976374041506187, 0.28345696560799194, 0.34217682723134235, 0.4036203239629508, 0.4686341982751361, 0.5398593959420559, 0.611654721682025, 0.6657599068069464, 0.7496703357300301, 0.8100501441937022, 0.9185987103069446, 0.9374576034758149, 1.019040275592999, 1.0554162128742068, 1.1135802465199327, 1.182537529794141], [0.22968925250847327, 0.2922970823046107, 0.35274582852757597, 0.4174417040029361, 0.4871971704080704, 0.5507475406048554, 0.633025488277632, 0.6976451242698412, 0.7628317466140881, 0.8509034490828982, 0.9564524593100567, 0.9703241649204704, 1.0446526701199645, 1.111080809988826], [0.2349008699724564, 0.2904223983279177, 0.3458400857965511, 0.4095447501276652, 0.47141816506490297, 0.5481246918435922, 0.609760526189346, 0.6833096894715935, 0.7582880826479124, 0.8215851180463982, 0.9215925019507814, 0.9470822543915589, 1.015595103749876, 1.0614050494397325, 1.1312231482481172, 1.180446263641346], [0.1896776520476422, 0.2522213332950862, 0.3115650278529971, 0.3815394473892493, 0.4509628146429252, 0.5153543830925975, 0.5990585452478494, 0.6615325633959421, 0.7209411810266899, 0.8050619200054423, 0.9008116889735631, 0.9358108067615025, 0.9868338399714025, 1.0626042109513594, 1.1141631931032516], [0.22843435322253391, 0.29106278991591505, 0.3426162477005228, 0.40265239606089853, 0.46750720183965155, 0.5380621860445167, 0.5983486783461447, 0.6559839243100867, 0.7356283522643015, 0.8066411950181918, 0.9069663076034642, 0.9243400886509239, 1.0042382647862609, 1.0375162941601177, 1.11636181495733, 1.1729951817439699]]
        '''
    compAvg = np.average(compUtilList)
    # Locate closest sample point for uniform and rudimentary to compAvg
    minListLen = np.min([len(i) for i in unifUtilList])
    unifUtilArr = np.array([i[:minListLen] for i in unifUtilList])
    unifAvgArr = np.average(unifUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(unifAvgArr.tolist()) if val > compAvg)
    unifSampSaved = round((compAvg - unifAvgArr[kInd - 1]) / (unifAvgArr[kInd] - unifAvgArr[kInd - 1]) * testInt) + (
            kInd - 1) * testInt
    print(unifSampSaved)
    '''29-MAR: 21 saved'''
    # Rudimentary
    minListLen = np.min([len(i) for i in origUtilList])
    origUtilArr = np.array([i[:minListLen] for i in origUtilList])
    origAvgArr = np.average(origUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(origAvgArr.tolist()) if val > compAvg)
    origSampSaved = round(
        (compAvg - origAvgArr[kInd - 1]) / (origAvgArr[kInd] - origAvgArr[kInd - 1]) * testInt * 3) + (
                            kInd - 1) * testInt * 3
    print(origSampSaved)
    '''29-MAR: 326 saved'''

    ##############################################
    ##############################################
    # Use different variance for untested nodes (TNvar = 1)
    ##############################################
    ##############################################
    SNpriorMean = np.repeat(sps.logit(0.1), numSN)
    # Establish test nodes according to assessment by regulators
    TNpriorMean = sps.logit(np.array([0.1, 0.1, 0.15, 0.15, 0.15, 0.1, 0.15, 0.1]))
    TNvar, SNvar = 1., 4.
    CSdict3['prior'] = prior_normal_assort(np.concatenate((SNpriorMean, TNpriorMean)),
                                           np.diag(np.concatenate((np.repeat(SNvar, numSN), np.repeat(TNvar, numTN)))))

    sampBudget = 180
    unifDes = np.zeros(numTN) + 1 / numTN
    origDes = np.sum(rd3_N, axis=1) / np.sum(rd3_N)

    # Use original loss parameters
    paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                                  marketvec=np.ones(numTN + numSN), candneighnum=1000)

    numtargetdraws, numDataDraws = 5100, 5000

    # Find heuristic allocation first
    utilDict = {'method': 'weightsNodeDraw3linear'}

    numReps = 10
    utilMatList = []
    # set testMax to highest expected allocation for any one node
    testMax, testInt = 100, 10
    testArr = np.arange(0, testMax + 1, testInt)
    for rep in range(numReps):
        CSdict3 = methods.GeneratePostSamples(CSdict3)
        # Withdraw a subset of MCMC prior draws
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New loss draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        # Get new data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        # Get marginal utilities at each test node
        currMargUtilMat = GetMargUtilAtNodes(dictTemp.copy(), testMax, testInt, lossDict.copy(), utilDict.copy(),
                                             masterDraws=CSdict3['postSamples'], printUpdate=True)
        print(repr(currMargUtilMat))
        utilMatList.append(currMargUtilMat)
        for mat in utilMatList:
            for i in range(8):
                plt.plot(testArr, mat[i], linewidth=0.2)
        avgUtilMat = np.average(np.array(utilMatList), axis=0)
        for i in range(8):
            plt.plot(testArr, avgUtilMat[i], linewidth=2)
        plt.ylim([0, 0.4])
        # plt.title('Comprehensive utility for allocations via heuristic\nUntested nodes')
        plt.show()
        plt.close()
    '''16-APR run
    utilMatList = [np.array([[ 0.00000000e+00,  2.42396675e-04,  3.20230823e-03,
         1.13284087e-02,  1.94657122e-02,  2.62061415e-02,
         3.88236208e-02,  4.04924781e-02,  5.47599928e-02,
         6.04057205e-02,  7.06427899e-02],
       [ 0.00000000e+00,  6.04563751e-03,  2.15447172e-02,
         4.23433729e-02,  6.10262885e-02,  8.10019989e-02,
         9.72524061e-02,  1.16060782e-01,  1.29981679e-01,
         1.48393422e-01,  1.61181604e-01],
       [ 0.00000000e+00,  1.00256887e-03,  3.26490281e-04,
         1.61728744e-03,  3.53224420e-03,  3.62641448e-03,
         6.02253268e-03,  1.02050223e-02,  1.98476879e-02,
         2.46427119e-02,  2.46614623e-02],
       [ 0.00000000e+00,  7.52085145e-04, -5.53392740e-05,
         2.39686647e-03,  4.13382996e-05,  9.55957180e-04,
         3.18593389e-03,  2.98822173e-03,  4.58077280e-03,
         4.33997973e-03,  2.42415501e-03],
       [ 0.00000000e+00,  5.67431291e-03,  1.58837032e-02,
         1.94440175e-02,  2.76135470e-02,  3.82355415e-02,
         4.75105727e-02,  5.23235386e-02,  5.89161023e-02,
         7.05563810e-02,  7.51840299e-02],
       [ 0.00000000e+00,  7.50460795e-03,  1.29316154e-02,
         2.05066460e-02,  2.23250269e-02,  3.31356099e-02,
         3.85569584e-02,  3.90576403e-02,  4.65455257e-02,
         5.37638762e-02,  5.89107823e-02],
       [ 0.00000000e+00,  1.09740517e-02,  2.61352022e-02,
         3.63076794e-02,  4.74362318e-02,  6.03036681e-02,
         6.83312098e-02,  8.07054688e-02,  9.02880404e-02,
         1.01733692e-01,  1.09758503e-01],
       [ 0.00000000e+00,  1.14155763e-02,  2.43294991e-02,
         4.45220000e-02,  5.05155301e-02,  6.62139035e-02,
         7.36593231e-02,  8.61472550e-02,  9.55610988e-02,
         1.08283330e-01,  1.23270234e-01]]),
         np.array([[0.        , 0.03811953, 0.06560311, 0.08433636, 0.10221565,
        0.11873004, 0.13370529, 0.15204305, 0.16315541, 0.17269487,
        0.18292992],
       [0.        , 0.05745969, 0.09640676, 0.12881145, 0.1605231 ,
        0.1795625 , 0.20482036, 0.2315843 , 0.24935131, 0.27113555,
        0.28813987],
       [0.        , 0.03002749, 0.0505948 , 0.06868421, 0.08341559,
        0.10024334, 0.11438373, 0.1256726 , 0.14096871, 0.15046224,
        0.16418731],
       [0.        , 0.03059142, 0.04750423, 0.06201837, 0.07118619,
        0.0822781 , 0.0882888 , 0.09761305, 0.09923054, 0.10852595,
        0.11097947],
       [0.        , 0.05709218, 0.09115235, 0.11319959, 0.13758884,
        0.15236476, 0.16676019, 0.18884446, 0.19817115, 0.21472644,
        0.22160759],
       [0.        , 0.05945712, 0.08883051, 0.10975297, 0.1286701 ,
        0.14350508, 0.1545749 , 0.16325775, 0.17687637, 0.18840856,
        0.19944866],
       [0.        , 0.06104748, 0.09719459, 0.12125061, 0.1411482 ,
        0.15563632, 0.1672097 , 0.18134778, 0.19457297, 0.20351773,
        0.21417662],
       [0.        , 0.06410309, 0.10725237, 0.13788962, 0.16213862,
        0.1818068 , 0.20168623, 0.21557091, 0.233843  , 0.24841118,
        0.262986  ]]),
        np.array([[0.        , 0.03944043, 0.06788605, 0.09297662, 0.10768374,
        0.13122295, 0.14108364, 0.15521496, 0.17203933, 0.17913223,
        0.19535031],
       [0.        , 0.08085556, 0.12142771, 0.15103712, 0.17632781,
        0.20515344, 0.22535116, 0.24601015, 0.26258103, 0.28463093,
        0.29580423],
       [0.        , 0.02942633, 0.05595274, 0.07199179, 0.09043403,
        0.1057393 , 0.12621354, 0.13690531, 0.14849022, 0.16690763,
        0.17324393],
       [0.        , 0.01492711, 0.02954841, 0.03804085, 0.04283141,
        0.05337256, 0.05798974, 0.06434694, 0.07247334, 0.07518617,
        0.07761606],
       [0.        , 0.08327445, 0.11991214, 0.14525866, 0.16990953,
        0.18824052, 0.20677904, 0.22194827, 0.23691794, 0.24877192,
        0.26187603],
       [0.        , 0.07117717, 0.10354621, 0.12350947, 0.13873205,
        0.15389297, 0.16646436, 0.17298562, 0.18134335, 0.19224953,
        0.19925224],
       [0.        , 0.05348188, 0.08130002, 0.1046593 , 0.12264074,
        0.1382026 , 0.15236904, 0.16603785, 0.17478198, 0.187192  ,
        0.19973066],
       [0.        , 0.07516747, 0.11995009, 0.15205239, 0.18099364,
        0.20275286, 0.22389967, 0.24422396, 0.26601848, 0.27932529,
        0.2883326 ]]),
        np.array([[0.        , 0.02562188, 0.04577167, 0.06458678, 0.08115325,
        0.10008975, 0.11362428, 0.12878142, 0.14326498, 0.15502926,
        0.16428072],
       [0.        , 0.03974193, 0.08302608, 0.11722163, 0.15561008,
        0.1813829 , 0.2103356 , 0.23168418, 0.25540997, 0.27593555,
        0.29465737],
       [0.        , 0.01527392, 0.03123377, 0.04818121, 0.06458226,
        0.08344139, 0.09397417, 0.10944844, 0.12033533, 0.13809788,
        0.14812667],
       [0.        , 0.01322704, 0.02314015, 0.03332947, 0.03984569,
        0.04595026, 0.05190643, 0.05661767, 0.0615517 , 0.06511808,
        0.0755046 ],
       [0.        , 0.0633064 , 0.10328246, 0.12561947, 0.15098846,
        0.16719065, 0.18476185, 0.20047884, 0.21491536, 0.22630393,
        0.24269069],
       [0.        , 0.04032244, 0.06594993, 0.08718239, 0.10461444,
        0.11985605, 0.13051566, 0.14289138, 0.15222888, 0.16054511,
        0.16984962],
       [0.        , 0.02188566, 0.05065604, 0.07762838, 0.10241211,
        0.12270522, 0.13948788, 0.15656684, 0.17060698, 0.18515217,
        0.19313174],
       [0.        , 0.04548308, 0.08743143, 0.12365583, 0.14916544,
        0.1768051 , 0.19703581, 0.21730265, 0.23723354, 0.25615117,
        0.27046461]]),
        np.array([[0.        , 0.0322023 , 0.05635331, 0.07997042, 0.09826405,
        0.1226143 , 0.1394517 , 0.15776901, 0.1718681 , 0.19048783,
        0.20085954],
       [0.        , 0.07794621, 0.13492776, 0.18111903, 0.21562345,
        0.24658048, 0.27158093, 0.29895699, 0.317687  , 0.3427732 ,
        0.35865271],
       [0.        , 0.02738942, 0.05193613, 0.07060464, 0.09016341,
        0.10968239, 0.12719286, 0.14051514, 0.15692391, 0.16994469,
        0.1848067 ],
       [0.        , 0.0265919 , 0.03912619, 0.05221717, 0.05843324,
        0.06556135, 0.0738122 , 0.07877649, 0.08530257, 0.09055843,
        0.09570565],
       [0.        , 0.11525455, 0.15632202, 0.18176932, 0.19929359,
        0.21261399, 0.2284878 , 0.24037799, 0.2493788 , 0.26154852,
        0.27118797],
       [0.        , 0.04668348, 0.06807766, 0.08683055, 0.10458763,
        0.11690899, 0.12891163, 0.14053241, 0.14860903, 0.16029701,
        0.16850239],
       [0.        , 0.05218547, 0.08543459, 0.11391074, 0.13700149,
        0.15743278, 0.17245873, 0.18871904, 0.20102133, 0.21434335,
        0.22715992],
       [0.        , 0.0652915 , 0.10099218, 0.12931194, 0.15363046,
        0.17941756, 0.20389919, 0.21980677, 0.23852828, 0.25258991,
        0.27162198]]),
        np.array([[ 0.00000000e+00, -1.33378661e-03,  3.38162518e-03,
         1.49139106e-02,  2.80576837e-02,  3.84459021e-02,
         4.62755576e-02,  5.67642130e-02,  6.79973215e-02,
         8.39850540e-02,  9.15474280e-02],
       [ 0.00000000e+00,  1.18347691e-03,  2.10499298e-02,
         4.10865610e-02,  6.28216666e-02,  8.31720117e-02,
         1.03439731e-01,  1.24987910e-01,  1.41352194e-01,
         1.64840530e-01,  1.76744298e-01],
       [ 0.00000000e+00,  3.86054535e-03,  8.98175388e-03,
         1.55423101e-02,  2.33724953e-02,  3.48543711e-02,
         4.74447507e-02,  5.49591604e-02,  6.68480173e-02,
         7.39104485e-02,  8.02011676e-02],
       [ 0.00000000e+00,  1.09943747e-04,  2.29725248e-03,
         8.39216248e-04,  2.65464394e-03,  5.78760141e-03,
         9.35390227e-03,  1.14212130e-02,  1.76976460e-02,
         2.12287665e-02,  2.01117794e-02],
       [ 0.00000000e+00,  2.67870894e-02,  4.87457890e-02,
         6.09746558e-02,  8.12638858e-02,  9.35765932e-02,
         1.04434553e-01,  1.19172216e-01,  1.24798547e-01,
         1.35955564e-01,  1.43883111e-01],
       [ 0.00000000e+00,  1.62988870e-02,  2.74358679e-02,
         3.78733344e-02,  4.69544824e-02,  5.87030336e-02,
         6.35329356e-02,  7.57123749e-02,  8.13563636e-02,
         9.04604442e-02,  9.72823619e-02],
       [ 0.00000000e+00,  1.80901377e-02,  3.04434521e-02,
         4.55417670e-02,  6.01873610e-02,  7.05585465e-02,
         8.22568399e-02,  9.20671934e-02,  1.04693344e-01,
         1.11546502e-01,  1.23048669e-01],
       [ 0.00000000e+00,  8.79836019e-03,  2.55305749e-02,
         3.62514610e-02,  5.23579577e-02,  6.69376819e-02,
         8.11826405e-02,  9.36614666e-02,  1.06692437e-01,
         1.20488393e-01,  1.36260552e-01]]),
         np.array([[0.        , 0.03227281, 0.05580771, 0.07839099, 0.0920771 ,
        0.106625  , 0.12028817, 0.13165082, 0.14430716, 0.15788463,
        0.16915312],
       [0.        , 0.07172128, 0.1100403 , 0.14057169, 0.17324914,
        0.19687106, 0.21613451, 0.23487285, 0.25535987, 0.27466824,
        0.29237743],
       [0.        , 0.01580887, 0.03402267, 0.04577325, 0.06468195,
        0.07417103, 0.08839253, 0.10121621, 0.11388309, 0.12389734,
        0.1369677 ],
       [0.        , 0.01493414, 0.02774604, 0.03508973, 0.04228311,
        0.04929565, 0.05502503, 0.05889306, 0.06808357, 0.0691562 ,
        0.07191895],
       [0.        , 0.04932403, 0.07852366, 0.09973913, 0.11685897,
        0.13438982, 0.15105106, 0.16628666, 0.17672423, 0.18926244,
        0.2017076 ],
       [0.        , 0.03448676, 0.05684674, 0.07801021, 0.09354317,
        0.11011874, 0.12071243, 0.13228707, 0.14263717, 0.14909786,
        0.15888542],
       [0.        , 0.05057643, 0.08178916, 0.10654252, 0.12702597,
        0.14101178, 0.15470422, 0.16545525, 0.17875911, 0.18586909,
        0.20000364],
       [0.        , 0.03418904, 0.06276877, 0.08847529, 0.10843572,
        0.12999336, 0.14807664, 0.16353795, 0.17813938, 0.19151506,
        0.20700502]]),
        np.array([[0.        , 0.01029876, 0.02618621, 0.03714465, 0.05136745,
        0.06868583, 0.07403239, 0.08955518, 0.09842382, 0.10802708,
        0.11838348],
       [0.        , 0.02566301, 0.05148878, 0.07227454, 0.09609234,
        0.11531758, 0.13572057, 0.14841502, 0.16651545, 0.18304201,
        0.19493594],
       [0.        , 0.00158555, 0.00785526, 0.01178175, 0.01821416,
        0.03034865, 0.0405738 , 0.04943213, 0.05972604, 0.06975148,
        0.07625012],
       [0.        , 0.00322665, 0.00528939, 0.00465704, 0.00935902,
        0.01603999, 0.01929987, 0.0260161 , 0.02342866, 0.02888371,
        0.03536483],
       [0.        , 0.02331508, 0.04874351, 0.06696975, 0.08280349,
        0.09810649, 0.11116115, 0.12295146, 0.13570265, 0.14812977,
        0.16027418],
       [0.        , 0.01556724, 0.02832697, 0.03915105, 0.0500025 ,
        0.05626295, 0.06353286, 0.07453253, 0.08186147, 0.08996653,
        0.09700578],
       [0.        , 0.04118172, 0.0652478 , 0.08028442, 0.0952952 ,
        0.11216426, 0.12027487, 0.13260793, 0.13957687, 0.15127941,
        0.15612507],
       [0.        , 0.03685627, 0.05677045, 0.07897042, 0.09689779,
        0.11158963, 0.13474686, 0.14946384, 0.16300807, 0.17735385,
        0.19397444]]),
        np.array([[0.        , 0.05157411, 0.08067115, 0.11075872, 0.13380034,
        0.15537734, 0.1715855 , 0.18774164, 0.20565422, 0.22442839,
        0.23694193],
       [0.        , 0.08679911, 0.14299572, 0.18616211, 0.22449118,
        0.25664096, 0.28218804, 0.3037789 , 0.33091671, 0.34481158,
        0.36621771],
       [0.        , 0.02840581, 0.05472313, 0.07404538, 0.09513166,
        0.11217247, 0.13378456, 0.14876145, 0.16712939, 0.18247988,
        0.19821322],
       [0.        , 0.01126617, 0.02782293, 0.04024639, 0.05105928,
        0.0613415 , 0.07007162, 0.07955166, 0.0851428 , 0.09326523,
        0.09810901],
       [0.        , 0.10325655, 0.14441763, 0.17828927, 0.1977895 ,
        0.21901271, 0.22955201, 0.25052757, 0.2601526 , 0.27561432,
        0.28483331],
       [0.        , 0.0332273 , 0.06203382, 0.08490596, 0.10248798,
        0.1188512 , 0.13304714, 0.14819313, 0.15772194, 0.16717978,
        0.17744525],
       [0.        , 0.06222392, 0.10563256, 0.13767777, 0.16159825,
        0.18009117, 0.19980196, 0.21397775, 0.22610048, 0.24073571,
        0.2517689 ],
       [0.        , 0.07809061, 0.1217117 , 0.15952437, 0.18626579,
        0.21286655, 0.23676512, 0.24991004, 0.27250332, 0.29025812,
        0.30455024]]),
         ]
    '''
    # Get average utility matrix
    avgUtilMat = np.average(np.array(utilMatList), axis=0)
    '''17-APR
    avgUtilMat = np.array([[0.        , 0.02538205, 0.04498479, 0.06382299, 0.07934277,
        0.09644414, 0.10876335, 0.12222364, 0.13571893, 0.14800834,
        0.1588988 ],
       [0.        , 0.04971288, 0.08698975, 0.1178475 , 0.14730723,
        0.17174255, 0.19409148, 0.21515012, 0.23435058, 0.25447011,
        0.2698568 ],
       [0.        , 0.01697561, 0.03284742, 0.04535798, 0.05928087,
        0.07269771, 0.0864425 , 0.09745727, 0.11046138, 0.1222327 ,
        0.13185092],
       [0.        , 0.01284738, 0.02249103, 0.02987057, 0.03529932,
        0.042287  , 0.04765928, 0.05291382, 0.05749907, 0.06180695,
        0.06530383],
       [0.        , 0.05858718, 0.08966481, 0.11014043, 0.12934553,
        0.14485901, 0.15894425, 0.17365678, 0.18396415, 0.19676325,
        0.20702717],
       [0.        , 0.03608056, 0.05710881, 0.0741914 , 0.08799082,
        0.10124829, 0.11109432, 0.12104999, 0.1299089 , 0.13910763,
        0.14739806],
       [0.        , 0.04129408, 0.06931482, 0.09153369, 0.11052728,
        0.12645626, 0.13965494, 0.1530539 , 0.16448901, 0.17570774,
        0.18610041],
       [0.        , 0.04659944, 0.07852634, 0.10562815, 0.12671122,
        0.14759816, 0.16677239, 0.18218054, 0.19905862, 0.21381959,
        0.22871841]])
    '''
    # Find allocation for sample budget
    allocArr = forwardAllocateWithBudget(avgUtilMat, int(sampBudget / testInt))
    designArr = allocArr / np.sum(allocArr, axis=0)
    # Get utility for this allocation at the sample budget
    utilDict.update({'method': 'weightsNodeDraw4linear'})
    compUtilList, unifUtilList, origUtilList = [], [], []
    numReps = 5
    for rep in range(numReps):
        CSdict3 = methods.GeneratePostSamples(CSdict3)
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New Bayes draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        print('Generating loss matrix...')
        tempLossMat = lf.lossMatSetBayesDraws(dictTemp['postSamples'], lossDict.copy(), lossDict['bayesDraws'])
        tempLossDict = lossDict.copy()
        tempLossDict.update({'lossMat': tempLossMat})
        newBayesDraws, newLossMat = lf.add_cand_neighbors(tempLossDict.copy(), CSdict3['postSamples'],
                                                          dictTemp['postSamples'])
        tempLossDict.update({'bayesDraws': newBayesDraws, 'lossMat': newLossMat})
        baseLoss = (np.sum(newLossMat, axis=1) / newLossMat.shape[1]).min()
        # Get a new set of data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        currCompUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[designArr],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Heuristic utility: ' + str(currCompUtil))
        compUtilList.append(currCompUtil)
        '''29-MAR
        compUtilList = [0.6804838568281597, 0.5856550003022245, 0.6254095101790922, 0.6439354877737644]
        '''
        # Find the equivalent uniform allocation
        currUnifUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[unifDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Uniform utility: ' + str(currUnifUtil))
        unifUtilList.append([currUnifUtil])
        unifAdd, contUnif, unifCount = 0, False, 0
        if currUnifUtil < currCompUtil:
            contUnif = True
        while contUnif:
            unifAdd += testInt
            print('Adding ' + str(unifAdd) + ' for uniform')
            currUnifUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[unifDes], numtests=sampBudget + unifAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currUnifUtil))
            unifUtilList[rep].append(currUnifUtil)
            if currUnifUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if unifCount < 3:
                    unifCount += 1
                else:
                    contUnif = False
        '''29-MAR
        unifUtilList = [[0.5935379024533667, 0.6254713250442667, 0.6562623195111779, 0.6808711621541659, 0.7168748757817776, 0.7319202982199968, 0.765975389199983, 0.565784036587917, 0.592618169504211, 0.6208240555762812, 0.6532950629853596, 0.6778629499522535, 0.6921282498672126, 0.7253217100080462], [0.4999784083728329, 0.5229632604016974, 0.5534855401939267, 0.5731979540749959, 0.6097894817376579, 0.6304150106550095, 0.6500021225537886, 0.6809552040636788, 0.5962994728733131, 0.6235856139238445, 0.6490825357322025, 0.6679002896907607, 0.689952461362362, 0.729467566402211], [0.5357689320992765], [0.5655418227865892]]
        '''
        # Find the equivalent rudimentary allocation
        currOrigUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[origDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Rudimentary utility: ' + str(currOrigUtil))
        origUtilList.append([currOrigUtil])
        origAdd, contOrig, origCount = 0, False, 0
        if currOrigUtil < currCompUtil:
            contOrig = True
        while contOrig:
            origAdd += testInt * 3
            print('Adding ' + str(origAdd) + ' for rudimentary')
            currOrigUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[origDes], numtests=sampBudget + origAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currOrigUtil))
            origUtilList[rep].append(currOrigUtil)
            if currOrigUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if origCount < 3:
                    origCount += 1
                else:
                    contOrig = False
        '''29-MAR
        origUtilList = [[0.34843534000650944, 0.4005118455503265, 0.45622075284999797, 0.5073008421834557, 0.5635113240585876, 0.6107395810111735, 0.674396936557184, 0.7166667259811059, 0.7543208845288172, 0.8187925618857808, 0.8588128348199424, 0.35335111317299805, 0.40469468111089313, 0.4593069090856492, 0.5068671317769411, 0.570169292777301, 0.6111119060965167, 0.6672151335331882, 0.7208431938545123, 0.7577294564634407, 0.799668591106506], [0.2652374343932866, 0.3139615727650482, 0.36564915522093067, 0.4171026398295572, 0.462240449591091, 0.5215522272061781, 0.5671010416871769, 0.6264890137355228, 0.6643840709030466, 0.7132330656948267, 0.7725553673573877, 0.354076087155788, 0.4112100366563256, 0.4576304271133229, 0.5152226969582179, 0.556176814219778, 0.6248842255036928, 0.6840278980223458, 0.7262536353476463, 0.7728169797367421, 0.8162761043729043], [0.2964179974941694], [0.3022531694985586]]
        '''
    compAvg = np.average(compUtilList)
    # Locate closest sample point for uniform and rudimentary to compAvg
    minListLen = np.min([len(i) for i in unifUtilList])
    unifUtilArr = np.array([i[:minListLen] for i in unifUtilList])
    unifAvgArr = np.average(unifUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(unifAvgArr.tolist()) if val > compAvg)
    unifSampSaved = round((compAvg - unifAvgArr[kInd - 1]) / (unifAvgArr[kInd] - unifAvgArr[kInd - 1]) * testInt) + (
            kInd - 1) * testInt
    print(unifSampSaved)
    '''20-APR: 32 saved'''
    # Rudimentary
    minListLen = np.min([len(i) for i in origUtilList])
    origUtilArr = np.array([i[:minListLen] for i in origUtilList])
    origAvgArr = np.average(origUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(origAvgArr.tolist()) if val > compAvg)
    origSampSaved = round(
        (compAvg - origAvgArr[kInd - 1]) / (origAvgArr[kInd] - origAvgArr[kInd - 1]) * testInt * 3) + (
                            kInd - 1) * testInt * 3
    print(origSampSaved)
    '''20-APR: 188 saved'''

    # Do again for different sample budget
    sampBudget = 90
    allocArr = forwardAllocateWithBudget(avgUtilMat, int(sampBudget / testInt))
    designArr = allocArr / np.sum(allocArr, axis=0)
    # Get utility for this allocation at the sample budget
    utilDict.update({'method': 'weightsNodeDraw4linear'})
    compUtilList, unifUtilList, origUtilList = [], [], []
    numReps = 3
    for rep in range(numReps):
        dictTemp = CSdict3.copy()
        dictTemp.update({'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws,
                                                                      replace=False)],
                         'numPostSamples': numtargetdraws})
        # New Bayes draws
        setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
        lossDict.update({'bayesDraws': setDraws})
        print('Generating loss matrix...')
        tempLossMat = lf.lossMatSetBayesDraws(dictTemp['postSamples'], lossDict.copy(), lossDict['bayesDraws'])
        tempLossDict = lossDict.copy()
        tempLossDict.update({'lossMat': tempLossMat})
        newBayesDraws, newLossMat = lf.add_cand_neighbors(tempLossDict.copy(), CSdict3['postSamples'],
                                                          dictTemp['postSamples'])
        tempLossDict.update({'bayesDraws': newBayesDraws, 'lossMat': newLossMat})
        baseLoss = (np.sum(newLossMat, axis=1) / newLossMat.shape[1]).min()
        # Get a new set of data draws
        utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})
        currCompUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[designArr],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Heuristic utility: ' + str(currCompUtil))
        compUtilList.append(currCompUtil)
        '''29-MAR
        compUtilList = [0.3805815394551968, 0.36876846876297176, 0.41539269145230584, 0.38378022038847304, 0.5441983167338162]
        '''
        # Find the equivalent uniform allocation
        currUnifUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[unifDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Uniform utility: ' + str(currUnifUtil))
        unifUtilList.append([currUnifUtil])
        unifAdd, contUnif, unifCount = 0, False, 0
        if currUnifUtil < currCompUtil:
            contUnif = True
        while contUnif:
            unifAdd += testInt
            print('Adding ' + str(unifAdd) + ' for uniform')
            currUnifUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[unifDes], numtests=sampBudget + unifAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currUnifUtil))
            unifUtilList[rep].append(currUnifUtil)
            if currUnifUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if unifCount < 3:
                    unifCount += 1
                else:
                    contUnif = False
        '''29-MAR
        unifUtilList = [[0.305247227403747, 0.3357249464373213, 0.3626944812762938, 0.3956869256258324, 0.4239946878214642, 0.44859454090193296, 0.4780217210651512, 0.3262409949329754, 0.356479873382773, 0.3918320755240847, 0.422728306063187, 0.4525469821128132, 0.4771473121053198, 0.3693674296849059, 0.400888485882104, 0.4216197051108277, 0.46247719886978356, 0.4956112838653417, 0.5088293838913844], [0.2953414451289973, 0.3397080779667161, 0.36691210563001464, 0.402824508403552, 0.43068699728609294, 0.459892072258524, 0.49180941459722316], [0.34415099322131715, 0.5033991057875093, 0.5371790606805726, 0.566770410814156, 0.5941634514992815, 0.6250576497234315, 0.6600476961593165], [0.31180800815399135], [0.46443908909599463]]
        '''
        # Find the equivalent rudimentary allocation
        currOrigUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[origDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
        print('Rudimentary utility: ' + str(currOrigUtil))
        origUtilList.append([currOrigUtil])
        origAdd, contOrig, origCount = 0, False, 0
        if currOrigUtil < currCompUtil:
            contOrig = True
        while contOrig:
            origAdd += testInt * 3
            print('Adding ' + str(origAdd) + ' for rudimentary')
            currOrigUtil = baseLoss - sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict,
                                                               designlist=[origDes], numtests=sampBudget + origAdd,
                                                               utildict=utilDict)[0]
            print('New utility: ' + str(currOrigUtil))
            origUtilList[rep].append(currOrigUtil)
            if currOrigUtil > currCompUtil:  # Add 3 evaluations once an evaluation surpasses the compUtil
                if origCount < 3:
                    origCount += 1
                else:
                    contOrig = False
        '''29-MAR
        origUtilList = [[0.15431498338652894, 0.2063698325148864, 0.2514022034105663, 0.3078058385976612, 0.363436904827787, 0.4105910415018954, 0.4662049685568954, 0.5252424788331918, 0.5750166194879602, 0.20211534099226247, 0.2456088365414355, 0.3113685575635432, 0.3485772923630095, 0.4106787783418877, 0.4597415651354564, 0.5149284314510636, 0.5763472326639567, 0.24135231739577812, 0.2912409142633243, 0.3446525019268467, 0.4033481748370433, 0.4519733201408451, 0.5162864278759591, 0.5644396858786278, 0.6288662413718229], [0.14867998709612262, 0.20923946936662619, 0.26165655507590957, 0.3227650252940295, 0.3779041328462265, 0.42619450788920155, 0.4885673139168105, 0.5420822584105647, 0.595657587361401], [0.18039322769543942, 0.341638334267937, 0.40035330962252713, 0.45721238310184775, 0.5133601514056974, 0.5645640118498916, 0.6268632657321302, 0.6794293725745533, 0.7325607783082035], [0.1508919418338599], [0.28238819816947736]]
        '''
    compAvg = np.average(compUtilList)
    # Locate closest sample point for uniform and rudimentary to compAvg
    minListLen = np.min([len(i) for i in unifUtilList])
    unifUtilArr = np.array([i[:minListLen] for i in unifUtilList])
    unifAvgArr = np.average(unifUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(unifAvgArr.tolist()) if val > compAvg)
    unifSampSaved = round((compAvg - unifAvgArr[kInd - 1]) / (unifAvgArr[kInd] - unifAvgArr[kInd - 1]) * testInt) + (
            kInd - 1) * testInt
    print(unifSampSaved)
    '''20-APR: XX saved'''
    # Rudimentary
    minListLen = np.min([len(i) for i in origUtilList])
    origUtilArr = np.array([i[:minListLen] for i in origUtilList])
    origAvgArr = np.average(origUtilArr, axis=0)
    kInd = next(x for x, val in enumerate(origAvgArr.tolist()) if val > compAvg)
    origSampSaved = round(
        (compAvg - origAvgArr[kInd - 1]) / (origAvgArr[kInd] - origAvgArr[kInd - 1]) * testInt * 3) + (
                            kInd - 1) * testInt * 3
    print(origSampSaved)
    '''20-APR: XXX saved'''

    ###########
    # Distribution of utility for particular budget at particular untested node, per bootstrap samples of Q
    ###########
    sampBudget = 50
    utilDict.update({'method': 'weightsNodeDraw3linear'})
    # New Bayes draws
    setDraws = CSdict3['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]
    lossDict.update({'bayesDraws': setDraws})
    dictTemp = CSdict3.copy()
    dictTemp.update(
        {'postSamples': CSdict3['postSamples'][choice(np.arange(numdraws), size=numtargetdraws, replace=False)],
         'numPostSamples': numtargetdraws})
    tempLossMat = lf.lossMatSetBayesDraws(dictTemp['postSamples'], lossDict.copy(), lossDict['bayesDraws'])
    tempLossDict = lossDict.copy()
    tempLossDict.update({'lossMat': tempLossMat})
    newBayesDraws, newLossMat = lf.add_cand_neighbors(tempLossDict.copy(), CSdict3['postSamples'],
                                                      dictTemp['postSamples'])
    tempLossDict.update({'bayesDraws': newBayesDraws, 'lossMat': newLossMat})
    baseLoss = (np.sum(newLossMat, axis=1) / newLossMat.shape[1]).min()
    # Get a new set of data draws
    utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})

    numReps = 500
    utilListMat = np.zeros((4, numReps))
    for rep in range(103, numReps):
        print('On replication ' + str(rep))
        Qvecs = np.random.multinomial(numBoot, SNprobs, size=numTN - 4) / numBoot
        dictTemp['Q'] = np.vstack((CSdict3['N'][:4] / np.sum(CSdict3['N'][:4], axis=1).reshape(4, 1), Qvecs))

        for tnInd in range(4):
            print('Test node ' + str(tnInd + 4))
            currDes = np.zeros(8)
            currDes[tnInd + 4] = 1.
            currUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[currDes],
                                                numtests=sampBudget, utildict=utilDict)[0]
            utilListMat[tnInd, rep] = currUtil
            print(currUtil)
            if np.mod(rep, 50) == 0 and rep > 0:
                plt.hist(utilListMat[tnInd, :rep + 1], alpha=0.3)
                plt.show()
                plt.close()

                plt.boxplot(np.array(utilListMat[:, :rep]).T)
                plt.ylim([0, 0.33])
                plt.show()
    # np.save('bootstrapUtilDist', np.array(utilListMat))
    # utilListMat = np.load('bootstrapUtilDist.npy')

    return


def example_chain():
    '''
    Code for the example supply chain of the introduction/Section 3.2 of the paper.
    '''

    '''
    1) N = np.array([[8, 7], [9, 6], [2, 13]])
    Y = np.array([[3, 0], [4, 1], [0, 0]])

    2) N = np.array([[10, 5], [9, 6], [2, 13]])
    Y = np.array([[3, 0], [3, 2], [0, 0]]) DEF NOT

    3) N = np.array([[10, 5], [11, 4], [2, 13]])
    Y = np.array([[3, 0], [4, 1], [0, 0]]) DEF NOT

    4) N = np.array([[5, 10], [8, 7], [3, 12]])
    Y = np.array([[2, 0], [4, 1], [0, 0]])

    5) N = np.array([[7, 8], [6, 4], [3, 12]])
    Y = np.array([[3, 0], [3, 1], [0, 0]])

    6) N = np.array([[6, 9], [5, 5], [3, 12]])
    Y = np.array([[3, 0], [4, 1], [0, 0]]) GOOD ONE; USING THIS 12/30/22

    7) N = np.array([[6, 9], [4, 6], [2, 13]])
    Y = np.array([[3, 0], [2, 1], [0, 0]]) NOPE

    8) N = np.array([[5, 10], [5, 5], [3, 12]])
    Y = np.array([[2, 0], [4, 1], [0, 0]]) MAYBE

    9) N = np.array([[8, 7], [5, 5], [3, 12]])
    Y = np.array([[3, 0], [4, 1], [0, 0]])

    '''
    N = np.array([[6, 2], [3, 1], [4, 2], [12, 3]])
    Y = np.array([[4, 0], [0, 0], [0, 0], [4, 1]])
    (numTN, numSN) = N.shape
    s, r = 0.9, 0.95
    scDict = util.generateRandDataDict(numImp=numSN, numOut=numTN, diagSens=s, diagSpec=r,
                                       numSamples=0, dataType='Tracked', randSeed=2)
    scDict['diagSens'], scDict['diagSpec'] = s, r
    scDict = util.GetVectorForms(scDict)
    scDict['N'], scDict['Y'] = N, Y
    scDict['prior'] = methods.prior_normal()
    scDict['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    scDict['SNnum'], scDict['TNnum'] = numSN, numTN
    # Generate posterior draws
    numdraws = 20000  # Evaluate choice here
    scDict['numPostSamples'] = numdraws
    scDict = methods.GeneratePostSamples(scDict)
    util.plotPostSamples(scDict, 'int90')

    # Sourcing matrix
    # todo: EVALUATE HERE
    Q = np.array([[0.4, 0.6], [0.2, 0.8], [0.5, 0.5], [0.7, 0.3]])
    scDict.update({'Q': Q})

    # Put designs here
    design1 = np.array([0., 1., 0., 0.])
    design2 = np.ones(numTN) / numTN
    design3 = np.array([0.5, 0., 0., 0.5])
    desList = [design1, design2, design3]

    testMax, testInt = 40, 4

    # Loss specification
    paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=1., riskthreshold=0.2, riskslope=0.6,
                                                  marketvec=np.ones(numTN + numSN), candneighnum=1000)

    numSetDraws, numtargetdraws, numBayesNeigh, numDataDraws = 6000, 2000, 1000, 1900

    # Utility dictionary
    utilDict = {'method': 'weightsNodeDraw4'}

    # Calculate utility matrix
    dictTemp = scDict.copy()
    dictTemp.update(
        {'postSamples': scDict['postSamples'][choice(np.arange(numdraws), size=numtargetdraws, replace=False)],
         'numPostSamples': numtargetdraws})
    print('Generating loss matrix...')
    setDraws = scDict['postSamples'][choice(np.arange(numdraws), size=numSetDraws, replace=False)]

    lossDict.update({'bayesDraws': setDraws, 'bayesEstNeighborNum': numBayesNeigh})
    # Get a new set of data draws
    utilDict.update({'dataDraws': setDraws[choice(np.arange(len(setDraws)), size=numDataDraws, replace=False)]})

    tempLossMat = lf.lossMatSetBayesDraws(dictTemp['postSamples'], lossDict.copy(), lossDict['bayesDraws'])
    tempLossDict = lossDict.copy()
    tempLossDict.update({'lossMat': tempLossMat})
    newBayesDraws, newLossMat = lf.add_cand_neighbors(tempLossDict.copy(), scDict['postSamples'],
                                                      dictTemp['postSamples'])
    tempLossDict.update({'bayesDraws': newBayesDraws, 'lossMat': newLossMat})
    baseLoss = (np.sum(newLossMat, axis=1) / newLossMat.shape[1]).min()
    # Loop through each design
    utilArrList = []
    for currDes in desList:
        utilArr = np.zeros(int(testMax / testInt))
        print('Design: ' + str(currDes))
        for budgetInd, currBudget in enumerate(np.arange(testInt, testMax + 1, testInt)):
            print('Budget:' + str(currBudget))
            currUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[currDes],
                                                numtests=currBudget, utildict=utilDict)[0]
            print('Utility: ' + str(currUtil))
            utilArr[budgetInd] = currUtil
        utilArrList.append(utilArr)
    '''7-APR run
    utilArrList = 
    '''
    # Add zeros to the beginning
    utilMatPlot = np.concatenate((np.zeros(3).reshape(3, 1), np.array(utilArrList)), axis=1)
    plotMargUtil(utilMatPlot, testMax, testInt, colors=['blue', 'red', 'green'],
                 labels=['Focused', 'Balanced', 'Adapted'], titleStr='$v=1.0$')

    #######################
    # Now for different underEstWt
    underWt = 10
    scoredict = {'name': 'AbsDiff', 'underEstWt': underWt}
    lossDict.update({'scoreDict': scoredict})

    tempLossMat = lf.lossMatSetBayesDraws(dictTemp['postSamples'], lossDict.copy(), lossDict['bayesDraws'])
    tempLossDict = lossDict.copy()
    tempLossDict.update({'lossMat': tempLossMat})
    newBayesDraws, newLossMat = lf.add_cand_neighbors(tempLossDict.copy(), scDict['postSamples'],
                                                      dictTemp['postSamples'])
    tempLossDict.update({'bayesDraws': newBayesDraws, 'lossMat': newLossMat})
    baseLoss = (np.sum(newLossMat, axis=1) / newLossMat.shape[1]).min()
    # Loop through each design
    utilArrList2 = []
    for currDes in desList:
        utilArr = np.zeros(int(testMax / testInt))
        print('Design: ' + str(currDes))
        for budgetInd, currBudget in enumerate(np.arange(testInt, testMax + 1, testInt)):
            print('Budget:' + str(currBudget))
            currUtil = baseLoss - \
                       sampf.sampling_plan_loss(priordatadict=dictTemp, lossdict=tempLossDict, designlist=[currDes],
                                                numtests=currBudget, utildict=utilDict)[0]
            print('Utility: ' + str(currUtil))
            utilArr[budgetInd] = currUtil
        utilArrList2.append(utilArr)
    '''7-APR run
    utilArrList2 = 
    '''
    utilMatPlot = np.concatenate((np.zeros(3).reshape(3, 1), np.array(utilArrList2)), axis=1)
    plotMargUtil(utilMatPlot, testMax, testInt, colors=['blue', 'red', 'green'],
                 labels=['Focused', 'Balanced', 'Adapted'], titleStr='$v=10$')

    return


def syntheticCaseStudy():
    '''

    '''
    numTN, numSN = 25, 25
    numSamples = 200
    s, r = 1.0, 1.0

    Q = np.zeros((numTN, numSN))

    Qrow = np.array([.01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01,
                     .02, .02, .02, .03, .03, .05, .05, .07, .07, .07, .10, .15, .20])
    random.seed(5)
    for TNind in range(numTN):
        random.shuffle(Qrow)
        Q[TNind] = Qrow
    '''
    # Qrow: [0.01, 0.03, 0.1 , 0.02, 0.01, 0.01, 0.07, 0.01, 0.01, 0.02, 0.2, 0.02,
    #        0.01, 0.01, 0.07, 0.15, 0.01, 0.01, 0.03, 0.07, 0.01, 0.01, 0.05, 0.05, 0.01])

    # SN rates: 1% baseline; 20% node: 25%, 5% node: ~25/30%, 7% node: 10%, 2% node: 40%
    # TN rates: 1% baseline; 1 major node: 25%, 1 minor node: 30%; 3 minor nodes: 10%; 1 minor minor node: 50%
    '''

    SNnames = ['Manufacturer ' + str(i + 1) for i in range(numSN)]
    TNnames = ['District ' + str(i + 1) for i in range(numTN)]

    trueRates = np.zeros(numSN + numTN)  # supply nodes first, test nodes second

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
    # print(np.sum(TNsampProbs)) # sampling probability should add up to 1.0

    TNtrueRates = [.02 for i in range(numTN)]  # Update SFP rates for TNs
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
        currSN = random.choices(SNnames, weights=Q[currTNind], k=1)[0]  # [TNnames.index(currTN)] to index Q
        currTNrate = trueRates[numSN + currTNind]
        currSNrate = trueRates[SNnames.index(currSN)]
        realRate = currTNrate + currSNrate - currTNrate * currSNrate
        realResult = np.random.binomial(1, p=realRate)
        if realResult == 1:
            result = np.random.binomial(1, p=s)
        if realResult == 0:
            result = np.random.binomial(1, p=1. - r)
        testingDataList.append([currTN, currSN, result])

    # Initialize needed parameters for the prior and the MCMC sampler
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
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=priorScale), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    numSN, numTN = lgDict['SNnum'], lgDict['TNnum']

    floorVal = 0.05  # Classification lines
    ceilVal = 0.20

    # Supply-node plot
    SNindsSubset = range(numSN)
    SNnames = [lgDict['SNnames'][i] for i in SNindsSubset]
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
    plt.text(26.3, ceilVal + .015, 'u=0.20', color='blue', alpha=0.5, size=9)
    plt.text(26.3, floorVal + .015, 'l=0.05', color='r', alpha=0.5, size=9)
    fig.tight_layout()
    plt.show()
    plt.close()

    # Test-node plot
    TNindsSubset = range(numTN)
    TNnames = [lgDict['TNnames'][i] for i in TNindsSubset]
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
    plt.text(26.4, ceilVal + .015, 'u=0.20', color='blue', alpha=0.5, size=9)
    plt.text(26.4, floorVal + .015, 'l=0.05', color='r', alpha=0.5, size=9)
    fig.tight_layout()
    plt.show()
    plt.close()

    # How many observed arcs are there?
    # np.count_nonzero(lgDict['N'])

    '''
    # Inspect raw data totals
    # Supply nodes
    for i in range(numSN): # sum across TNs to see totals for SNs
        currTotal = np.sum(lgDict['N'],axis=0)[i]
        currPos = np.sum(lgDict['Y'],axis=0)[i]
        print(lgDict['SNnames'][i]+': ' +str(currTotal)[:-2]+' samples, '
              + str(currPos)[:-2] + ' positives, ' + str(currPos/currTotal)[:5] + ' rate')
    # Test nodes
    for i in range(numTN): # sum across SNs to see totals for TNs
        currTotal = np.sum(lgDict['N'],axis=1)[i]
        currPos = np.sum(lgDict['Y'],axis=1)[i]
        print(lgDict['TNnames'][i]+': ' +str(currTotal)[:-2]+' samples, '
              + str(currPos)[:-2] + ' positives, ' + str(currPos/currTotal)[:5] + ' rate')

    # SNs, TNs with at least ten samples and 10% SFP rate
    for i in range(numSN): # sum across TNs to see totals for SNs
        currTotal = np.sum(lgDict['N'],axis=0)[i]
        currPos = np.sum(lgDict['Y'],axis=0)[i]
        if currPos/currTotal>0.1 and currTotal>10:
            print(lgDict['SNnames'][i]+': ' +str(currTotal)[:-2]+' samples, '
              + str(currPos)[:-2] + ' positives, ' + str(currPos/currTotal)[:5] + ' rate')
    # Test nodes
    for i in range(numTN): # sum across SNs to see totals for TNs
        currTotal = np.sum(lgDict['N'],axis=1)[i]
        currPos = np.sum(lgDict['Y'],axis=1)[i]
        if currPos / currTotal > 0.1 and currTotal > 10:
            print(lgDict['TNnames'][i]+': ' +str(currTotal)[:-2]+' samples, '
              + str(currPos)[:-2] + ' positives, ' + str(currPos/currTotal)[:5] + ' rate')

    # 90% intervals for SFP rates at SNs, TNs, using proportion CI
    for i in range(numSN):  # sum across TNs to see totals for SNs
        currTotal = np.sum(lgDict['N'], axis=0)[i]
        currPos = np.sum(lgDict['Y'], axis=0)[i]
        pHat = currPos/currTotal
        lowerBd = pHat-(1.645*np.sqrt(pHat*(1-pHat)/currTotal))
        upperBd = pHat+(1.645*np.sqrt(pHat*(1-pHat)/currTotal))
        print(lgDict['SNnames'][i]+': ('+str(lowerBd)[:5]+', '+str(upperBd)[:5]+')')
    # Test nodes
    for i in range(numTN):  # sum across SNs to see totals for TNs
        currTotal = np.sum(lgDict['N'], axis=1)[i]
        currPos = np.sum(lgDict['Y'], axis=1)[i]
        pHat = currPos / currTotal
        lowerBd = pHat - (1.645 * np.sqrt(pHat * (1 - pHat) / currTotal))
        upperBd = pHat + (1.645 * np.sqrt(pHat * (1 - pHat) / currTotal))
        print(lgDict['TNnames'][i] + ': (' + str(lowerBd)[:5] + ', ' + str(upperBd)[:5] + ')')


    # Print quantiles for analysis tables
    SNinds = lgDict['SNnames'].index('Manufacturer 4')
    print('Manufacturer 4: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    SNinds = lgDict['SNnames'].index('Manufacturer 11')
    print('Manufacturer 11: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    SNinds = lgDict['SNnames'].index('Manufacturer 23')
    print('Manufacturer 23: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    TNinds = lgDict['TNnames'].index('District 6')
    print('District 6: (' + str(np.quantile(lgDict['postSamples'][:, len(lgDict['SNnames']) + TNinds], 0.05))[
        :5] + ',' + str(np.quantile(lgDict['postSamples'][:, len(lgDict['SNnames']) + TNinds], 0.95))[:5] + ')')
    TNinds = lgDict['TNnames'].index('District 7')
    print('District 7: (' + str(np.quantile(lgDict['postSamples'][:, len(lgDict['SNnames']) + TNinds], 0.05))[
        :5] + ',' + str(np.quantile(lgDict['postSamples'][:, len(lgDict['SNnames']) + TNinds], 0.95))[:5] + ')')

    # Print sourcing probability matrix

    '''

    return


def STUDYMCMCmetrics():
    '''Use Vehtari (2021) to establish metrics for our MCMC chains'''

    # Define MCMC generator that takes different initial point
    def GenerateMCMCdraws(dataTblDict):
        '''
        Retrives posterior samples under the appropriate Tracked or Untracked
        likelihood model, given data inputs, and entered posterior sampler.

        INPUTS
        ------
        dataTblDict is an input dictionary with the following keys:
            N,Y: Number of tests, number of positive tests on each test node-supply node
                 track (Tracked) or test node (Untracked)
            diagSens,diagSpec: Diagnostic sensitivity and specificity
            prior: Prior distribution object with lpdf,lpdf_jac methods
            numPostSamples: Number of posterior distribution samples to generate
            MCMCDict: Dictionary for the desired MCMC sampler to use for generating
            posterior samples; requies a key 'MCMCType' that is one of
            'MetropolisHastings', 'Langevin', 'NUTS', or 'STAN'; necessary arguments
            for the sampler should be contained as keys within MCMCDict
        OUTPUTS
        -------
        Returns dataTblDict with key 'postSamples' that contains the non-transformed
        poor-quality likelihoods.
        '''
        # change utilities to nuts if wanting to use the Gelman sampler
        import logistigate.logistigate.mcmcsamplers.adjustedNUTS as adjnuts
        if not all(key in dataTblDict for key in ['type', 'N', 'Y', 'diagSens', 'diagSpec',
                                                  'MCMCdict', 'prior', 'numPostSamples']):
            print('The input dictionary does not contain all required information for generating posterior samples.' +
                  ' Please check and try again.')
            return {}

        print('Generating posterior samples...')

        N, Y = dataTblDict['N'], dataTblDict['Y']
        sens, spec = dataTblDict['diagSens'], dataTblDict['diagSpec']

        MCMCdict = dataTblDict['MCMCdict']

        startTime = time.time()
        # Run NUTS (Hoffman & Gelman, 2011)
        prior, M = dataTblDict['prior'], dataTblDict['numPostSamples']
        Madapt, delta = MCMCdict['Madapt'], MCMCdict['delta']
        # Initial point for sampler
        if not 'initBeta' in dataTblDict['MCMCdict'].keys():
            beta0 = -2 * np.ones(N.shape[1] + N.shape[0])
        else:
            beta0 = dataTblDict['MCMCdict']['initBeta'].copy()

        def TargetForNUTS(beta):
            return methods.Tracked_LogPost(beta, N, Y, sens, spec, prior), \
                   methods.Tracked_LogPost_Grad(beta, N, Y, sens, spec, prior)

        samples, lnprob, epsilon = adjnuts.nuts6(TargetForNUTS, M, Madapt, beta0, delta)

        dataTblDict.update({'acc_rate': 'NA'})  # FIX LATER
        # Transform samples back
        postSamples = sps.expit(samples)

        # Record generation time
        endTime = time.time()

        dataTblDict.update({'postSamples': postSamples,
                            'postSamplesGenTime': endTime - startTime})
        print('Posterior samples generated')
        return dataTblDict

    # Use Exploratory case study setting
    rd3_N = np.array([[1., 1., 10., 1., 3., 0., 1., 6., 7., 5., 0., 0., 4.],
                      [1., 1., 4., 2., 0., 1., 1., 2., 0., 4., 0., 0., 1.],
                      [3., 17., 31., 4., 2., 0., 1., 6., 0., 23., 1., 2., 5.],
                      [1., 1., 15., 2., 0., 0., 0., 1., 0., 6., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    rd3_Y = np.array([[0., 0., 7., 0., 3., 0., 1., 0., 1., 0., 0., 0., 4.],
                      [0., 0., 2., 2., 0., 1., 1., 0., 0., 1., 0., 0., 1.],
                      [0., 0., 15., 3., 2., 0., 0., 2., 0., 1., 1., 2., 5.],
                      [0., 0., 5., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

    (numTN, numSN) = rd3_N.shape
    s, r = 1., 1.
    CSdict3 = util.generateRandDataDict(numImp=numSN, numOut=numTN, diagSens=s, diagSpec=r,
                                        numSamples=0, dataType='Tracked', randSeed=2)
    CSdict3['diagSens'], CSdict3['diagSpec'] = s, r
    CSdict3 = util.GetVectorForms(CSdict3)
    CSdict3['N'], CSdict3['Y'] = rd3_N, rd3_Y

    SNpriorMean = np.repeat(sps.logit(0.1), numSN)
    # Establish test nodes according to assessment by regulators
    TNpriorMean = sps.logit(np.array([0.1, 0.1, 0.15, 0.15, 0.15, 0.1, 0.15, 0.1]))
    TNvar, SNvar = 2., 4.
    CSdict3['prior'] = prior_normal_assort(np.concatenate((SNpriorMean, TNpriorMean)),
                                   np.diag(np.concatenate((np.repeat(SNvar, numSN), np.repeat(TNvar, numTN)))))

    ##### REMOVE LATER
    # CSdict3['TNnames'] = ['ASHANTI', 'BRONG AHAFO', 'CENTRAL', 'EASTERN REGION', 'GREATER ACCRA', 'NORTHERN SECTOR', 'VOLTA', 'WESTERN']
    CSdict3['TNnames'] = ['MOD_39', 'MOD_17', 'MODHIGH_95', 'MODHIGH_26',
                          'MODHIGH_EXPL_1', 'MOD_EXPL_1', 'MODHIGH_EXPL_2', 'MOD_EXPL_2']
    CSdict3['SNnames'] = ['ACME FORMULATION PVT. LTD.', 'AS GRINDEKS', 'BELCO PHARMA', 'BHARAT PARENTERALS LTD',
                          'HUBEI TIANYAO PHARMACEUTICALS CO LTD.', 'MACIN REMEDIES INDIA LTD',
                          'NORTH CHINA PHARMACEUTICAL CO. LTD', 'NOVARTIS PHARMA', 'PFIZER',
                          'PIRAMAL HEALTHCARE UK LIMITED', 'PUSHKAR PHARMA',
                          'SHANDOND SHENGLU PHARMACEUTICAL CO.LTD.', 'SHANXI SHUGUANG PHARM']

    # Region catchment proportions
    TNcach = np.array([0.17646, 0.05752, 0.09275, 0.09488, 0.17695, 0.22799, 0.07805, 0.0954])
    tempQ = CSdict3['N'][:4] / np.sum(CSdict3['N'][:4], axis=1).reshape(4, 1)
    tempTNcach = TNcach[:4] / np.sum(TNcach[:4])
    SNcach = np.matmul(tempTNcach, tempQ)
    # Normalize market weights s.t. sum of TN terms equals sum of SN terms equals number of TNs
    # TNcach = TNcach * TNcach.shape[0] / TNcach.sum()
    # SNcach = SNcach * TNcach.sum() / SNcach.sum()
    ###################

    CSdict3['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    CSdict3['SNnum'], CSdict3['TNnum'] = numSN, numTN
    # Generate posterior draws
    numdraws = 100000  # Evaluate choice here
    CSdict3['numPostSamples'] = numdraws
    CSdict3 = GenerateMCMCdraws(CSdict3)

    # Establish metrics per Vehtari (2021)
    M = 10  # Number of chains
    N = numdraws  # to shorten expressions
    # Sample initial points from prior
    np.random.seed(10)
    newBetaArr = CSdict3['prior'].rand(M)
    # Generate chains from initial points
    CSdict3['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4, 'initBeta': newBetaArr[0]}
    CSdict3 = GenerateMCMCdraws(CSdict3)
    chainArr = CSdict3['postSamples']
    chainArr = chainArr.reshape((1, N, numSN + numTN))
    # Generate new chains with different initial points
    for m in range(1, M):
        CSdict3['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4, 'initBeta': newBetaArr[m]}
        CSdict3 = GenerateMCMCdraws(CSdict3)
        chainArr = np.concatenate((chainArr, CSdict3['postSamples'].reshape((1, N, numSN + numTN))))
    # Save chainArr object for working with later
    # np.save('chainArr.npy', chainArr)
    # chainArr = np.load('chainArr.npy')
    # Calculate statistics for mean of each SN/TN SFP rate; per Gelman (1992)
    potentScaleRarr = np.empty((numSN + numTN))
    for k in range(numSN + numTN):
        # Look at mean
        seqMeans = np.average(chainArr[:, :, k], axis=1)
        seqMeansMean = np.average(seqMeans)
        Bn = np.sum((seqMeans - seqMeansMean) ** 2) / (M - 1)  # across-chains variance
        seqVars = np.var(chainArr[:, :, k], axis=1, ddof=1)  # within-chains variances
        W = np.average(seqVars)  # avg within-chains variance
        seqVar = ((N - 1) / N) * W + Bn  # target variance
        tMean = seqMeansMean  # targe mean
        # allow for sampling variability of mean/var
        tScale = np.sqrt(seqVar + Bn / M)
        cov1term = np.cov((seqVars, seqMeans ** 2))[0, 1]
        cov2term = np.cov((seqVars, seqMeans))[0, 1]
        varV = (((N - 1) / N) ** 2) * (1 / M) * np.var(seqVars, ddof=1) + \
               (((M + 1) / (M * N)) ** 2) * (2 / (M - 1)) * ((Bn * N) ** 2) + \
               2 * ((M + 1) * (N - 1) / (M * N * N)) * (N / M) * (cov1term - 2 * seqMeansMean * cov2term)
        df = 2 * ((tScale ** 2) ** 2) / varV
        # potential scale factor
        rootR = np.sqrt(((tScale ** 2) / W) * (df / (df - 2)))
        potentScaleRarr[k] = rootR
    # histogram
    plt.hist(potentScaleRarr)
    plt.show()
    plt.close()
    # Calculate split-R; per Gelman (2013)
    splitRarr = np.empty((numSN + numTN))
    # Split chains in half to ascertain non-stationarity
    splitChainArr = np.concatenate((chainArr[:, :int(numdraws / 2), :], chainArr[:, int(numdraws / 2):, :]))
    for k in range(numSN + numTN):
        # Look at mean
        seqMeans = np.average(splitChainArr[:, :, k], axis=1)
        seqMeansMean = np.average(seqMeans)
        B = np.sum((seqMeans - seqMeansMean) ** 2) * N / (M - 1)  # across-chains variance
        seqVars = np.var(splitChainArr[:, :, k], axis=1, ddof=1)  # within-chains variances
        W = np.average(seqVars)  # avg within-chains variance
        seqVar = ((N - 1) / N) * W + B / N  # target variance
        rootR = np.sqrt(seqVar / W)
        splitRarr[k] = rootR
    # histogram
    plt.hist(splitRarr)
    plt.show()
    plt.close()
    # Calculate split-R for different chain sizes
    for chainSize in [500, 1000, 5000, 10000, 50000, 100000]:
        N = chainSize
        splitRarr = np.empty((numSN + numTN))
        # Split chains in half to ascertain non-stationarity
        splitChainArr = np.concatenate((chainArr[:, :int(N / 2), :], chainArr[:, int(N / 2):N, :]))
        for k in range(numSN + numTN):
            # Look at mean
            seqMeans = np.average(splitChainArr[:, :, k], axis=1)
            seqMeansMean = np.average(seqMeans)
            B = np.sum((seqMeans - seqMeansMean) ** 2) * N / (M - 1)  # across-chains variance
            seqVars = np.var(splitChainArr[:, :, k], axis=1, ddof=1)  # within-chains variances
            W = np.average(seqVars)  # avg within-chains variance
            seqVar = ((N - 1) / N) * W + B / N  # target variance
            rootR = np.sqrt(seqVar / W)
            splitRarr[k] = rootR
        # histogram
        plt.hist(splitRarr, label=str(N))
    plt.legend()
    plt.title('Split-$\hat{R}$ for SFP-rate means, under different chain sizes\n10 chains')
    plt.show()
    plt.close()
    # ESS
    # First inspect autocorrelation plot
    lagNum = 50  # Set desired lag truncation amount here
    autocorrArr = np.empty((lagNum, M, numSN + numTN))  # 100 lags for M chains for all SFP rates
    for m in range(M):
        for k in range(numSN + numTN):
            # Nearest size with power of 2
            data = chainArr[m, :, k]
            size = 2 ** np.ceil(np.log2(2 * len(data) - 1)).astype('int')
            ndata = data - np.mean(data)  # Normalized data
            fft = np.fft.fft(ndata, size)  # Compute the FFT
            pwr = np.abs(fft) ** 2  # Get the power spectrum
            # Calculate the autocorrelation from inverse FFT of the power spectrum
            acorr = np.fft.ifft(pwr).real / np.var(data) / len(data)
            autocorrArr[:, m, k] = acorr[1:lagNum + 1]  # skip lag=0
    for k in range(numSN + numTN):
        for m in range(M):
            plt.bar(np.arange(1, lagNum), autocorrArr[lagNum, m, k], alpha=0.1)
        plt.ylim([-0.1, 0.5])
        plt.show()
        plt.close()
    # Get combined autocorrletaion at each lag
    # Skip lag=0
    combAutocorrArr = np.empty((lagNum, numSN + numTN))
    for k in range(numSN + numTN):
        seqMeans = np.average(chainArr[:, :, k], axis=1)
        seqMeansMean = np.average(seqMeans)
        B = np.sum((seqMeans - seqMeansMean) ** 2) * N / (M - 1)  # across-chains variance
        seqVars = np.var(chainArr[:, :, k], axis=1, ddof=1)  # within-chains variances
        W = np.average(seqVars)  # avg within-chains variance
        seqVar = ((N - 1) / N) * W + B / N  # target variance
        for t in range(lagNum):
            combAutocorrArr[t, k] = 1 - ((W - np.average(seqVars * autocorrArr[t, :, k])) / seqVar)
    # Plot for each SFP rate
    for k in range(numSN + numTN):
        plt.plot(combAutocorrArr[:, k], alpha=0.2)
    plt.title('Combined autocorrelations at each node')
    plt.show()
    plt.close()
    # Get ESS at each node
    Seff = N * M / (1 + 2 * np.sum(combAutocorrArr, axis=0))
    plt.hist(Seff)
    plt.title('Histogram of effective sample sizes at each node (21 nodes)\n10 MCMC chains of 100,000 draws each')
    plt.show()
    plt.close()
    #######
    # Redo Seff calculations after transforming chains to 0.05 marginal quantiles
    alph = 0.05
    quantArr = np.empty((numSN + numTN))
    chain05Arr = np.empty(chainArr.shape)
    for k in range(numSN + numTN):
        quantArr[k] = np.quantile(chainArr[:, :, k], alph)
        chain05Arr[:, :, k] = (chainArr[:, :, k] > quantArr[k]).astype(int)
    # First inspect autocorrelation plot
    lagNum = 20  # Set desired lag truncation amount here
    autocorrArr = np.empty((lagNum, M, numSN + numTN))  # 20 lags for M chains for all SFP rates
    for m in range(M):
        for k in range(numSN + numTN):
            # Nearest size with power of 2
            data = chain05Arr[m, :, k]
            size = 2 ** np.ceil(np.log2(2 * len(data) - 1)).astype('int')
            ndata = data - np.mean(data)  # Normalized data
            fft = np.fft.fft(ndata, size)  # Compute the FFT
            pwr = np.abs(fft) ** 2  # Get the power spectrum
            # Calculate the autocorrelation from inverse FFT of the power spectrum
            acorr = np.fft.ifft(pwr).real / np.var(data) / len(data)
            autocorrArr[:, m, k] = acorr[1:lagNum + 1]  # skip lag=0
    for k in range(numSN + numTN):
        for m in range(M):
            plt.bar(np.arange(lagNum), autocorrArr[:, m, k], alpha=0.1)
        plt.ylim([-0.1, 0.5])
        plt.show()
        plt.close()
    # Get combined autocorrletaion at each lag
    # Skip lag=0
    combAutocorrArr = np.empty((lagNum, numSN + numTN))
    for k in range(numSN + numTN):
        seqMeans = np.average(chain05Arr[:, :, k], axis=1)
        seqMeansMean = np.average(seqMeans)
        B = np.sum((seqMeans - seqMeansMean) ** 2) * N / (M - 1)  # across-chains variance
        seqVars = np.var(chain05Arr[:, :, k], axis=1, ddof=1)  # within-chains variances
        W = np.average(seqVars)  # avg within-chains variance
        seqVar = ((N - 1) / N) * W + B / N  # target variance
        for t in range(lagNum):
            combAutocorrArr[t, k] = 1 - ((W - np.average(seqVars * autocorrArr[t, :, k])) / seqVar)
    # Plot for each SFP rate
    for k in range(numSN + numTN):
        plt.plot(combAutocorrArr[:, k], alpha=0.2)
    plt.title('Combined autocorrelations at each node')
    plt.show()
    plt.close()
    # Get ESS at each node
    Seff = N * M / (1 + 2 * np.sum(combAutocorrArr, axis=0))
    plt.hist(Seff)
    plt.title('Histogram of effective sample sizes for $.05$ quantiles\n10 MCMC chains of 100,000 draws each')
    plt.show()
    plt.close()

    # Do for all quantiles, at node 6 (7)
    prec = 100  # precision
    quantileArr = np.arange(1, prec + 1) / prec
    k = 6
    quantArr = np.quantile(chainArr[:, :, k], quantileArr)
    SeffArr = np.empty(prec)
    for qind, q in enumerate(range(quantileArr.shape[0])):
        print('On quantile: ' + str(q))
        chainQArr = (chainArr[:, :, k] > quantArr[q]).astype(int)
        # First inspect autocorrelation plot
        lagNum = 20  # Set desired lag truncation amount here
        autocorrArr = np.empty((lagNum, M))  # 20 lags for M chains for all SFP rates
        for m in range(M):
            # Nearest size with power of 2
            data = chainQArr[m, :]
            size = 2 ** np.ceil(np.log2(2 * len(data) - 1)).astype('int')
            ndata = data - np.mean(data)  # Normalized data
            fft = np.fft.fft(ndata, size)  # Compute the FFT
            pwr = np.abs(fft) ** 2  # Get the power spectrum
            # Calculate the autocorrelation from inverse FFT of the power spectrum
            acorr = np.fft.ifft(pwr).real / np.var(data) / len(data)
            autocorrArr[:, m] = acorr[1:lagNum + 1]  # skip lag=0
        # Get combined autocorrletaion at each lag
        # Skip lag=0
        combAutocorrArr = np.empty((lagNum))
        seqMeans = np.average(chainQArr, axis=1)
        seqMeansMean = np.average(seqMeans)
        B = np.sum((seqMeans - seqMeansMean) ** 2) * N / (M - 1)  # across-chains variance
        seqVars = np.var(chainQArr, axis=1, ddof=1)  # within-chains variances
        W = np.average(seqVars)  # avg within-chains variance
        seqVar = ((N - 1) / N) * W + B / N  # target variance
        for t in range(lagNum):
            combAutocorrArr[t] = 1 - ((W - np.average(seqVars * autocorrArr[t, :])) / seqVar)
        # Get ESS at each node
        SeffArr[qind] = N * M / (1 + 2 * np.sum(combAutocorrArr, axis=0))
    plt.plot(quantileArr, SeffArr, '.')
    plt.title('ESS for quantiles at Node 6')
    plt.show()
    plt.close()

    # Redo Seff calculations after transforming chains to 0.05 marginal quantiles
    alph = 0.95
    quantArr = np.empty((numSN + numTN))
    chain95Arr = np.empty(chainArr.shape)
    for k in range(numSN + numTN):
        quantArr[k] = np.quantile(chainArr[:, :, k], alph)
        chain95Arr[:, :, k] = (chainArr[:, :, k] > quantArr[k]).astype(int)
    # First inspect autocorrelation plot
    lagNum = 20  # Set desired lag truncation amount here
    autocorrArr = np.empty((lagNum, M, numSN + numTN))  # 20 lags for M chains for all SFP rates
    for m in range(M):
        for k in range(numSN + numTN):
            # Nearest size with power of 2
            data = chain95Arr[m, :, k]
            size = 2 ** np.ceil(np.log2(2 * len(data) - 1)).astype('int')
            ndata = data - np.mean(data)  # Normalized data
            fft = np.fft.fft(ndata, size)  # Compute the FFT
            pwr = np.abs(fft) ** 2  # Get the power spectrum
            # Calculate the autocorrelation from inverse FFT of the power spectrum
            acorr = np.fft.ifft(pwr).real / np.var(data) / len(data)
            autocorrArr[:, m, k] = acorr[1:lagNum + 1]  # skip lag=0
    for k in range(numSN + numTN):
        for m in range(M):
            plt.bar(np.arange(1, lagNum + 1), autocorrArr[:, m, k], alpha=0.1)
        plt.ylim([-0.1, 0.5])
        plt.show()
        plt.close()
    # Get combined autocorrletaion at each lag
    # Skip lag=0
    combAutocorrArr = np.empty((lagNum, numSN + numTN))
    for k in range(numSN + numTN):
        seqMeans = np.average(chain95Arr[:, :, k], axis=1)
        seqMeansMean = np.average(seqMeans)
        B = np.sum((seqMeans - seqMeansMean) ** 2) * N / (M - 1)  # across-chains variance
        seqVars = np.var(chain95Arr[:, :, k], axis=1, ddof=1)  # within-chains variances
        W = np.average(seqVars)  # avg within-chains variance
        seqVar = ((N - 1) / N) * W + B / N  # target variance
        for t in range(lagNum):
            combAutocorrArr[t, k] = 1 - ((W - np.average(seqVars * autocorrArr[t, :, k])) / seqVar)
    # Plot for each SFP rate
    for k in range(numSN + numTN):
        plt.plot(combAutocorrArr[:, k], alpha=0.2)
    plt.title('Combined autocorrelations at each node')
    plt.show()
    plt.close()
    # Get ESS at each node
    Seff = N * M / (1 + 2 * np.sum(combAutocorrArr, axis=0))
    plt.hist(Seff)
    plt.title(
        'Histogram of effective sample sizes for $' + str(alph) + '$ quantiles\n10 MCMC chains of 100,000 draws each')
    plt.show()
    plt.close()

    # What are our ESS for each single chain?
    lagNum = 50
    autocorrArr = np.empty((lagNum, M, numSN + numTN))  # 100 lags for M chains for all SFP rates
    for m in range(M):
        for k in range(numSN + numTN):
            # Nearest size with power of 2
            data = chainArr[m, :, k]
            size = 2 ** np.ceil(np.log2(2 * len(data) - 1)).astype('int')
            ndata = data - np.mean(data)  # Normalized data
            fft = np.fft.fft(ndata, size)  # Compute the FFT
            pwr = np.abs(fft) ** 2  # Get the power spectrum
            # Calculate the autocorrelation from inverse FFT of the power spectrum
            acorr = np.fft.ifft(pwr).real / np.var(data) / len(data)
            autocorrArr[:, m, k] = acorr[1:lagNum + 1]  # skip lag=0
    # Inspect correlation estimates
    for k in range(numSN + numTN):
        for m in range(M):
            plt.plot(np.arange(1, lagNum + 1), autocorrArr[:, m, k], alpha=0.1)
    plt.ylim([-0.1, 0.5])
    plt.title('Autocorrelation up to lag 50 for all nodes and chains (210 total)')
    plt.show()
    plt.close()

    NeffArr = N / (1 + 2 * np.sum(autocorrArr, axis=0))
    # Plot
    plt.hist(NeffArr.flatten())
    plt.title(
        'Histogram of $N_{eff}$ using lag of ' + str(lagNum) + '\n10 MCMC chains of 100,000 draws each')
    plt.show()
    plt.close()

    # Rank plots
    prec = 100  # precision
    quantileArr = np.arange(1, prec + 1) / prec
    for k in range(numSN + numTN):
        # Get quantiles
        currQuantiles = np.quantile(chainArr[:, :, k], quantileArr)
        fig, axs = plt.subplots(M)
        fig.suptitle('Rank plots for Node ' + str(k))
        for m in range(M):
            axs[m].hist(chainArr[m, :, k], bins=currQuantiles)
        plt.show()
        plt.close()
    # Plot ESS as draws increase
    lagNum = 400
    drawsSet = [500, 1000, 2000]
    NeffArrArr = np.empty((len(drawsSet), M, numSN + numTN))
    for drawsind, draws in enumerate(drawsSet):
        print('On draws of ' + str(draws) + '...')
        autocorrArr = np.empty((lagNum, M, numSN + numTN))  # 100 lags for M chains for all SFP rates
        for m in range(M):
            for k in range(numSN + numTN):
                # Nearest size with power of 2
                data = chainArr[m, :draws, k]
                size = 2 ** np.ceil(np.log2(2 * len(data) - 1)).astype('int')
                ndata = data - np.mean(data)  # Normalized data
                fft = np.fft.fft(ndata, size)  # Compute the FFT
                pwr = np.abs(fft) ** 2  # Get the power spectrum
                # Calculate the autocorrelation from inverse FFT of the power spectrum
                acorr = np.fft.ifft(pwr).real / np.var(data) / len(data)
                autocorrArr[:, m, k] = acorr[1:lagNum + 1]  # skip lag=0
        NeffArrArr[drawsind, :, :] = N / (1 + 2 * np.sum(autocorrArr, axis=0))
    # Plot for chain and node
    colors = cm.rainbow(np.linspace(0, 1., M))
    for m in range(M):
        for k in range(numSN + numTN):
            plt.plot(drawsSet, NeffArrArr[:, m, k], marker="o", linestyle="-",
                     lw=0.5, color=colors[m])
    plt.title('ESS vs. number of draws\nFor each chain and node')
    plt.show()
    plt.close()

    return
