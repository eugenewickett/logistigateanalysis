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

# 23-MAY-23
def STUDY_baselineloss():
    """Run the Familiar setting under 5k-5k, 10k-5k, and 11k-5k appraoches and get baseline loss estimates"""
    # PROVINCES-MANUFACTURERS; familiar SETTING
    Nfam = np.array([[1., 1., 10., 1., 3., 0., 1., 6., 7., 5., 0., 0., 4.],
                     [1., 1., 4., 2., 0., 1., 1., 2., 0., 4., 0., 0., 1.],
                     [3., 17., 31., 4., 2., 0., 1., 6., 0., 23., 1., 2., 5.],
                     [1., 1., 15., 2., 0., 0., 0., 1., 0., 6., 0., 0., 0.]])
    Yfam = np.array([[0., 0., 7., 0., 3., 0., 1., 0., 1., 0., 0., 0., 4.],
                     [0., 0., 2., 2., 0., 1., 1., 0., 0., 1., 0., 0., 1.],
                     [0., 0., 15., 3., 2., 0., 0., 2., 0., 1., 1., 2., 5.],
                     [0., 0., 5., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

    (numTN, numSN) = Nfam.shape  # For later use
    csdict_fam = util.initDataDict(Nfam, Yfam)  # Initialize necessary logistigate keys

    # Update node names
    csdict_fam['TNnames'] = ['MOD_39', 'MOD_17', 'MODHIGH_95', 'MODHIGH_26']
    csdict_fam['SNnames'] = ['MNFR ' + str(i + 1) for i in range(numSN)]

    # Build prior; establish test node risk according to assessment by regulators
    SNpriorMean = np.repeat(sps.logit(0.1), numSN)
    TNpriorMean = sps.logit(np.array([0.1, 0.1, 0.15, 0.15]))
    TNvar, SNvar = 2., 4.  # Variances for use with prior
    csdict_fam['prior'] = prior_normal_assort(np.concatenate((SNpriorMean, TNpriorMean)),
                                              np.diag(
                                                  np.concatenate((np.repeat(SNvar, numSN), np.repeat(TNvar, numTN)))))

    # Set up MCMC
    csdict_fam['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    # Generate posterior draws
    numdraws = 80000
    csdict_fam['numPostSamples'] = numdraws
    np.random.seed(1000)  # To replicate draws later
    # csdict_fam = methods.GeneratePostSamples(csdict_fam)

    # Loss specification
    paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                                  marketvec=np.ones(numTN + numSN), candneighnum=1000)

    # List for storing baseline loss values
    list_5k = []
    list_6k = []
    list_10k = []
    list_11k = []

    numtruthdraws = 5000  # This stays constant

    numReps = 100
    for rep in range(numReps):
        print('Rep: ' + str(rep))
        csdict_fam = methods.GeneratePostSamples(csdict_fam)  # Get new MCMC draws
        # Get 5k estimate first
        numbigcanddraws, numcanddraws = 10000, 5000
        bigcanddraws = csdict_fam['postSamples'][choice(np.arange(numdraws), size=numbigcanddraws, replace=False)]
        currcanddraws = bigcanddraws[choice(np.arange(numbigcanddraws), size=numcanddraws, replace=False)]
        currtruthdraws = currcanddraws[choice(np.arange(numcanddraws), size=numtruthdraws, replace=False)]
        paramdict.update({'canddraws': currcanddraws, 'truthdraws': currtruthdraws})
        # Build loss matrix
        lossmatrix = lf.build_loss_matrix(currtruthdraws, currcanddraws, paramdict)
        list_5k.append(sampf.baseloss(lossmatrix))
        # Add neighbors to best candidate
        paramdict.update({'lossmatrix': lossmatrix})
        _, lossmatrix = lf.add_cand_neighbors(paramdict, csdict_fam['postSamples'], currtruthdraws)
        list_6k.append(sampf.baseloss(lossmatrix))
        # Do same for larger set of candidates
        paramdict.update({'canddraws': bigcanddraws})
        lossmatrix = lf.build_loss_matrix(currtruthdraws, bigcanddraws, paramdict)
        list_10k.append(sampf.baseloss(lossmatrix))
        # Add neighbors
        paramdict.update({'lossmatrix': lossmatrix})
        _, lossmatrix = lf.add_cand_neighbors(paramdict, csdict_fam['postSamples'], currtruthdraws)
        list_11k.append(sampf.baseloss(lossmatrix))

    # Plot distribution of gap from 5k
    list_6k_diffs = [list_5k[i] - list_6k[i] for i in range(len(list_5k))]
    list_10k_diffs = [list_5k[i] - list_10k[i] for i in range(len(list_5k))]
    list_11k_diffs = [list_5k[i] - list_11k[i] for i in range(len(list_5k))]

    np.save(os.path.join('', '17MAY_plots', 'list_5k'), np.array(list_5k))
    np.save(os.path.join('', '17MAY_plots', 'list_6k'), np.array(list_6k))
    np.save(os.path.join('', '17MAY_plots', 'list_10k'), np.array(list_10k))
    np.save(os.path.join('', '17MAY_plots', 'list_11k'), np.array(list_11k))

    plt.hist([list_6k_diffs, list_10k_diffs, list_11k_diffs], label=['5k+1k', '10k', '10k+1k'], alpha=0.7)
    plt.title('Histogram of loss decreases as compared with 5k-sized candidate set')
    plt.legend()
    plt.show()
    plt.close()

    plt.hist([list_6k, list_10k, list_11k, list_5k], label=['5k+1k', '10k', '10k+1k', '5k'], alpha=0.7, bins=20)
    plt.title('Histogram of null loss values under different candidate set sizes')
    plt.legend()
    plt.show()
    plt.close()

    return
