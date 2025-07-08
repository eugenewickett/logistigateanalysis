'''
This script includes analysis of the chosen prior and completes a prior predictive check to ensure that
our prior is reasonably aligned with observed data and can yield such data.
'''
from logistigate.logistigate import utilities as util # Pull from the submodule "develop" branch
from logistigate.logistigate import methods
from logistigate.logistigate.priors import prior_normal_assort
from logistigate.logistigate import lossfunctions as lf
from logistigate.logistigate import samplingplanfunctions as sampf

import os
import numpy as np
import matplotlib.cm as cm
import matplotlib
import matplotlib.pyplot as plt

from numpy.random import choice
import scipy.special as sps

# Set up initial data
Nfam = np.array([[1., 1., 10., 1., 3., 0., 1., 6., 7., 5., 0., 0., 4.],
                      [1., 1., 4., 2., 0., 1., 1., 2., 0., 4., 0., 0., 1.],
                      [3., 17., 31., 4., 2., 0., 1., 6., 0., 23., 1., 2., 5.],
                      [1., 1., 15., 2., 0., 0., 0., 1., 0., 6., 0., 0., 0.]])
Yfam = np.array([[0., 0., 7., 0., 3., 0., 1., 0., 1., 0., 0., 0., 4.],
                      [0., 0., 2., 2., 0., 1., 1., 0., 0., 1., 0., 0., 1.],
                      [0., 0., 15., 3., 2., 0., 0., 2., 0., 1., 1., 2., 5.],
                      [0., 0., 5., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
(numTN, numSN) = Nfam.shape # For later use
csdict_existing = util.initDataDict(Nfam, Yfam) # Initialize necessary logistigate keys
csdict_existing['TNnames'] = ['MOD_39', 'MOD_17', 'MODHIGH_95', 'MODHIGH_26']
csdict_existing['SNnames'] = ['MNFR ' + str(i + 1) for i in range(numSN)]

# Some summaries
TNtesttotals = np.sum(Nfam, axis=1)
TNsfptotals = np.sum(Yfam, axis=1)
TNrates = np.divide(TNsfptotals,TNtesttotals)
print('Tests at each test node:')
print(TNtesttotals)
print('Positives at each test node:')
print(TNsfptotals)
print('Positive rates at each test node:')
print(TNrates)

# Build prior
SNpriorMean = np.repeat(sps.logit(0.1), numSN)
# Establish test node priors according to assessment by regulators
TNpriorMean = sps.logit(np.array([0.1, 0.1, 0.15, 0.15]))
priorMean = np.concatenate((SNpriorMean, TNpriorMean))
TNvar, SNvar = 2., 4.  # Variances for use with prior; supply nodes are wide due to large
priorCovar = np.diag(np.concatenate((np.repeat(SNvar, numSN), np.repeat(TNvar, numTN))))
priorObj = prior_normal_assort(priorMean, priorCovar)
csdict_existing['prior'] = priorObj

# Prior predictive check: simulate data using our prior for traces observed in N,
# and check against positives observed in Y

nsim = 1000  # Number of simulated data sets
priorsim = priorObj.rand(nsim)
simlist = []
Nlist = []
Ylist = []
for tn in range(Nfam.shape[0]):
    for sn in range(Nfam.shape[1]):
        if Nfam[tn, sn] > 0:
            currN = Nfam[tn, sn]
            currConsolRateVec = sps.expit(priorsim[:, sn]) + sps.expit(priorsim[:,numSN+tn]) -\
                                sps.expit(priorsim[:, sn])*sps.expit(priorsim[:,numSN+tn])
            # Generate Y data from prior and N at trace
            Ysim = np.random.binomial(currN, currConsolRateVec)
            simlist.append(Ysim)
            Ylist.append(Yfam[tn, sn])
            Nlist.append(currN)

Nlist_sort, Ylist_sort, simlist_sort = zip(*sorted(zip(Nlist, Ylist, simlist), key=lambda x: x[0]))

fig, axs = plt.subplots(4, 9, figsize=(7, 4))
currpltrow, currpltcol = 0, 0
for i in range(len(Nlist_sort)):
    _, _, patches = axs[currpltrow, currpltcol].hist(simlist_sort[i],
                                     bins=np.arange(0, Nlist_sort[i]+2),
                                     color='cyan')
    patches[int(Ylist_sort[i])].set_fc('black')
    axs[currpltrow, currpltcol].set_xticks([])
    axs[currpltrow, currpltcol].set_yticks([])
    if currpltcol == 8:
        currpltrow += 1
        currpltcol = 0
    else:
        currpltcol += 1
plt.suptitle('Prior predictive distributions vs. observed SFPs,\nfor 36 traces',
             size=14)
plt.tight_layout()
plt.show()




