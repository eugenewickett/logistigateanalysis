# This script seeks an example supply chain where correlated sourcing among test nodes
# leads to bad IP-RP solutions.
# We will then identify algorithms for addressing this type of "bad" solution
from logistigate.logistigate import utilities as util  # Pull from the submodule "develop" branch
from logistigate.logistigate import methods
from logistigate.logistigate.priors import prior_normal_assort
from logistigate.logistigate import lossfunctions as lf
from logistigate.logistigate import samplingplanfunctions as sampf
from logistigate.logistigate import orienteering as opf

from orienteering.senegalsetup import *

import os
import pickle
import time

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import matplotlib.ticker as mtick

import pandas as pd
import numpy as np
from numpy.random import choice
import random
import itertools
import scipy.stats as sps
import scipy.special as spsp

import scipy.optimize as spo
from scipy.optimize import LinearConstraint
from scipy.optimize import milp

import string

# Initialize some standard plot parameters
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.family"] = "serif"

# TODO: CONSIDER ADJUSTING HERE
# Regions and Manufacturers; number of prior tests and the minimum sampled from visisted districts
numreg, nummanuf, numpriortests, priortestmin = 5, 5, 40, 3
distnumlist = [3, 3, 3, 3, 3]  # Number of districts per region
regfixedcost, numdist = 0.25, np.sum(distnumlist)
numpriordiststested = round(numdist * (20 / 45)) # 20/45 districts were tested in Senegal dataset
# District travel times; use the same for each region, with a capital, a near district, and a far district
#   first district is always 0.05 days
dist_fixedcostdays = [0.05, 0.15, 0.25]

# Generate distance matrix for regions; this is approximated from the sample figure in the paper
regcost_mat = np.array([[0, 1.3, 1.0, 3.2, 3.4],
                        [0., 0., 0.7, 3.4, 2.3],
                        [0., 0., 0., 2.5, 2.0],
                        [0., 0., 0., 0., 1.8],
                        [0., 0., 0., 0., 0.]])
regcost_mat = regcost_mat/3 # scale to approximate the avg Sengal dataset travel time
np.sum(np.sum(regcost_mat))/np.count_nonzero(regcost_mat)
for i in range(numreg):
    for j in range(numreg):
        if j < i:
            regcost_mat[i, j] = regcost_mat[j, i]
for i in range(numreg):
        for j in range(numreg):
            if j > 0 and j != i:
                regcost_mat[i, j] = regcost_mat[i, j] + regfixedcost
# Check that triangle inequality is satisfied
for i in range(numreg):
    for j in range(numreg):
        for k in range(numreg):
            if regcost_mat[i, j] > regcost_mat[i, k] + regcost_mat[k, j]:
                print('Triangle inequality issue: ' + str(i) + ' to ' + str(j) + ' via ' + str(k))

# Get region and manufacturer names
regnames = ['Reg'+str(i+1) for i in range(numreg)]
manufnames = ['Manuf'+string.ascii_uppercase[i] for i in range(nummanuf)]

# GENERATE SFP RATES, SOURCING PATTERNS, AND PRIOR DATA SET
rseed = 23026    # TODO: CHANGE HERE
# First generate SFP rates for all districts and manufacturers; use wide priors used in case study dataset
lgdict = {'TNnum': numdist, 'SNnum': nummanuf}
SetupSenegalPriors(lgdict, randseed=rseed)
lgdict['truerates'] = lgdict['prior'].expitrand()[0]  # SNs then TNs
# Generate sourcing patterns with high variance; use a chi-sq distribution w df=5 to choose sourcing, then normalize
sourcingmat = np.zeros((numdist, nummanuf))
for d in range(numdist):
    tempvec = np.random.chisquare(5, size=nummanuf)
    sourcingmat[d] = tempvec/np.sum(tempvec)
# plt.hist(sourcingmat.reshape(nummanuf*numdist))
# plt.show()
# Randomly choose districts to test, weighted by distance to capital (Reg1)
# Allocate numpriortests across districts, weighted by distance from capital
distnames = []
disttraveltimes = []
for currregind in range(numreg):
    currregtraveltime = regcost_mat[0][currregind]
    for currdistind in range(distnumlist[currregind]):
        distnames.append('Dist'+str(currregind+1)+string.ascii_lowercase[currdistind])
        disttraveltimes.append(currregtraveltime+dist_fixedcostdays[currdistind])
# Make adjusted probability weights of district being sampled in prior data
wtconstant = 0.1  # TODO: MAYBE INCREASE HERE IF DON'T WANT DISTANCE-WEIGHTED SAMPLING PROBABILITIES
tempvec = [1/(disttraveltimes[x]+wtconstant) for x in range(len(disttraveltimes))]
priordistweights = tempvec/np.sum(tempvec)
# Choose which districts are sampled
priordists = np.random.choice(np.arange(numdist), size=numpriordiststested, replace=False, p=priordistweights)
priordists.sort()
# Allocate tests to sampled dists; use subset of priordistweights, but ensure minimum is met at each tested dist
samplevec = np.repeat(priortestmin, numpriordiststested)
addsamplevec = np.random.choice(np.arange(len(priordists)), size=numpriortests-np.sum(samplevec),
                                replace=True, p=priordistweights[priordists]/np.sum(priordistweights[priordists]))
for x in addsamplevec:
    samplevec[x] += 1
# Determine number of positives by sampling from sourcing vectors and true SFP rates
N, Y = np.zeros(sourcingmat.shape), np.zeros(sourcingmat.shape)
for currNiInd, currNi in enumerate(samplevec):  # grab number of tests for sampled districts
    currd = priordists[currNiInd]
    currYi = 0
    for n in range(currNi):  # identify manuf
        currmanuf = np.random.choice(np.arange(nummanuf), p=sourcingmat[currd], size=1)[0]
        SNrate, TNrate = lgdict['truerates'][currmanuf], lgdict['truerates'][nummanuf + currd]
        trueposrate = SNrate + TNrate - SNrate*TNrate
        testres = np.random.binomial(1, p=trueposrate)
        N[currd, currmanuf] += 1
        Y[currd, currmanuf] += testres















