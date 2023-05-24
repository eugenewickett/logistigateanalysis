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
# Debug why utility evaluations are not changing with different weights matrices

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

csdict_fam['TNnames'] = ['MOD_39', 'MOD_17', 'MODHIGH_95', 'MODHIGH_26']
csdict_fam['SNnames'] = ['MNFR ' + str(i + 1) for i in range(numSN)]

SNpriorMean = np.repeat(sps.logit(0.1), numSN)
TNpriorMean = sps.logit(np.array([0.1, 0.1, 0.15, 0.15]))
TNvar, SNvar = 2., 4.  # Variances for use with prior
csdict_fam['prior'] = prior_normal_assort(np.concatenate((SNpriorMean, TNpriorMean)),
                                np.diag(np.concatenate((np.repeat(SNvar, numSN), np.repeat(TNvar, numTN)))))

csdict_fam['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
numdraws = 20000
csdict_fam['numPostSamples'] = numdraws
np.random.seed(999)  # To replicate draws later
csdict_fam = methods.GeneratePostSamples(csdict_fam)

numcanddraws, numtruthdraws, numdatadraws, numcandneigh = 5000, 5000, 3000, 1000

paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN), candneighnum=numcandneigh)

canddraws, truthdraws, datadraws = util.distribute_draws(csdict_fam['postSamples'], numcanddraws,
                                                                     numtruthdraws, numdatadraws)
paramdict.update({'canddraws': canddraws, 'truthdraws': truthdraws, 'datadraws': datadraws})
paramdict.update({'lossmatrix': lf.build_loss_matrix(truthdraws, canddraws, paramdict)})


import scipy.optimize as spo

# What is loss at different budgets?

# First our standard approach
base = sampf.baseloss(paramdict['lossmatrix'])

# Now with a budget
sampbudget = 100
des = np.array([0.,1.,0.,0.]) # Node 2
allocarr = des * sampbudget
W = sampf.build_weights_matrix(truthdraws,datadraws,allocarr,csdict_fam)

LW = np.matmul(paramdict['lossmatrix'], W)
LWmins = LW.min(axis=0)
loss100 = np.average(LWmins)
util100 = base - loss100
print('Utility at 100 tests, under standard approach: '+str(round(util100,4)))

# Using optimization
def cand_obj_val(x, truthdraws, Wvec, paramdict):
    '''function for optimization step'''
    numnodes = x.shape[0]
    scoremat = lf.score_diff_matrix(truthdraws, x.reshape(1, numnodes), paramdict['scoredict'])[0]
    riskvec = lf.risk_check_array(truthdraws,paramdict['riskdict'])
    #Wvalvec = np.sum(W, axis=1) / W.shape[1]
    return np.sum(np.sum(scoremat*riskvec,axis=1)*Wvec)
# RETURNS SAME VALUES AS IN (LW) MATRIX IF CANDDRAW IS USED FOR x; example:
i, j = 1, 1
x = canddraws[i]
print(cand_obj_val(x,truthdraws,W[:,j],paramdict))
print(LW[i, j])

# define an optimization function for a set of parameters, truthdraws, and weights matrix
def get_bayes_min_cand(truthdraws, Wvec, paramdict, xinit='na'):
    # Initialize with random truthdraw if not provided
    if isinstance(xinit, str):
        xinit = truthdraws[choice(np.arange(truthdraws.shape[0]))]
    # Minimize expected candidate loss
    # NEED BOUNDS?
    #bds = spo.Bounds(np.repeat(0., xinit.shape[0]), np.repeat(1., xinit.shape[0]))
    spoOutput = spo.minimize(cand_obj_val, xinit, args=(truthdraws, Wvec, paramdict), #bounds=bds,
                             tol= 1e-8)  # Reduce tolerance?
    return spoOutput

# First the baseline loss
opt_output = get_bayes_min_cand(truthdraws, np.ones(numtruthdraws)/numtruthdraws, paramdict)
base_opt = opt_output.fun
xinit_base = opt_output.x # use this as our xinit from now on

# Now do iterations over data at the budget level
# Use same W as already generated; stop using running average rule
j = -1 # initialize data index
eps = 5e-3 # stopping rule range
stopamount = 10
rangelist = [0,1]
minvalslist = []
cumavglist = []
optiterlist = []
while np.max(rangelist) - np.min(rangelist) > eps and j < numdatadraws-1: # numdatadraws-1: # our stopping rule
    # increment data index and get new data weights vector
    j += 1
    optout = get_bayes_min_cand(truthdraws, W[:,j], paramdict, xinit=xinit_base)
    minvalslist.append(optout.fun)
    optiterlist.append((optout.nit, optout.nfev, optout.njev))
    cumavglist = np.cumsum(minvalslist)/np.arange(1,j+2)
    if j > stopamount:
        rangelist = cumavglist[-stopamount:]
    if np.mod(j,10) == 3:
        plt.plot(cumavglist)
        plt.show()
cumavglist = np.cumsum(minvalslist)/np.arange(1,j+2)
'''array([1.96978738, 1.98984604, 1.92498042, 1.90914338, 1.87208908,
       1.91120212, 1.9156915 , 1.90938327, 1.92141665, 1.92748459,
       1.94021617, 1.94700074, 1.95314667, 1.95147585, 1.95190607,
       1.94510328, 1.94729559, 1.94150284, 1.94518582, 1.94084529,
       1.93928301, 1.9410227 , 1.94729947, 1.9443391 , 1.94466099,
       1.9447737 , 1.93786483, 1.9389918 , 1.93020185, 1.93042901,
       1.93237228, 1.92294939, 1.92830584, 1.93211457, 1.93074189,
       1.93563182, 1.93730933, 1.94006747, 1.94044184, 1.94104232,
       1.94213395, 1.94015846, 1.93651846, 1.93469153, 1.93278984,
       1.93496678, 1.93807659, 1.9372057 , 1.93814744, 1.93328721,
       1.93433664, 1.93405664, 1.93583416, 1.93913853, 1.94029727,
       1.93941599, 1.93330626, 1.93232921, 1.9316439 , 1.93308907,
       1.93511659, 1.93720467, 1.93676929, 1.93679705, 1.93699911,
       1.93788661, 1.93888563, 1.9364474 , 1.93563462, 1.93729646])'''
plt.plot(cumavglist)
plt.title('Cumulative loss average over data, w/ direct optimization approach\n$N=100$, TN 2, $\epsilon=$'+
          str(eps))
plt.xlabel('Number of data draws')
plt.ylabel('Loss')
plt.ylim([0,2.5])
plt.show()

###############
# What happens if we increase the number of truth draws?
###############
truthdraws = csdict_fam['postSamples'].copy()
datadraws = truthdraws[choice(np.arange(truthdraws.shape[0]), size=100, replace=False)]
W = sampf.build_weights_matrix(truthdraws,datadraws,allocarr,csdict_fam)

# Rerun
j = -1 # initialize data index
eps = 5e-3 # stopping rule range
stopamount = 10
rangelist = [0,1]
minvalslist, cumavglist, optiterlist = [], [], []
while np.max(rangelist) - np.min(rangelist) > eps and j < numdatadraws-1: # numdatadraws-1: # our stopping rule
    # increment data index and get new data weights vector
    j += 1
    print('Iteration: '+str(j))
    optout = get_bayes_min_cand(truthdraws, W[:,j], paramdict, xinit=xinit_base)
    minvalslist.append(optout.fun)
    optiterlist.append((optout.nit, optout.nfev, optout.njev))
    cumavglist = np.cumsum(minvalslist)/np.arange(1,j+2)
    if j > stopamount:
        rangelist = cumavglist[-stopamount:]
    if np.mod(j,10) == 3:
        plt.plot(cumavglist)
        plt.show()
'''23-MAY
minvalslist = [2.1377922035514776, 1.7476243097543676, 1.8703210375683126, 2.091253214468296, 2.0797411301370476, 2.0822853808616117, 1.8722179261164498, 1.9911451990343063, 2.0369052690040976, 2.056041524201491, 1.8568608572514584, 2.1173014151462866, 1.8870038388866166, 1.942677172113613, 2.1147624581996065, 1.9606855854390939, 1.873400182805489, 1.9041392629085196, 1.8734779900086176, 1.9834943184904088, 2.0411340927145454, 1.9271728447474483, 1.9280262090040503, 1.918107916790159, 1.9649220284056104, 1.8592642893661862, 2.1672923062756526, 2.028735567398024, 2.0117332604198923, 2.0179534864672726, 2.0834370674779006, 1.9505691573637989, 1.8594785145098673, 2.016531022318071, 2.0144163253126015, 1.8994931549526222, 1.9711167274691712, 2.1764755553751307, 2.0559854764688397, 1.8909597800077673, 1.9433277687710258, 2.062324132898774, 1.979891914548022, 1.842078196357371, 1.8252696525101824, 1.993891818617405, 2.116069863562463, 2.1096517304040985, 1.7540192631945692, 1.939660769909262, 1.777071734304363, 1.9375661547620788, 1.9556578410212047, 2.0479380924295256, 1.8141876928648575, 1.964550528741992, 1.8207257355937747, 1.8841668107302716, 1.7884514841162495, 1.9345838650847464, 2.00011822255491, 1.8377318992115357, 2.0865509600518157, 1.946626317118803, 2.0044206539400524, 1.9560390486367982, 1.7995928495694227, 1.7426486085334054, 1.8766799518155861, 1.8070439601404542, 1.9193894966846523, 2.02387281875314, 1.702304006786754, 1.6324846156279698, 1.9542219356435306, 1.9037537128808684, 1.9830167657305093, 1.9277180855410347, 2.008637728431884, 2.1040560572604177, 1.936388950873615, 2.026146789387046]
optiterlist = [(96, 4044, 224), (91, 3071, 170), (92, 3575, 198), (106, 3522, 195), (85, 3000, 166), (78, 3625, 201), (103, 3504, 194), (106, 4763, 264), (74, 3252, 180), (69, 3739, 207), (61, 3180, 176), (85, 3450, 191), (75, 3647, 202), (41, 2010, 111), (34, 1866, 103), (72, 2547, 141), (34, 1956, 108), (86, 3126, 173), (75, 2892, 160), (95, 4871, 270), (66, 3195, 177), (59, 2514, 139), (77, 3270, 181), (61, 3342, 185), (89, 3990, 221), (67, 3262, 181), (73, 3684, 204), (46, 2171, 120), (75, 2910, 161), (79, 2603, 144), (88, 3234, 179), (129, 4764, 264), (78, 3495, 194), (71, 2745, 152), (110, 4542, 252), (75, 3210, 178), (83, 5142, 285), (63, 3607, 200), (86, 3558, 197), (77, 3140, 174), (47, 2352, 130), (50, 2310, 128), (99, 3792, 210), (76, 3482, 193), (75, 3518, 195), (85, 3609, 200), (74, 2946, 163), (82, 3735, 207), (73, 4260, 236), (73, 3557, 197), (109, 4115, 228), (85, 3378, 187), (88, 3342, 185), (88, 3918, 217), (69, 2708, 150), (53, 2219, 123), (119, 4727, 262), (105, 3936, 218), (108, 4689, 260), (98, 4205, 233), (82, 3090, 171), (68, 2818, 156), (90, 3915, 217), (94, 3606, 200), (79, 3393, 188), (80, 2837, 157), (61, 3288, 182), (43, 2004, 111), (35, 2262, 125), (91, 3736, 207), (97, 4584, 254), (89, 2999, 166), (91, 5213, 289), (61, 2904, 161), (84, 3972, 220), (88, 3360, 186), (76, 2817, 156), (77, 3126, 173), (89, 3792, 210), (79, 3594, 199), (97, 4494, 249), (87, 3465, 192)]
'''
cumavglist = np.cumsum(minvalslist)/np.arange(1,j+2)
plt.plot(cumavglist)
plt.title('Cumulative loss average over data, w/ dirOpt, $\Gamma_{truth}=20k$ (not 5k)\n$N=100$, TN 2, $\epsilon=$'+
          str(eps))
plt.xlabel('Number of data draws')
plt.ylabel('Loss')
plt.ylim([0,2.5])
plt.show()


###############
# Do we get improvement by including the gradient?
###############
# define a gradient function for any candidate vector x
def cand_obj_val_jac(x, truthdraws, Wvec, paramdict):
    '''function for optimization step'''
    numnodes = x.shape[0]
    scoremat = lf.score_diff_matrix(truthdraws, x.reshape(1, numnodes), paramdict['scoredict'])[0]
    riskvec = lf.risk_check_array(truthdraws,paramdict['riskdict'])
    #Wvalvec = np.sum(W, axis=1) / W.shape[1]
    return np.sum(np.sum(scoremat*riskvec,axis=1)*Wvec)




