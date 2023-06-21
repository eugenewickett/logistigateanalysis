"""
Script that generates the modeling sensitivity table from the case study in the paper: allocations and sampling
budget saved.
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


testint = 10

# Pull allocations and average utilities from files
fam_alloc = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_alloc.npy'))
fam_util_avg = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_util_avg.npy'))
print('Base allocation at 90:\n'+str(fam_alloc[:,9]))
print('Base allocation at 180:\n'+str(fam_alloc[:,18]))
util_avg_arr = np.load(os.path.join('casestudyoutputs', '31MAY', 'util_avg_arr_fam.npy'))
util_avg_unif_90, util_avg_unif_180 = util_avg_arr[1, 9:], util_avg_arr[1, 18:]
util_avg_rudi_90, util_avg_rudi_180 = util_avg_arr[2, 9:], util_avg_arr[2, 18:]
alloc90, alloc180 = fam_util_avg[9], fam_util_avg[18]
kInd = next(x for x, val in enumerate(util_avg_unif_90) if val > alloc90)
unif90saved = round((alloc90 - util_avg_unif_90[kInd - 1]) / (util_avg_unif_90[kInd] - util_avg_unif_90[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_unif_180) if val > alloc180)
unif180saved = round((alloc180 - util_avg_unif_180[kInd-1]) / (util_avg_unif_180[kInd]-util_avg_unif_180[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_rudi_90) if val > alloc90)
rudi90saved = round((alloc90 - util_avg_rudi_90[kInd - 1]) / (util_avg_rudi_90[kInd] - util_avg_rudi_90[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_rudi_180) if val > alloc180)
rudi180saved = round((alloc180 - util_avg_rudi_180[kInd-1]) / (util_avg_rudi_180[kInd]-util_avg_rudi_180[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
print('Saved vs Unif at 90: '+str(unif90saved))
print('Saved vs Rudi at 90: '+str(rudi90saved))
print('Saved vs Unif at 180: '+str(unif180saved))
print('Saved vs Rudi at 180: '+str(rudi180saved))


# Test node prior variance: 1
fam_MS_alloc = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_priorvar_1_alloc.npy'))
fam_MS_util_avg = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_priorvar_1_util_avg.npy'))
util_avg_rudi_90 = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_priorvar_1_util_avg_rudi_90.npy'))
util_avg_rudi_180 = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_priorvar_1_util_avg_rudi_180.npy'))
util_avg_unif_90 = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_priorvar_1_util_avg_unif_90.npy'))
util_avg_unif_180 = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_priorvar_1_util_avg_unif_180.npy'))
alloc90 = fam_MS_util_avg[9]
alloc180 = fam_MS_util_avg[18]
kInd = next(x for x, val in enumerate(util_avg_unif_90) if val > alloc90)
unif90saved = round((alloc90 - util_avg_unif_90[kInd - 1]) / (util_avg_unif_90[kInd] - util_avg_unif_90[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_unif_180) if val > alloc180)
unif180saved = round((alloc180 - util_avg_unif_180[kInd-1]) / (util_avg_unif_180[kInd]-util_avg_unif_180[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_rudi_90) if val > alloc90)
rudi90saved = round((alloc90 - util_avg_rudi_90[kInd - 1]) / (util_avg_rudi_90[kInd] - util_avg_rudi_90[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
kInd = next(x for x, val in enumerate(util_avg_rudi_180) if val > alloc180)
rudi180saved = round((alloc180 - util_avg_rudi_180[kInd-1]) / (util_avg_rudi_180[kInd]-util_avg_rudi_180[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
print('Prior variance of 1, allocation at 90:\n'+str(fam_MS_alloc[:,9]))
print('Saved vs Unif at 90: '+str(unif90saved))
print('Saved vs Rudi at 90: '+str(rudi90saved))
print('Prior variance of 1, allocation at 180:\n'+str(fam_MS_alloc[:,18]))
print('Saved vs Unif at 180: '+str(unif180saved))
print('Saved vs Rudi at 180: '+str(rudi180saved))

# Test node prior variance: 4
fam_MS_alloc = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_priorvar_4_alloc.npy'))
fam_MS_util_avg = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_priorvar_4_util_avg.npy'))
util_avg_rudi_90 = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_priorvar_4_util_avg_rudi_90.npy'))
util_avg_rudi_180 = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_priorvar_4_util_avg_rudi_180.npy'))
util_avg_unif_90 = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_priorvar_4_util_avg_unif_90.npy'))
util_avg_unif_180 = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_priorvar_4_util_avg_unif_180.npy'))
alloc90 = fam_MS_util_avg[9]
alloc180 = fam_MS_util_avg[18]
kInd = next(x for x, val in enumerate(util_avg_unif_90) if val > alloc90)
unif90saved = round((alloc90 - util_avg_unif_90[kInd - 1]) / (util_avg_unif_90[kInd] - util_avg_unif_90[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_unif_180) if val > alloc180)
unif180saved = round((alloc180 - util_avg_unif_180[kInd-1]) / (util_avg_unif_180[kInd]-util_avg_unif_180[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_rudi_90) if val > alloc90)
rudi90saved = round((alloc90 - util_avg_rudi_90[kInd - 1]) / (util_avg_rudi_90[kInd] - util_avg_rudi_90[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
kInd = next(x for x, val in enumerate(util_avg_rudi_180) if val > alloc180)
rudi180saved = round((alloc180 - util_avg_rudi_180[kInd-1]) / (util_avg_rudi_180[kInd]-util_avg_rudi_180[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
print('Prior variance of 4, allocation at 90:\n'+str(fam_MS_alloc[:,9]))
print('Saved vs Unif at 90: '+str(unif90saved))
print('Saved vs Rudi at 90: '+str(rudi90saved))
print('Prior variance of 4, allocation at 180:\n'+str(fam_MS_alloc[:,18]))
print('Saved vs Unif at 180: '+str(unif180saved))
print('Saved vs Rudi at 180: '+str(rudi180saved))

# Risk slope: 0.3
fam_MS_alloc = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_riskslope_03_alloc.npy'))
fam_MS_util_avg = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_riskslope_03_util_avg.npy'))
util_avg_rudi_90 = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_riskslope_03_util_avg_rudi_90.npy'))
util_avg_rudi_180 = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_riskslope_03_util_avg_rudi_180.npy'))
util_avg_unif_90 = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_riskslope_03_util_avg_unif_90.npy'))
util_avg_unif_180 = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_riskslope_03_util_avg_unif_180.npy'))
alloc90 = fam_MS_util_avg[9]
alloc180 = fam_MS_util_avg[18]
kInd = next(x for x, val in enumerate(util_avg_unif_90) if val > alloc90)
unif90saved = round((alloc90 - util_avg_unif_90[kInd - 1]) / (util_avg_unif_90[kInd] - util_avg_unif_90[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_unif_180) if val > alloc180)
unif180saved = round((alloc180 - util_avg_unif_180[kInd-1]) / (util_avg_unif_180[kInd]-util_avg_unif_180[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_rudi_90) if val > alloc90)
rudi90saved = round((alloc90 - util_avg_rudi_90[kInd - 1]) / (util_avg_rudi_90[kInd] - util_avg_rudi_90[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
kInd = next(x for x, val in enumerate(util_avg_rudi_180) if val > alloc180)
rudi180saved = round((alloc180 - util_avg_rudi_180[kInd-1]) / (util_avg_rudi_180[kInd]-util_avg_rudi_180[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
print('Risk slope of 0.3, allocation at 90:\n'+str(fam_MS_alloc[:,9]))
print('Saved vs Unif at 90: '+str(unif90saved))
print('Saved vs Rudi at 90: '+str(rudi90saved))
print('Risk slope of 0.3, allocation at 180:\n'+str(fam_MS_alloc[:,18]))
print('Saved vs Unif at 180: '+str(unif180saved))
print('Saved vs Rudi at 180: '+str(rudi180saved))

# Risk slope: 0.9
fam_MS_alloc = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_riskslope_09_alloc.npy'))
fam_MS_util_avg = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_riskslope_09_util_avg.npy'))
util_avg_rudi_90 = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_riskslope_09_util_avg_rudi_90.npy'))
util_avg_rudi_180 = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_riskslope_09_util_avg_rudi_180.npy'))
util_avg_unif_90 = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_riskslope_09_util_avg_unif_90.npy'))
util_avg_unif_180 = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_riskslope_09_util_avg_unif_180.npy'))
alloc90 = fam_MS_util_avg[9]
alloc180 = fam_MS_util_avg[18]
kInd = next(x for x, val in enumerate(util_avg_unif_90) if val > alloc90)
unif90saved = round((alloc90 - util_avg_unif_90[kInd - 1]) / (util_avg_unif_90[kInd] - util_avg_unif_90[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_unif_180) if val > alloc180)
unif180saved = round((alloc180 - util_avg_unif_180[kInd-1]) / (util_avg_unif_180[kInd]-util_avg_unif_180[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_rudi_90) if val > alloc90)
rudi90saved = round((alloc90 - util_avg_rudi_90[kInd - 1]) / (util_avg_rudi_90[kInd] - util_avg_rudi_90[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
kInd = next(x for x, val in enumerate(util_avg_rudi_180) if val > alloc180)
rudi180saved = round((alloc180 - util_avg_rudi_180[kInd-1]) / (util_avg_rudi_180[kInd]-util_avg_rudi_180[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
print('Risk slope of 0.9, allocation at 90:\n'+str(fam_MS_alloc[:,9]))
print('Saved vs Unif at 90: '+str(unif90saved))
print('Saved vs Rudi at 90: '+str(rudi90saved))
print('Risk slope of 0.9, allocation at 180:\n'+str(fam_MS_alloc[:,18]))
print('Saved vs Unif at 180: '+str(unif180saved))
print('Saved vs Rudi at 180: '+str(rudi180saved))

# Underestimation weight: 1
fam_MS_alloc = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_underestweight_1_alloc.npy'))
fam_MS_util_avg = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_underestweight_1_util_avg.npy'))
util_avg_rudi_90 = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_underestweight_1_util_avg_rudi_90.npy'))
util_avg_rudi_180 = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_underestweight_1_util_avg_rudi_180.npy'))
util_avg_unif_90 = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_underestweight_1_util_avg_unif_90.npy'))
util_avg_unif_180 = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_underestweight_1_util_avg_unif_180.npy'))
alloc90 = fam_MS_util_avg[9]
alloc180 = fam_MS_util_avg[18]
kInd = next(x for x, val in enumerate(util_avg_unif_90) if val > alloc90)
unif90saved = round((alloc90 - util_avg_unif_90[kInd - 1]) / (util_avg_unif_90[kInd] - util_avg_unif_90[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_unif_180) if val > alloc180)
unif180saved = round((alloc180 - util_avg_unif_180[kInd-1]) / (util_avg_unif_180[kInd]-util_avg_unif_180[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_rudi_90) if val > alloc90)
rudi90saved = round((alloc90 - util_avg_rudi_90[kInd - 1]) / (util_avg_rudi_90[kInd] - util_avg_rudi_90[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
kInd = next(x for x, val in enumerate(util_avg_rudi_180) if val > alloc180)
rudi180saved = round((alloc180 - util_avg_rudi_180[kInd-1]) / (util_avg_rudi_180[kInd]-util_avg_rudi_180[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
print('Underestimation weight of 1, allocation at 90:\n'+str(fam_MS_alloc[:,9]))
print('Saved vs Unif at 90: '+str(unif90saved))
print('Saved vs Rudi at 90: '+str(rudi90saved))
print('Underestimation weight of 1, allocation at 180:\n'+str(fam_MS_alloc[:,18]))
print('Saved vs Unif at 180: '+str(unif180saved))
print('Saved vs Rudi at 180: '+str(rudi180saved))

# Underestimation weight: 10
fam_MS_alloc = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_underestweight_10_alloc.npy'))
fam_MS_util_avg = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_underestweight_10_util_avg.npy'))
util_avg_rudi_90 = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_underestweight_10_util_avg_rudi_90.npy'))
util_avg_rudi_180 = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_underestweight_10_util_avg_rudi_180.npy'))
util_avg_unif_90 = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_underestweight_10_util_avg_unif_90.npy'))
util_avg_unif_180 = np.load(os.path.join('casestudyoutputs', '13JUN', 'fam_MS_underestweight_10_util_avg_unif_180.npy'))
alloc90 = fam_MS_util_avg[9]
alloc180 = fam_MS_util_avg[18]
kInd = next(x for x, val in enumerate(util_avg_unif_90) if val > alloc90)
unif90saved = round((alloc90 - util_avg_unif_90[kInd - 1]) / (util_avg_unif_90[kInd] - util_avg_unif_90[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_unif_180) if val > alloc180)
unif180saved = round((alloc180 - util_avg_unif_180[kInd-1]) / (util_avg_unif_180[kInd]-util_avg_unif_180[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_rudi_90) if val > alloc90)
rudi90saved = round((alloc90 - util_avg_rudi_90[kInd - 1]) / (util_avg_rudi_90[kInd] - util_avg_rudi_90[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
kInd = next(x for x, val in enumerate(util_avg_rudi_180) if val > alloc180)
rudi180saved = round((alloc180 - util_avg_rudi_180[kInd-1]) / (util_avg_rudi_180[kInd]-util_avg_rudi_180[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
print('Underestimation weight of 10, allocation at 90:\n'+str(fam_MS_alloc[:,9]))
print('Saved vs Unif at 90: '+str(unif90saved))
print('Saved vs Rudi at 90: '+str(rudi90saved))
print('Underestimation weight of 10, allocation at 180:\n'+str(fam_MS_alloc[:,18]))
print('Saved vs Unif at 180: '+str(unif180saved))
print('Saved vs Rudi at 180: '+str(rudi180saved))


######### EXPLORATORY SETTING ###############
fam_alloc = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_alloc.npy'))
fam_util_avg = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_util_avg.npy'))
print('Base allocation at 90:\n'+str(fam_alloc[:,9]))
print('Base allocation at 180:\n'+str(fam_alloc[:,18]))
util_avg_arr = np.load(os.path.join('casestudyoutputs', '31MAY', 'util_avg_arr_expl.npy'))
util_avg_unif_90, util_avg_unif_180 = util_avg_arr[1, 9:], util_avg_arr[1, 18:]
util_avg_rudi_90, util_avg_rudi_180 = util_avg_arr[2, 9:], util_avg_arr[2, 18:]
alloc90, alloc180 = fam_util_avg[9], fam_util_avg[18]
# Extend rudimentary utility a bit to get estimate
slope = (util_avg_rudi_180[-1] - util_avg_rudi_180[0]) / util_avg_rudi_180.shape[0]
addutil = slope*np.arange(30) + util_avg_rudi_180[-1]
util_avg_rudi_180 = np.concatenate((util_avg_rudi_180, addutil))

kInd = next(x for x, val in enumerate(util_avg_unif_90) if val > alloc90)
unif90saved = round((alloc90 - util_avg_unif_90[kInd - 1]) / (util_avg_unif_90[kInd] - util_avg_unif_90[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_unif_180) if val > alloc180)
unif180saved = round((alloc180 - util_avg_unif_180[kInd-1]) / (util_avg_unif_180[kInd]-util_avg_unif_180[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_rudi_90) if val > alloc90)
rudi90saved = round((alloc90 - util_avg_rudi_90[kInd - 1]) / (util_avg_rudi_90[kInd] - util_avg_rudi_90[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_rudi_180) if val > alloc180)
rudi180saved = round((alloc180 - util_avg_rudi_180[kInd-1]) / (util_avg_rudi_180[kInd]-util_avg_rudi_180[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
print('Saved vs Unif at 90: '+str(unif90saved))
print('Saved vs Rudi at 90: '+str(rudi90saved))
print('Saved vs Unif at 180: '+str(unif180saved))
print('Saved vs Rudi at 180: '+str(rudi180saved))

# Prior variance: 4
fam_MS_alloc = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_priorvar_1_alloc.npy'))
fam_MS_util_avg = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_priorvar_1_util_avg.npy'))
util_avg_rudi_90 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_priorvar_1_util_avg_rudi_90.npy'))
util_avg_rudi_180 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_priorvar_1_util_avg_rudi_180.npy'))
util_avg_unif_90 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_priorvar_1_util_avg_unif_90.npy'))
util_avg_unif_180 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_priorvar_1_util_avg_unif_180.npy'))
alloc90 = fam_MS_util_avg[9]
alloc180 = fam_MS_util_avg[18]
kInd = next(x for x, val in enumerate(util_avg_unif_90) if val > alloc90)
unif90saved = round((alloc90 - util_avg_unif_90[kInd - 1]) / (util_avg_unif_90[kInd] - util_avg_unif_90[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_unif_180) if val > alloc180)
unif180saved = round((alloc180 - util_avg_unif_180[kInd-1]) / (util_avg_unif_180[kInd]-util_avg_unif_180[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_rudi_90) if val > alloc90)
rudi90saved = round((alloc90 - util_avg_rudi_90[kInd - 1]) / (util_avg_rudi_90[kInd] - util_avg_rudi_90[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
kInd = next(x for x, val in enumerate(util_avg_rudi_180) if val > alloc180)
rudi180saved = round((alloc180 - util_avg_rudi_180[kInd-1]) / (util_avg_rudi_180[kInd]-util_avg_rudi_180[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
print('Prior variance of 1, allocation at 90:\n'+str(fam_MS_alloc[:,9]))
print('Saved vs Unif at 90: '+str(unif90saved))
print('Saved vs Rudi at 90: '+str(rudi90saved))
print('Prior variance of 1, allocation at 180:\n'+str(fam_MS_alloc[:,18]))
print('Saved vs Unif at 180: '+str(unif180saved))
print('Saved vs Rudi at 180: '+str(rudi180saved))

# Risk slope: 0.3
fam_MS_alloc = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_riskslope_03_alloc.npy'))
fam_MS_util_avg = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_riskslope_03_util_avg.npy'))
util_avg_rudi_90 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_riskslope_03_util_avg_rudi_90.npy'))
util_avg_rudi_180 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_riskslope_03_util_avg_rudi_180.npy'))
util_avg_unif_90 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_riskslope_03_util_avg_unif_90.npy'))
util_avg_unif_180 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_riskslope_03_util_avg_unif_180.npy'))
alloc90 = fam_MS_util_avg[9]
alloc180 = fam_MS_util_avg[18]
kInd = next(x for x, val in enumerate(util_avg_unif_90) if val > alloc90)
unif90saved = round((alloc90 - util_avg_unif_90[kInd - 1]) / (util_avg_unif_90[kInd] - util_avg_unif_90[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_unif_180) if val > alloc180)
unif180saved = round((alloc180 - util_avg_unif_180[kInd-1]) / (util_avg_unif_180[kInd]-util_avg_unif_180[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_rudi_90) if val > alloc90)
rudi90saved = round((alloc90 - util_avg_rudi_90[kInd - 1]) / (util_avg_rudi_90[kInd] - util_avg_rudi_90[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
kInd = next(x for x, val in enumerate(util_avg_rudi_180) if val > alloc180)
rudi180saved = round((alloc180 - util_avg_rudi_180[kInd-1]) / (util_avg_rudi_180[kInd]-util_avg_rudi_180[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
print('Risk slope of 0.3, allocation at 90:\n'+str(fam_MS_alloc[:,9]))
print('Saved vs Unif at 90: '+str(unif90saved))
print('Saved vs Rudi at 90: '+str(rudi90saved))
print('Risk slope of 0.3, allocation at 180:\n'+str(fam_MS_alloc[:,18]))
print('Saved vs Unif at 180: '+str(unif180saved))
print('Saved vs Rudi at 180: '+str(rudi180saved))

# Risk slope: 0.9
fam_MS_alloc = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_riskslope_09_alloc.npy'))
fam_MS_util_avg = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_riskslope_09_util_avg.npy'))
util_avg_rudi_90 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_riskslope_09_util_avg_rudi_90.npy'))
util_avg_rudi_180 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_riskslope_09_util_avg_rudi_180.npy'))
util_avg_unif_90 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_riskslope_09_util_avg_unif_90.npy'))
util_avg_unif_180 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_riskslope_09_util_avg_unif_180.npy'))
alloc90 = fam_MS_util_avg[9]
alloc180 = fam_MS_util_avg[18]
kInd = next(x for x, val in enumerate(util_avg_unif_90) if val > alloc90)
unif90saved = round((alloc90 - util_avg_unif_90[kInd - 1]) / (util_avg_unif_90[kInd] - util_avg_unif_90[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_unif_180) if val > alloc180)
unif180saved = round((alloc180 - util_avg_unif_180[kInd-1]) / (util_avg_unif_180[kInd]-util_avg_unif_180[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_rudi_90) if val > alloc90)
rudi90saved = round((alloc90 - util_avg_rudi_90[kInd - 1]) / (util_avg_rudi_90[kInd] - util_avg_rudi_90[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
kInd = next(x for x, val in enumerate(util_avg_rudi_180) if val > alloc180)
rudi180saved = round((alloc180 - util_avg_rudi_180[kInd-1]) / (util_avg_rudi_180[kInd]-util_avg_rudi_180[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
print('Risk slope of 0.9, allocation at 90:\n'+str(fam_MS_alloc[:,9]))
print('Saved vs Unif at 90: '+str(unif90saved))
print('Saved vs Rudi at 90: '+str(rudi90saved))
print('Risk slope of 0.9, allocation at 180:\n'+str(fam_MS_alloc[:,18]))
print('Saved vs Unif at 180: '+str(unif180saved))
print('Saved vs Rudi at 180: '+str(rudi180saved))

# Sourcing change: 1
fam_MS_alloc = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_sourcing_1_alloc.npy'))
fam_MS_util_avg = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_sourcing_1_util_avg.npy'))
util_avg_rudi_90 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_sourcing_1_util_avg_rudi_90.npy'))
util_avg_rudi_180 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_sourcing_1_util_avg_rudi_180.npy'))
util_avg_unif_90 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_sourcing_1_util_avg_unif_90.npy'))
util_avg_unif_180 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_sourcing_1_util_avg_unif_180.npy'))
alloc90 = fam_MS_util_avg[9]
alloc180 = fam_MS_util_avg[18]
kInd = next(x for x, val in enumerate(util_avg_unif_90) if val > alloc90)
unif90saved = round((alloc90 - util_avg_unif_90[kInd - 1]) / (util_avg_unif_90[kInd] - util_avg_unif_90[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_unif_180) if val > alloc180)
unif180saved = round((alloc180 - util_avg_unif_180[kInd-1]) / (util_avg_unif_180[kInd]-util_avg_unif_180[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_rudi_90) if val > alloc90)
rudi90saved = round((alloc90 - util_avg_rudi_90[kInd - 1]) / (util_avg_rudi_90[kInd] - util_avg_rudi_90[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
kInd = next(x for x, val in enumerate(util_avg_rudi_180) if val > alloc180)
rudi180saved = round((alloc180 - util_avg_rudi_180[kInd-1]) / (util_avg_rudi_180[kInd]-util_avg_rudi_180[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
print('Souring matrix 1, allocation at 90:\n'+str(fam_MS_alloc[:,9]))
print('Saved vs Unif at 90: '+str(unif90saved))
print('Saved vs Rudi at 90: '+str(rudi90saved))
print('Sourcing matrix 1, allocation at 180:\n'+str(fam_MS_alloc[:,18]))
print('Saved vs Unif at 180: '+str(unif180saved))
print('Saved vs Rudi at 180: '+str(rudi180saved))

# Sourcing change: 2
fam_MS_alloc = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_sourcing_2_alloc.npy'))
fam_MS_util_avg = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_sourcing_2_util_avg.npy'))
util_avg_rudi_90 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_sourcing_2_util_avg_rudi_90.npy'))
util_avg_rudi_180 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_sourcing_2_util_avg_rudi_180.npy'))
util_avg_unif_90 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_sourcing_2_util_avg_unif_90.npy'))
util_avg_unif_180 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_sourcing_2_util_avg_unif_180.npy'))
alloc90 = fam_MS_util_avg[9]
alloc180 = fam_MS_util_avg[18]
kInd = next(x for x, val in enumerate(util_avg_unif_90) if val > alloc90)
unif90saved = round((alloc90 - util_avg_unif_90[kInd - 1]) / (util_avg_unif_90[kInd] - util_avg_unif_90[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_unif_180) if val > alloc180)
unif180saved = round((alloc180 - util_avg_unif_180[kInd-1]) / (util_avg_unif_180[kInd]-util_avg_unif_180[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_rudi_90) if val > alloc90)
rudi90saved = round((alloc90 - util_avg_rudi_90[kInd - 1]) / (util_avg_rudi_90[kInd] - util_avg_rudi_90[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
kInd = next(x for x, val in enumerate(util_avg_rudi_180) if val > alloc180)
rudi180saved = round((alloc180 - util_avg_rudi_180[kInd-1]) / (util_avg_rudi_180[kInd]-util_avg_rudi_180[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
print('Souring matrix 2, allocation at 90:\n'+str(fam_MS_alloc[:,9]))
print('Saved vs Unif at 90: '+str(unif90saved))
print('Saved vs Rudi at 90: '+str(rudi90saved))
print('Sourcing matrix 2, allocation at 180:\n'+str(fam_MS_alloc[:,18]))
print('Saved vs Unif at 180: '+str(unif180saved))
print('Saved vs Rudi at 180: '+str(rudi180saved))

# Underestimation weight: 1
fam_MS_alloc = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_underestweight_1_alloc.npy'))
fam_MS_util_avg = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_underestweight_1_util_avg.npy'))
util_avg_rudi_90 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_underestweight_1_util_avg_rudi_90.npy'))
util_avg_rudi_180 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_underestweight_1_util_avg_rudi_180.npy'))
util_avg_unif_90 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_underestweight_1_util_avg_unif_90.npy'))
util_avg_unif_180 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_underestweight_1_util_avg_unif_180.npy'))
alloc90 = fam_MS_util_avg[9]
alloc180 = fam_MS_util_avg[18]
kInd = next(x for x, val in enumerate(util_avg_unif_90) if val > alloc90)
unif90saved = round((alloc90 - util_avg_unif_90[kInd - 1]) / (util_avg_unif_90[kInd] - util_avg_unif_90[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_unif_180) if val > alloc180)
unif180saved = round((alloc180 - util_avg_unif_180[kInd-1]) / (util_avg_unif_180[kInd]-util_avg_unif_180[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_rudi_90) if val > alloc90)
rudi90saved = round((alloc90 - util_avg_rudi_90[kInd - 1]) / (util_avg_rudi_90[kInd] - util_avg_rudi_90[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
kInd = next(x for x, val in enumerate(util_avg_rudi_180) if val > alloc180)
rudi180saved = round((alloc180 - util_avg_rudi_180[kInd-1]) / (util_avg_rudi_180[kInd]-util_avg_rudi_180[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
print('Underestimation weight of 1, allocation at 90:\n'+str(fam_MS_alloc[:,9]))
print('Saved vs Unif at 90: '+str(unif90saved))
print('Saved vs Rudi at 90: '+str(rudi90saved))
print('Underestimation weight of 1, allocation at 180:\n'+str(fam_MS_alloc[:,18]))
print('Saved vs Unif at 180: '+str(unif180saved))
print('Saved vs Rudi at 180: '+str(rudi180saved))

# Underestimation weight: 10
fam_MS_alloc = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_underestweight_10_alloc.npy'))
fam_MS_util_avg = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_underestweight_10_util_avg.npy'))
util_avg_rudi_90 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_underestweight_10_util_avg_rudi_90.npy'))
util_avg_rudi_180 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_underestweight_10_util_avg_rudi_180.npy'))
util_avg_unif_90 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_underestweight_10_util_avg_unif_90.npy'))
util_avg_unif_180 = np.load(os.path.join('casestudyoutputs', '15JUN', 'expl_MS_underestweight_10_util_avg_unif_180.npy'))
alloc90 = fam_MS_util_avg[9]
alloc180 = fam_MS_util_avg[18]
kInd = next(x for x, val in enumerate(util_avg_unif_90) if val > alloc90)
unif90saved = round((alloc90 - util_avg_unif_90[kInd - 1]) / (util_avg_unif_90[kInd] - util_avg_unif_90[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_unif_180) if val > alloc180)
unif180saved = round((alloc180 - util_avg_unif_180[kInd-1]) / (util_avg_unif_180[kInd]-util_avg_unif_180[kInd - 1]) *\
                    testint) + (kInd - 1) * testint
kInd = next(x for x, val in enumerate(util_avg_rudi_90) if val > alloc90)
rudi90saved = round((alloc90 - util_avg_rudi_90[kInd - 1]) / (util_avg_rudi_90[kInd] - util_avg_rudi_90[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
kInd = next(x for x, val in enumerate(util_avg_rudi_180) if val > alloc180)
rudi180saved = round((alloc180 - util_avg_rudi_180[kInd-1]) / (util_avg_rudi_180[kInd]-util_avg_rudi_180[kInd - 1]) *\
                    testint*5) + (kInd - 1) * testint*5
print('Underestimation weight of 10, allocation at 90:\n'+str(fam_MS_alloc[:,9]))
print('Saved vs Unif at 90: '+str(unif90saved))
print('Saved vs Rudi at 90: '+str(rudi90saved))
print('Underestimation weight of 10, allocation at 180:\n'+str(fam_MS_alloc[:,18]))
print('Saved vs Unif at 180: '+str(unif180saved))
print('Saved vs Rudi at 180: '+str(rudi180saved))



