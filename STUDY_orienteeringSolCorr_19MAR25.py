
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from orienteering.senegalsetup import *
from logistigate.logistigate import orienteering as opf

plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.family"] = "serif"

# Pull data from newly constructed CSV files
dept_df, regcost_mat, regNames, deptNames, manufNames, numReg, testdatadict = GetSenegalCSVData()
(numTN, numSN) = testdatadict['N'].shape  # For later use

PrintDataSummary(testdatadict)  # Cursory data check

# Set up logistigate dictionary
lgdict = util.initDataDict(testdatadict['N'], testdatadict['Y'])
lgdict.update({'TNnames': deptNames, 'SNnames': manufNames})

# Set up priors for SFP rates at nodes
SetupSenegalPriors(lgdict)

# Set up MCMC
lgdict['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 1000, 'delta': 0.4}
mcmcfilestr = os.path.join('orienteering', 'numpy_objects', 'draws')

RetrieveMCMCBatches(lgdict, 5, mcmcfilestr, maxbatchnum=99, rand=True, randseed=1122)

# Add boostrap-sampled sourcing vectors for non-tested test nodes; 20 is the avg number of tests per visited dept
AddBootstrapQ(lgdict, numboot=int(np.sum(lgdict['N'])/np.count_nonzero(np.sum(lgdict['Q'], axis=1))), randseed=44)
Q = lgdict['Q']
print(Q[0])

# Retreive previously calculated candidate solutions
candpklstr_1400 = os.path.join('orienteering', 'pkl_paths', 'candpaths_df_1400.pkl')
candpaths_df_1400 = pd.read_pickle(candpklstr_1400)

# Extract candidates for which we have utility estimates
df_main = candpaths_df_1400[candpaths_df_1400['Uoracle'] > 0.0]
# Build UBgap as absolute value and percentage of actual utility
df_main = df_main.assign(UBgap=lambda df_main:df_main.IPRPobj-df_main.Uoracle)
df_main = df_main.assign(UBgapperc=lambda df_main:df_main.UBgap/df_main.Uoracle)

# GOAL: Build TN/SN interpolations that better approximate the utility
# Step 1: Build uncertainty weights with respect to the decision threshold
t = 0.15  # From case study paramdict
omega_vec = np.zeros(numSN+numTN)
numdraws = lgdict['postSamples'].shape[0]
for SNind in range(numSN):  # SNs first
    val1 = len(np.where(lgdict['postSamples'][:, SNind]<t)[0])
    val2 = numdraws - val1
    omega_vec[SNind] = 2-(2*max(val1,val2)/numdraws)
for TNind in range(numSN, numSN+numTN):
    val1 = len(np.where(lgdict['postSamples'][:, TNind] < t)[0])
    val2 = numdraws - val1
    omega_vec[TNind] = 2 - (2 * max(val1, val2) / numdraws)

# Step 2: Use omegas to decompose the utility interpolations into test-node and supply-node elements
# bigomega is the sum of uncertainty measures at each test node
bigomega_vec = np.zeros(numTN)
for TNind in range(numTN):
    bigomega_vec[TNind] = omega_vec[numSN+TNind] + np.sum(Q[TNind]*omega_vec[:numSN])

# Get TN-elements of interpolations
interp_df = pd.read_csv(os.path.join('orienteering', 'csv_utility', 'interp_df_BASE.csv'))
interp_TN_df = interp_df.copy()
for TNind in range(numTN):
    const = omega_vec[numSN+TNind]/bigomega_vec[TNind]
    interp_TN_df.at[TNind, 'Util_lo'] = interp_df.at[TNind, 'Util_lo'] * const
    interp_TN_df.at[TNind, 'Util_hi'] = interp_df.at[TNind, 'Util_hi'] * const

# Get SN-elements of interpolations
interp_SN_df = pd.DataFrame({'SuppName': manufNames, 'Bounds': np.zeros(numSN),
                             'Util_lo': np.zeros(numSN), 'Util_hi': np.zeros(numSN)})
interp_df_temp = interp_df.copy()  # Use so as to adjust TNs with differing upper bounds
modinds = interp_df_temp['Bounds'] < 81
for modind in range(len(interp_df_temp)):
    if modinds[modind] == True:
        interp_df_temp.at[modind, 'Bounds'] = 81
        interp_df_temp.at[modind, 'Util_hi'] = interp_df.at[modind, 'Util_hi'] * 1.03
for SNind in range(numSN):
    lovals, hivals = [], []
    for TNind in range(numTN):
        if Q[TNind, SNind] > 0:
            const = omega_vec[SNind]*Q[TNind, SNind]/bigomega_vec[TNind]
            lovals.append(const * interp_df_temp.at[TNind, 'Util_lo'])
            hivals.append(const * interp_df_temp.at[TNind, 'Util_hi'])
            interp_SN_df.at[SNind, 'Bounds'] = interp_df_temp.at[TNind, 'Bounds']
    # TODO: Try averaging according to sourcing probabilities
    interp_SN_df.at[SNind, 'Util_lo'] = np.average(lovals)
    interp_SN_df.at[SNind, 'Util_hi'] = np.average(hivals)


def GetInterpVectors(interp_df):
    """Build needed interpolation vectors for use with relaxed program"""
    lvec, juncvec, m1vec, m2vec, bds, lovals, hivals = [], [], [], [], [], [], []
    for ind in range(interp_df.shape[0]):
        row = interp_df.iloc[ind]
        currBound, loval, hival = row['Bounds'], row['Util_lo'], row['Util_hi']
        # Get interpolation values
        _, _, l, k, m1, m2 = opf.GetTriangleInterpolation([0, 1, currBound], [0, loval, hival])
        lvec.append(l)
        juncvec.append(k)
        m1vec.append(m1)
        m2vec.append(m2)
        bds.append(currBound)
        lovals.append(loval)
        hivals.append(hival)

    return lvec, juncvec, m1vec, m2vec, bds, lovals, hivals

# Get vectors of zero intercepts, junctures, and interpolation slopes for each of our Utilde evals at each district
lvec, juncvec, m1vec, m2vec, bds, lovals, hivals = GetInterpVectors(interp_df)
lvec_TN, juncvec_TN, m1vec_TN, m2vec_TN, bds_TN, lovals_TN, hivals_TN = GetInterpVectors(interp_TN_df)
lvec_SN, juncvec_SN, m1vec_SN, m2vec_SN, bds_SN, lovals_SN, hivals_SN = GetInterpVectors(interp_SN_df)

# Construct new objective using TN/SN-element interpolations
# Check we get the IPRP objective when re-running everything
def getObjVec(l, m1, m2):
    return np.concatenate((np.array(l), np.array(m1), np.array(m2)))

ind = 3

objVec = getObjVec(lvec, m1vec, m2vec)
distaccessVar = np.array(eval(df_main.iloc[ind]['DistAccessBinaryVec']))
allocVar = df_main.iloc[ind]['Allocation']
allocVar1, allocVar2 = np.zeros(numTN), np.zeros(numTN)
for i in range(numTN):
    if allocVar[i] > juncvec[i]:
        allocVar1[i] = juncvec[i] - 1
        allocVar2[i] = allocVar[i] - juncvec[i] + 1
    elif allocVar[i] > 0:
        allocVar1[i] = allocVar[i]


decvarVec = np.concatenate((distaccessVar, allocVar1, allocVar2))
print(np.sum(decvarVec*objVec))
print(df_main.iloc[ind]['IPRPobj'])

# Now build a new objective, using element interpolations
def getObjVec_elemental(l_TN, m1_TN, m2_TN, l_SN, m1_SN, m2_SN):
    return np.concatenate((np.array(l_TN), np.array(m1_TN), np.array(m2_TN),
                           np.array(l_SN), np.array(m1_SN), np.array(m2_SN)))

objVec_elem = getObjVec_elemental(lvec_TN, m1vec_TN, m2vec_TN, lvec_SN, m1vec_SN, m2vec_SN)

store_Uelem, store_Uoracle, store_IPRP = [], [], []

for ind in range(len(df_main)):
    distaccessVar = np.array(eval(df_main.iloc[ind]['DistAccessBinaryVec']))
    allocVar = df_main.iloc[ind]['Allocation']
    allocVar1, allocVar2 = np.zeros(numTN), np.zeros(numTN)
    for i in range(numTN):
        if allocVar[i] > juncvec_TN[i]:
            allocVar1[i] = juncvec_TN[i] - 1
            allocVar2[i] = allocVar[i] - juncvec_TN[i] + 1
        elif allocVar[i] > 0:
            allocVar1[i] = allocVar[i]
    # Build SN allocations through Q
    SNallocVar = np.matmul(allocVar, Q)
    SNallocVar1, SNallocVar2 = np.zeros(numSN), np.zeros(numSN)
    for j in range(numSN):
        if SNallocVar[j] > juncvec_SN[j]:
            SNallocVar1[j] = juncvec_SN[j] - 1
            SNallocVar2[j] = SNallocVar[j] - juncvec_SN[j] + 1
        elif SNallocVar[j] > 0:
            SNallocVar1[j] = SNallocVar[j]
    SNdistaccessVar = np.zeros(numSN)
    for j in range(numSN):
        if SNallocVar[j] > 0:
            SNdistaccessVar[j] = 1

    decvarVec_elem = np.concatenate((distaccessVar, allocVar1, allocVar2,
                                 SNdistaccessVar, SNallocVar1, SNallocVar2))
    store_Uelem.append(np.sum(decvarVec_elem*objVec_elem))
    store_Uoracle.append(df_main.iloc[ind]['Uoracle'])
    store_IPRP.append(df_main.iloc[ind]['IPRPobj'])
    # print(np.sum(decvarVec_elem*objVec_elem))
    # print(df_main.iloc[ind]['IPRPobj'])
    # print(df_main.iloc[ind]['Uoracle'])


plt.plot(range(len(df_main)), store_IPRP, '^', label='IPRP obj. (UB)')
plt.plot(range(len(df_main)), store_Uelem, 'x', label='Decomp. est.')
plt.plot(range(len(df_main)), store_Uoracle, 'o', label='Utility')
plt.legend()
plt.show()

plt.plot(store_Uoracle, store_IPRP, '^', label='IPRP obj. (UB)')
plt.plot(store_Uoracle, store_Uelem, 'x', label='Decomp. est.')
plt.plot(store_Uoracle, store_Uoracle, '-')
plt.xlabel('Utility')
plt.ylabel('Estimate')
plt.ylim([1.8,3.1])
plt.xlim([1.8,3.1])
plt.legend()
plt.show()