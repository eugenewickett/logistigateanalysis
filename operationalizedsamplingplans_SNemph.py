from logistigate.logistigate import utilities as util # Pull from the submodule "develop" branch
from logistigate.logistigate import methods
from logistigate.logistigate.priors import prior_normal_assort
from logistigate.logistigate import lossfunctions as lf
from logistigate.logistigate import samplingplanfunctions as sampf
from logistigate.logistigate import orienteering as opf

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

plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.family"] = "serif"

# Pull data from analysis of first paper
def GetSenegalDataMatrices(deidentify=False):
    # Pull Senegal data from MQDB
    SCRIPT_DIR = os.getcwd()
    filesPath = os.path.join(SCRIPT_DIR, 'MQDfiles')
    outputFileName = os.path.join(filesPath, 'pickleOutput')
    openFile = open(outputFileName, 'rb')  # Read the file
    dataDict = pickle.load(openFile)

    SEN_df = dataDict['df_SEN']
    # 7 unique Province_Name_GROUPED; 23 unique Facility_Location_GROUPED; 66 unique Facility_Name_GROUPED
    # Remove 'Missing' and 'Unknown' labels
    SEN_df_2010 = SEN_df[(SEN_df['Date_Received'] == '7/12/2010') & (SEN_df['Manufacturer_GROUPED'] != 'Unknown') & (
                SEN_df['Facility_Location_GROUPED'] != 'Missing')].copy()
    tbl_SEN_G1_2010 = SEN_df_2010[['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    tbl_SEN_G1_2010 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in tbl_SEN_G1_2010]
    tbl_SEN_G2_2010 = SEN_df_2010[['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    tbl_SEN_G2_2010 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in tbl_SEN_G2_2010]

    SEN_df_2010.pivot_table(index=['Manufacturer_GROUPED'], columns=['Final_Test_Conclusion'],
                            aggfunc='size', fill_value=0)
    SEN_df_2010.pivot_table(index=['Province_Name_GROUPED'], columns=['Final_Test_Conclusion'],
                            aggfunc='size', fill_value=0)
    SEN_df_2010.pivot_table(index=['Facility_Location_GROUPED'], columns=['Final_Test_Conclusion'],
                            aggfunc='size', fill_value=0)
    pivoted = SEN_df_2010.pivot_table(index=['Facility_Name_GROUPED'], columns=['Final_Test_Conclusion'],
                                      aggfunc='size', fill_value=0)
    # pivoted[:15]
    # SEN_df_2010['Province_Name_GROUPED'].unique()
    SEN_df_2010[SEN_df_2010['Province_Name_GROUPED'].isin(['Dakar', 'Kaffrine', 'Kedougou', 'Kaolack'])].pivot_table(
        index=['Manufacturer_GROUPED'], columns=['Province_Name_GROUPED'], aggfunc='size', fill_value=0)
    SEN_df_2010[SEN_df_2010['Province_Name_GROUPED'].isin(['Matam', 'Kolda', 'Saint Louis'])].pivot_table(
        index=['Manufacturer_GROUPED'], columns=['Province_Name_GROUPED'], aggfunc='size', fill_value=0)
    SEN_df_2010[SEN_df_2010['Province_Name_GROUPED'].isin(['Dakar', 'Kaffrine', 'Kedougou', 'Kaolack']) & SEN_df_2010[
        'Final_Test_Conclusion'].isin(['Fail'])].pivot_table(
        index=['Manufacturer_GROUPED'], columns=['Province_Name_GROUPED', 'Final_Test_Conclusion'],
        aggfunc='size', fill_value=0)
    SEN_df_2010[SEN_df_2010['Province_Name_GROUPED'].isin(['Matam', 'Kolda', 'Saint Louis']) & SEN_df_2010[
        'Final_Test_Conclusion'].isin(['Fail'])].pivot_table(
        index=['Manufacturer_GROUPED'], columns=['Province_Name_GROUPED', 'Final_Test_Conclusion'],
        aggfunc='size', fill_value=0)
    SEN_df_2010[SEN_df_2010['Province_Name_GROUPED'].isin(['Dakar', 'Kaffrine', 'Kedougou', 'Kaolack'])].pivot_table(
        index=['Facility_Location_GROUPED'], columns=['Province_Name_GROUPED'], aggfunc='size', fill_value=0)
    SEN_df_2010[SEN_df_2010['Province_Name_GROUPED'].isin(['Matam', 'Kolda', 'Saint Louis'])].pivot_table(
        index=['Facility_Location_GROUPED'], columns=['Province_Name_GROUPED'], aggfunc='size', fill_value=0)
    SEN_df_2010[SEN_df_2010['Province_Name_GROUPED'].isin(['Dakar', 'Kaffrine', 'Kedougou', 'Kaolack'])].pivot_table(
        index=['Facility_Name_GROUPED'], columns=['Province_Name_GROUPED'], aggfunc='size', fill_value=0)
    SEN_df_2010[SEN_df_2010['Province_Name_GROUPED'].isin(['Dakar', 'Kaffrine'])].pivot_table(
        index=['Facility_Name_GROUPED'], columns=['Province_Name_GROUPED'], aggfunc='size', fill_value=0)
    SEN_df_2010[SEN_df_2010['Province_Name_GROUPED'].isin(['Matam', 'Kolda', 'Saint Louis'])].pivot_table(
        index=['Facility_Name_GROUPED'], columns=['Province_Name_GROUPED'], aggfunc='size', fill_value=0)
    SEN_df_2010[SEN_df_2010['Facility_Name_GROUPED'].isin(['Hopitale Regionale de Koda',
                                                           "Pharmacie Keneya"])].pivot_table(
        index=['Facility_Location_GROUPED'], columns=['Facility_Name_GROUPED'], aggfunc='size', fill_value=0)
    SEN_df_2010[SEN_df_2010['Province_Name_GROUPED'].isin(['Dakar'])].pivot_table(
        index=['Facility_Location_GROUPED'], columns=['Final_Test_Conclusion'], aggfunc='size', fill_value=0)
    SEN_df_2010[SEN_df_2010['Facility_Location_GROUPED'].isin(['Tambacounda'])].pivot_table(
        index=['Manufacturer_GROUPED'], columns=['Final_Test_Conclusion'], aggfunc='size', fill_value=0)

    SEN_df_2010['Facility_Location_GROUPED'].count()

    orig_MANUF_lst = ['Ajanta Pharma Limited', 'Aurobindo Pharmaceuticals Ltd', 'Bliss Gvis Pharma Ltd', 'Cipla Ltd',
                      'Cupin', 'EGR pharm Ltd', 'El Nasr', 'Emcure Pharmaceuticals Ltd', 'Expharm',
                      'F.Hoffmann-La Roche Ltd', 'Gracure Pharma Ltd', 'Hetdero Drugs Limited', 'Imex Health',
                      'Innothera Chouzy', 'Ipca Laboratories', 'Lupin Limited', 'Macleods Pharmaceuticals Ltd',
                      'Matrix Laboratories Limited', 'Medico Remedies Pvt Ltd', 'Mepha Ltd', 'Novartis', 'Odypharm Ltd',
                      'Pfizer', 'Sanofi Aventis', 'Sanofi Synthelabo']
    orig_PROV_lst = ['Dakar', 'Kaffrine', 'Kaolack', 'Kedougou', 'Kolda', 'Matam', 'Saint Louis']
    orig_LOCAT_lst = ['Dioum', 'Diourbel', 'Fann- Dakar', 'Guediawaye', 'Hann', 'Kaffrine (City)', 'Kanel',
                      'Kaolack (City)', 'Kebemer', 'Kedougou (City)', 'Kolda (City)', 'Koumpantoum', 'Matam (City)',
                      'Mbour-Thies', 'Medina', 'Ouro-Sogui', 'Richard Toll', 'Rufisque-Dakar', 'Saint Louis (City)',
                      'Tambacounda', 'Thies', 'Tivaoune', 'Velingara']
    # DEIDENTIFICATION
    if deidentify == True:
        # Replace Manufacturers
        shuf_MANUF_lst = orig_MANUF_lst.copy()
        random.seed(333)
        random.shuffle(shuf_MANUF_lst)
        # print(shuf_MANUF_lst)
        for i in range(len(shuf_MANUF_lst)):
            currName = shuf_MANUF_lst[i]
            newName = 'Mnfr. ' + str(i + 1)
            for ind, item in enumerate(tbl_SEN_G1_2010):
                if item[1] == currName:
                    tbl_SEN_G1_2010[ind][1] = newName
            for ind, item in enumerate(tbl_SEN_G2_2010):
                if item[1] == currName:
                    tbl_SEN_G2_2010[ind][1] = newName
        # Replace Province
        shuf_PROV_lst = orig_PROV_lst.copy()
        random.seed(333)
        random.shuffle(shuf_PROV_lst)
        # print(shuf_PROV_lst)
        for i in range(len(shuf_PROV_lst)):
            currName = shuf_PROV_lst[i]
            newName = 'Province ' + str(i + 1)
            for ind, item in enumerate(tbl_SEN_G1_2010):
                if item[0] == currName:
                    tbl_SEN_G1_2010[ind][0] = newName
        # Replace Facility Location
        shuf_LOCAT_lst = orig_LOCAT_lst.copy()
        random.seed(333)
        random.shuffle(shuf_LOCAT_lst)
        # print(shuf_LOCAT_lst)
        for i in range(len(shuf_LOCAT_lst)):
            currName = shuf_LOCAT_lst[i]
            newName = 'District ' + str(i + 1)
            for ind, item in enumerate(tbl_SEN_G2_2010):
                if item[0] == currName:
                    tbl_SEN_G2_2010[ind][0] = newName
        # Swap Districts 7 & 8
        for ind, item in enumerate(tbl_SEN_G2_2010):
            if item[0] == 'District 7':
                tbl_SEN_G2_2010[ind][0] = 'District 8'
            elif item[0] == 'District 8':
                tbl_SEN_G2_2010[ind][0] = 'District 7'

    # Now form data dictionary
    retDict = util.testresultsfiletotable(tbl_SEN_G2_2010, csvName=False)
    if deidentify == True:
        retlist_MANUF = shuf_MANUF_lst.copy()
        retlist_PROV = shuf_PROV_lst.copy()
        retlist_LOCAT = shuf_LOCAT_lst.copy()
    else:
        retlist_MANUF = orig_MANUF_lst.copy()
        retlist_PROV = orig_PROV_lst.copy()
        retlist_LOCAT = orig_LOCAT_lst.copy()

    return retDict['N'], retDict['Y'], retlist_MANUF, retlist_PROV, retlist_LOCAT

# Pull data from newly constructed CSV files
def GetSenegalCSVData():
    """
    Travel out-and-back times for districts/departments are expressed as the proportion of a 10-hour workday, and
    include a 30-minute collection time; traveling to every region outside the HQ region includes a 2.5 hour fixed cost
    """
    dept_df = pd.read_csv('operationalizedsamplingplans/senegal_csv_files/deptfixedcosts.csv', header=0)
    regcost_mat = pd.read_csv('operationalizedsamplingplans/senegal_csv_files/regarcfixedcosts.csv', header=None)
    regNames = ['Dakar', 'Diourbel', 'Fatick', 'Kaffrine', 'Kaolack', 'Kedougou', 'Kolda', 'Louga', 'Matam',
                'Saint-Louis', 'Sedhiou', 'Tambacounda', 'Thies', 'Ziguinchor']
    # Get testing results
    testresults_df = pd.read_csv('operationalizedsamplingplans/senegal_csv_files/dataresults.csv', header=0)
    manufNames = testresults_df.Manufacturer.sort_values().unique().tolist()
    deptNames = dept_df['Department'].sort_values().tolist()
    testdatadict = {'dataTbl': testresults_df.values.tolist(), 'type': 'Tracked', 'TNnames': deptNames,
                    'SNnames': manufNames}
    testdatadict = util.GetVectorForms(testdatadict)

    return dept_df, regcost_mat, regNames, deptNames, manufNames, len(regNames), testdatadict

dept_df, regcost_mat, regNames, deptNames, manufNames, numReg, testdatadict = GetSenegalCSVData()
(numTN, numSN) = testdatadict['N'].shape # For later use

def GetRegion(dept_str, dept_df):
    """Retrieves the region associated with a department"""
    return dept_df.loc[dept_df['Department']==dept_str,'Region'].values[0]

def GetDeptChildren(reg_str, dept_df):
    """Retrieves the departments associated with a region"""
    return dept_df.loc[dept_df['Region']==reg_str,'Department'].values.tolist()

def PrintDataSummary(datadict):
    """print data summaries for datadict which should have keys 'N' and 'Y' """
    N, Y = datadict['N'], datadict['Y']
    # Overall data
    print('TNs by SNs: ' + str(N.shape) + '\nNumber of Obsvns: ' + str(N.sum()) + '\nNumber of SFPs: ' + str(
        Y.sum()) + '\nSFP rate: ' + str(round(Y.sum() / N.sum(), 4)))
    # TN-specific data
    print('Tests at TNs: ' + str(np.sum(N, axis=1)) + '\nSFPs at TNs: ' + str(np.sum(Y, axis=1)) + '\nSFP rates: '+str(
            (np.sum(Y, axis=1) / np.sum(N, axis=1)).round(4)))
    return
# printDataSummary(testdatadict)

# Set up logistigate dictionary
lgdict = util.initDataDict(testdatadict['N'], testdatadict['Y'])
lgdict.update({'TNnames':deptNames, 'SNnames':manufNames})

def SetupSenegalPriors(lgdict, randseed=15):
    """Set up priors for SFP rates at nodes"""
    numTN, numSN = lgdict['TNnum'], lgdict['SNnum']
    # All SNs are `Moderate'
    SNpriorMean = np.repeat(spsp.logit(0.1), numSN)
    # TNs are randomly assigned risk, such that 5% are in the 1st and 7th levels, 10% are in the 2nd and 6th levels,
    #   20% are in the 3rd and 5th levels, and 30% are in the 4th level
    np.random.seed(randseed)
    tempCategs = np.random.multinomial(n=1, pvals=[0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05], size=numTN)
    riskMeans = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25]
    randriskinds = np.mod(np.where(tempCategs.flatten() == 1), len(riskMeans))[0]
    TNpriorMean = spsp.logit(np.array([riskMeans[randriskinds[i]] for i in range(numTN)]))
    # Concatenate prior means
    priorMean = np.concatenate((SNpriorMean, TNpriorMean))
    TNvar, SNvar = 2., 3.  # Variances for use with prior; supply nodes are wider due to unknown risk assessments
    priorCovar = np.diag(np.concatenate((np.repeat(SNvar, numSN), np.repeat(TNvar, numTN))))
    priorObj = prior_normal_assort(priorMean, priorCovar)
    lgdict['prior'] = priorObj
    return

# Set up priors for SFP rates at nodes
SetupSenegalPriors(lgdict)

# Use this function to identify good choice of Madapt
def GetMCMCTracePlots(lgdict, numburnindraws=2000, numdraws=1000):
    """
    Provides a grid of trace plots across all nodes for numdraws draws of the corresponding SFP rates
    """
    # Generate MCMC draws
    templgdict = lgdict.copy()
    templgdict['MCMCdict'].update({'Madapt':numburnindraws, 'numPostSamples': numdraws})
    templgdict = methods.GeneratePostSamples(templgdict, maxTime=5000)
    # Make a grid of subplots
    numTN, numSN = lgdict['TNnum'], lgdict['SNnum']
    dim1 = int(np.ceil(np.sqrt(numTN + numSN)))
    dim2 = int(np.ceil((numTN + numSN) / dim1))

    plotrownum, plotcolnum = 4, 4
    numloops = int(np.ceil((numTN + numSN) / (plotrownum * plotcolnum)))

    currnodeind = 0

    for currloop in range(numloops):
        fig, ax = plt.subplots(nrows=plotrownum, ncols=plotcolnum, figsize=(10,10))
        for row in ax:
            for col in row:
                if currnodeind < numTN + numSN:
                    col.plot(templgdict['postSamples'][:, currnodeind], linewidth=0.5)
                    col.title.set_text('Node ' + str(currnodeind))
                    col.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    col.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    currnodeind += 1
        plt.tight_layout()
        plt.show()

    return
#methods.GetMCMCTracePlots(lgdict, numburnindraws=1000, numdraws=1000)

# Set up MCMC
lgdict['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 1000, 'delta': 0.4}

# Generate batch of MCMC samples
def GenerateMCMCBatch(lgdict, batchsize, randseed, filedest):
    """Generates a batch of MCMC draws and saves it to the specified file destination"""
    lgdict['numPostSamples'] = batchsize
    lgdict = methods.GeneratePostSamples(lgdict, maxTime=5000)
    np.save(filedest, lgdict['postSamples'])
    return
# GenerateMCMCBatch(lgdict, 5000, 300, os.path.join('operationalizedsamplingplans', 'numpy_objects', 'draws1'))

def RetrieveMCMCBatches(lgdict, numbatches, filedest_leadstring):
    """Adds previously generated MCMC draws to lgdict, using the file destination marked by filedest_leadstring"""
    tempobj = np.load(filedest_leadstring + '1.npy')
    for drawgroupind in range(2, numbatches+1):
        newobj = np.load(filedest_leadstring + str(drawgroupind) + '.npy')
        tempobj = np.concatenate((tempobj, newobj))
    lgdict.update({'postSamples': tempobj, 'numPostSamples': tempobj.shape[0]})
    return
# Pull previously generated MCMC draws
RetrieveMCMCBatches(lgdict, 20, os.path.join('operationalizedsamplingplans', 'numpy_objects', 'draws'))
# util.plotPostSamples(lgdict, 'int90')

def AddBootstrapQ(lgdict, numboot, randseed):
    """Add bootstrap-sampled sourcing vectors for unvisited test nodes"""

    numvisitedTNs = np.count_nonzero(np.sum(lgdict['Q'], axis=1))
    SNprobs = np.sum(lgdict['N'], axis=0) / np.sum(lgdict['N'])
    np.random.seed(randseed)
    Qvecs = np.random.multinomial(numboot, SNprobs, size=lgdict['TNnum'] - numvisitedTNs) / numboot
    Qindcount = 0
    tempQ = lgdict['Q'].copy()
    for i in range(tempQ.shape[0]):
        if lgdict['Q'][i].sum() == 0:
            tempQ[i] = Qvecs[Qindcount]
            Qindcount += 1
    lgdict.update({'Q': tempQ})
    return
# Add boostrap-sampled sourcing vectors for non-tested test nodes
AddBootstrapQ(lgdict, numboot=20, randseed=44)

# Loss specification
# TODO: INSPECT CHOICE HERE LATER, ESP MARKETVEC
markVec = np.concatenate((np.ones(numSN)*10, np.ones(numTN)))
paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=markVec)

def SetupParameterDictionary(paramdict, numtruthdraws, numdatadraws, randseed):
    """Sets up parameter dictionary with desired truth and data draws"""
    np.random.seed(randseed)
    truthdraws, datadraws = util.distribute_truthdata_draws(lgdict['postSamples'], numtruthdraws, numdatadraws)
    paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
    paramdict.update({'baseloss': sampf.baseloss(paramdict['truthdraws'], paramdict)})
    return
# Set up parameter dictionary
SetupParameterDictionary(paramdict, 100000, 300, randseed=56)
util.print_param_checks(paramdict)  # Check of used parameters

# Non-importance sampling estimate of utility
def getUtilityEstimate(n, lgdict, paramdict, zlevel=0.95):
    """
    Return a utility estimate average and confidence interval for allocation array n
    """
    testnum = int(np.sum(n))
    des = n/testnum
    currlosslist = sampf.sampling_plan_loss_list(des, testnum, lgdict, paramdict)
    currloss_avg, currloss_CI = sampf.process_loss_list(currlosslist, zlevel=zlevel)
    return paramdict['baseloss'] - currloss_avg, (paramdict['baseloss']-currloss_CI[1], paramdict['baseloss']-currloss_CI[0])

# For identifying benchmark allocations
def FindTSPPathForGivenNodes(reglist, f_reg):
    """
    Returns an sequence of indices corresponding to the shortest path through all indices, per the traversal costs
    featured in f_reg; uses brute force, so DO NOT use with lists larger than 10 elements or so
    Uses first index as the HQ region, and assumes all paths must start and end at this region
    """
    HQind = reglist[0]
    nonHQindlist = reglist[1:]
    permutlist = list(itertools.permutations(nonHQindlist))
    currbestcost = np.inf
    currbesttup = 0
    for permuttuple in permutlist:
        currind = HQind
        currpermutcost = 0
        for ind in permuttuple:
            currpermutcost += f_reg[currind, ind]
            currind = ind
        currpermutcost += f_reg[currind,HQind]
        if currpermutcost < currbestcost:
            currbestcost = currpermutcost
            currbesttup = permuttuple
    besttuplist = [currbesttup[i] for i in range(len(currbesttup))]
    besttuplist.insert(0,HQind)
    return besttuplist, currbestcost

def GetAllocVecFromLists(distNames, distList, allocList):
    """Function for generating allocation vector for benchmarks, only using names and a list of test levels"""
    numDist = len(distNames)
    n = np.zeros(numDist)
    for distElem, dist in enumerate(distList):
        distind = distNames.index(dist)
        n[distind] = allocList[distElem]
    return n

# Orienteering parameters
batchcost, batchsize, B, ctest = 0, 700, 700, 2
batchsize = B
bigM = B*ctest

dept_df_sort = dept_df.sort_values('Department')

FTEcostperday = 200
f_dept = np.array(dept_df_sort['DeptFixedCostDays'].tolist())*FTEcostperday
f_reg = np.array(regcost_mat)*FTEcostperday

##########################
##########################
# Calculate utility for candidates and benchmarks
##########################
##########################

util.print_param_checks(paramdict)

### Benchmarks ###
# LeastVisited
deptList_LeastVisited = ['Keur Massar', 'Pikine', 'Bambey', 'Mbacke', 'Fatick', 'Foundiougne', 'Gossas']
allocList_LeastVisited = [20, 20, 20, 19, 19, 19, 19]
n_LeastVisited = GetAllocVecFromLists(deptNames, deptList_LeastVisited, allocList_LeastVisited)
util_LeastVisited_unif, util_LeastVisited_unif_CI = sampf.getImportanceUtilityEstimate(n_LeastVisited, lgdict,
                                                                paramdict, numimportdraws=50000)
print('LeastVisited:',util_LeastVisited_unif, util_LeastVisited_unif_CI)
# 1-APR
#

# MostSFPs (uniform)
deptList_MostSFPs_unif = ['Dakar', 'Guediawaye', 'Diourbel', 'Saint-Louis', 'Podor']
allocList_MostSFPs_unif = [20, 19, 19, 19, 19]
n_MostSFPs_unif = GetAllocVecFromLists(deptNames, deptList_MostSFPs_unif, allocList_MostSFPs_unif)
util_MostSFPs_unif, util_MostSFPs_unif_CI = sampf.getImportanceUtilityEstimate(n_MostSFPs_unif, lgdict,
                                                                paramdict, numimportdraws=50000)
print('MostSFPs (unform):',util_MostSFPs_unif, util_MostSFPs_unif_CI)
# 1-APR
#

# MostSFPs (weighted)
deptList_MostSFPs_wtd = ['Dakar', 'Guediawaye', 'Diourbel', 'Saint-Louis', 'Podor']
allocList_MostSFPs_wtd = [15, 19, 12, 14, 36]
n_MostSFPs_wtd = GetAllocVecFromLists(deptNames, deptList_MostSFPs_wtd, allocList_MostSFPs_wtd)
util_MostSFPs_wtd, util_MostSFPs_wtd_CI = sampf.getImportanceUtilityEstimate(n_MostSFPs_wtd, lgdict,
                                                                paramdict, numimportdraws=50000)
print('MostSFPs (weighted):', util_MostSFPs_wtd, util_MostSFPs_wtd_CI)
# 1-APR
#

# MoreDistricts (uniform)
deptList_MoreDist_unif = ['Dakar', 'Guediawaye', 'Keur Massar', 'Pikine', 'Rufisque', 'Thies',
                          'Mbour', 'Tivaoune', 'Diourbel', 'Bambey', 'Mbacke']
allocList_MoreDist_unif = [9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8]
n_MoreDist_unif = GetAllocVecFromLists(deptNames, deptList_MoreDist_unif, allocList_MoreDist_unif)
util_MoreDist_unif, util_MoreDist_unif_CI = sampf.getImportanceUtilityEstimate(n_MoreDist_unif, lgdict,
                                                                paramdict, numimportdraws=50000)
print('MoreDistricts (unform):', util_MoreDist_unif, util_MoreDist_unif_CI)
# 1-APR
#

# MoreDistricts (weighted)
deptList_MoreDist_wtd = ['Dakar', 'Guediawaye', 'Keur Massar', 'Pikine', 'Rufisque', 'Thies',
                          'Mbour', 'Tivaoune', 'Diourbel', 'Bambey', 'Mbacke']
allocList_MoreDist_wtd = [6, 5, 13, 13, 6, 5, 6, 7, 5, 13, 13]
n_MoreDist_wtd = GetAllocVecFromLists(deptNames, deptList_MoreDist_wtd, allocList_MoreDist_wtd)
util_MoreDist_wtd, util_MoreDist_wtd_CI = sampf.getImportanceUtilityEstimate(n_MoreDist_wtd, lgdict,
                                                                paramdict, numimportdraws=50000)
print('MoreDistricts (weighted):', util_MoreDist_wtd, util_MoreDist_wtd_CI)
# 1-APR
#

# MoreTests (uniform)
deptList_MoreTests_unif = ['Dakar', 'Guediawaye', 'Keur Massar', 'Pikine', 'Rufisque', 'Thies',
                          'Mbour', 'Tivaoune']
allocList_MoreTests_unif = [22, 22, 22, 22, 22, 22, 22, 22]
n_MoreTests_unif = GetAllocVecFromLists(deptNames, deptList_MoreTests_unif, allocList_MoreTests_unif)
util_MoreTests_unif, util_MoreTests_unif_CI = sampf.getImportanceUtilityEstimate(n_MoreTests_unif, lgdict,
                                                                paramdict, numimportdraws=50000)
print('MostTests (unform):', util_MoreTests_unif, util_MoreTests_unif_CI)
# 1-APR
#

# MoreTests (weighted)
deptList_MoreTests_wtd = ['Dakar', 'Guediawaye', 'Keur Massar', 'Pikine', 'Rufisque', 'Thies',
                          'Mbour', 'Tivaoune']
allocList_MoreTests_wtd = [13, 14, 43, 43, 15, 14, 15, 19]
n_MoreTests_wtd = GetAllocVecFromLists(deptNames, deptList_MoreTests_wtd, allocList_MoreTests_wtd)
util_MoreTests_wtd, util_MoreTests_wtd_CI = sampf.getImportanceUtilityEstimate(n_MoreTests_wtd, lgdict,
                                                                paramdict, numimportdraws=50000)
print('MoreTests (weighted):', util_MoreTests_wtd, util_MoreTests_wtd_CI)
# 1-APR
#

#######
# B=1400
#######

# IP-RP allocation
deptList_IPRP = ['Dakar', 'Keur Massar', 'Pikine', 'Louga', 'Linguere', 'Kaolack', 'Guinguineo',
                 'Nioro du Rip', 'Kaffrine', 'Birkilane', 'Malem Hoddar', 'Bambey', 'Mbacke',
                 'Fatick', 'Foundiougne', 'Gossas']
allocList_IPRP = [19, 21, 7, 7, 11, 38, 9, 18, 8, 8, 8, 10, 7, 11, 10, 9]
n_IPRP = GetAllocVecFromLists(deptNames, deptList_IPRP, allocList_IPRP)
util_IPRP, util_IPRP_CI = sampf.getImportanceUtilityEstimate(n_IPRP, lgdict, paramdict,
                                                             numimportdraws=50000)
print('IPRP:',util_IPRP, util_IPRP_CI)
# 2-APR
#

# LeastVisited
deptList_LeastVisited = ['Keur Massar', 'Pikine', 'Louga', 'Linguere', 'Goudiry', 'Guinguineo',
                         'Nioro du Rip', 'Birkilane', 'Koungheul', 'Malem Hoddar', 'Bambey', 'Mbacke',
                         'Fatick', 'Foundiougne', 'Gossas']
allocList_LeastVisited = [5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
n_LeastVisited = GetAllocVecFromLists(deptNames, deptList_LeastVisited, allocList_LeastVisited)
util_LeastVisited_unif, util_LeastVisited_unif_CI = sampf.getImportanceUtilityEstimate(n_LeastVisited, lgdict,
                                                                paramdict, numimportdraws=50000)
print('LeastVisited:',util_LeastVisited_unif, util_LeastVisited_unif_CI)
# 2-APR
#

# MostSFPs (uniform)
deptList_MostSFPs_unif = ['Dakar', 'Guediawaye', 'Tambacounda', 'Koumpentoum', 'Diourbel', 'Saint-Louis',
                          'Podor', 'Kolda', 'Velingara', 'Matam', 'Kanel']
allocList_MostSFPs_unif = [8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7]
n_MostSFPs_unif = GetAllocVecFromLists(deptNames, deptList_MostSFPs_unif, allocList_MostSFPs_unif)
util_MostSFPs_unif, util_MostSFPs_unif_CI = sampf.getImportanceUtilityEstimate(n_MostSFPs_unif, lgdict,
                                                                paramdict, numimportdraws=50000)
print('MostSFPs (unform):',util_MostSFPs_unif, util_MostSFPs_unif_CI)
# 2-APR
#

# MostSFPs (weighted)
deptList_MostSFPs_wtd = ['Dakar', 'Guediawaye', 'Tambacounda', 'Koumpentoum', 'Diourbel', 'Saint-Louis',
                          'Podor', 'Kolda', 'Velingara', 'Matam', 'Kanel']
allocList_MostSFPs_wtd = [6, 8, 6, 8, 5, 5, 14, 5, 9, 6, 8]
n_MostSFPs_wtd = GetAllocVecFromLists(deptNames, deptList_MostSFPs_wtd, allocList_MostSFPs_wtd)
util_MostSFPs_wtd, util_MostSFPs_wtd_CI = sampf.getImportanceUtilityEstimate(n_MostSFPs_wtd, lgdict,
                                                                paramdict, numimportdraws=50000)
print('MostSFPs (weighted):', util_MostSFPs_wtd, util_MostSFPs_wtd_CI)
# 2-APR
#

# MoreDistricts (uniform)
deptList_MoreDist_unif = ['Dakar', 'Guediawaye', 'Keur Massar', 'Pikine', 'Rufisque', 'Thies',
                          'Mbour', 'Tivaoune', 'Kaolack', 'Guinguineo', 'Nioro du Rip', 'Kaffrine',
                          'Birkilane', 'Koungheul', 'Malem Hoddar',  'Diourbel', 'Bambey', 'Mbacke',
                          'Fatick', 'Foundiougne', 'Gossas']
allocList_MoreDist_unif = [8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 7, 7]
n_MoreDist_unif = GetAllocVecFromLists(deptNames, deptList_MoreDist_unif, allocList_MoreDist_unif)
util_MoreDist_unif, util_MoreDist_unif_CI = sampf.getImportanceUtilityEstimate(n_MoreDist_unif, lgdict,
                                                                paramdict, numimportdraws=50000)
print('MoreDistricts (unform):', util_MoreDist_unif, util_MoreDist_unif_CI)
# 2-APR
#

# MoreDistricts (weighted)
deptList_MoreDist_wtd = ['Dakar', 'Guediawaye', 'Keur Massar', 'Pikine', 'Rufisque', 'Thies',
                          'Mbour', 'Tivaoune', 'Kaolack', 'Guinguineo', 'Nioro du Rip', 'Kaffrine',
                          'Birkilane', 'Koungheul', 'Malem Hoddar',  'Diourbel', 'Bambey', 'Mbacke',
                          'Fatick', 'Foundiougne', 'Gossas']
allocList_MoreDist_wtd = [4, 5, 9, 9, 5, 5, 5, 6, 5, 9, 9, 7, 9, 9, 9, 4, 10, 10, 10, 10, 10]
n_MoreDist_wtd = GetAllocVecFromLists(deptNames, deptList_MoreDist_wtd, allocList_MoreDist_wtd)
util_MoreDist_wtd, util_MoreDist_wtd_CI = sampf.getImportanceUtilityEstimate(n_MoreDist_wtd, lgdict,
                                                                paramdict, numimportdraws=50000)
print('MoreDistricts (weighted):', util_MoreDist_wtd, util_MoreDist_wtd_CI)
# 2-APR
#

# MoreTests (uniform)
deptList_MoreTests_unif = ['Dakar', 'Guediawaye', 'Keur Massar', 'Pikine', 'Rufisque', 'Thies', 'Mbour',
                           'Tivaoune', 'Diourbel', 'Bambey', 'Mbacke', 'Fatick', 'Foundiougne', 'Gossas']
allocList_MoreTests_unif = [27, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26]
n_MoreTests_unif = GetAllocVecFromLists(deptNames, deptList_MoreTests_unif, allocList_MoreTests_unif)
util_MoreTests_unif, util_MoreTests_unif_CI = sampf.getImportanceUtilityEstimate(n_MoreTests_unif, lgdict,
                                                                paramdict, numimportdraws=50000)
print('MostTests (unform):', util_MoreTests_unif, util_MoreTests_unif_CI)
# 2-APR
#

# MoreTests (weighted)
deptList_MoreTests_wtd = ['Dakar', 'Guediawaye', 'Keur Massar', 'Pikine', 'Rufisque', 'Thies', 'Mbour',
                           'Tivaoune', 'Diourbel', 'Bambey', 'Mbacke', 'Fatick', 'Foundiougne', 'Gossas']
allocList_MoreTests_wtd = [15, 16, 36, 36, 16, 16, 16, 19, 15, 36, 36, 36, 36, 36]
n_MoreTests_wtd = GetAllocVecFromLists(deptNames, deptList_MoreTests_wtd, allocList_MoreTests_wtd)
util_MoreTests_wtd, util_MoreTests_wtd_CI = sampf.getImportanceUtilityEstimate(n_MoreTests_wtd, lgdict,
                                                                paramdict, numimportdraws=50000)
print('MoreTests (weighted):', util_MoreTests_wtd, util_MoreTests_wtd_CI)
# 2-APR
#


##########################
##########################
# END calculate utility for candidates and benchmarks
##########################
##########################

#######
# Generate interpolation values
#######
maxpriortests = int(np.max(np.sum(lgdict['N'],axis=1)))
deptallocbds = np.array([maxpriortests for i in range(len(deptNames))])

util_lo, util_lo_CI = [], []
util_hi_imp, util_hi_CI_imp = [], []
loparamdict = paramdict.copy()
lotruthdraws, lodatadraws = util.distribute_truthdata_draws(lgdict['postSamples'], lgdict['postSamples'].shape[0], 1000)
loparamdict.update({'truthdraws': lotruthdraws, 'datadraws': lodatadraws})
for i in range(len(deptNames)):
    currbd = int(deptallocbds[i])
    n = np.zeros(numTN)
    print('Getting utility for ' + deptNames[i] + ', at 1 test...')
    n[i] = 1
    currlo, currlo_CI = getUtilityEstimate(n, lgdict, loparamdict)
    print(currlo, currlo_CI)
    util_lo.append(currlo)
    util_lo_CI.append(currlo_CI)
    print('Getting utility for ' + deptNames[i] + ', at ' + str(currbd) + ' tests...')
    n[i] = currbd
    currhi_imp, currhi_CI_imp = sampf.getImportanceUtilityEstimate(n, lgdict, paramdict, numimportdraws=50000)
    print(currhi_imp, currhi_CI_imp)
    util_hi_imp.append(currhi_imp)
    util_hi_CI_imp.append(currhi_CI_imp)
'''
# 1st run: 30k imp draws
util_hi_imp = [0.3906806364519433, 0.37900277450375164, 0.3125001286929532, 0.29495861975553517, 0.1484237775864603, 0.11712893687658266, 0.2491887695057997, 0.18748908632695382, 0.37597343292086194, 0.3138836003083476, 0.37269612148611486, 0.26872878180024884, 0.35535587179775696, 0.16475204125187837, 0.24957612548629804, 0.16849467150581354, 0.16117134264092492, 0.2886873340956626, 0.15584807336130346, 0.19014490579459498, 0.3088897890583908, 0.16660470044417508, 0.12478914780106365, 0.36051397028615995, 0.2885597926524781, 0.3547341887416735, 0.35240183304921757, 0.16846768737411466, 0.28431504530094287, 0.15322035049142357, 0.31073053682450613, 0.3497771893559207, 0.23278279534773816, 0.30563854854423056, 0.2656809525105466, 0.35366903855184617, 0.0806069387652677, 0.1263231406305021, 0.2527240768625827, 0.35659808113103253, 0.25586223074083314, 0.1431744020501, 0.08642080227579108, 0.19829483675732185, 0.2071626763841259, 0.2899497353746412]
util_hi_CI_imp = [(0.38595297319382027, 0.3954082997100663), (0.3737541827074189, 0.3842513663000844), (0.30760789372496156, 0.3173923636609448), (0.28937533040854113, 0.3005419091025292), (0.1453044707311033, 0.1515430844418173), (0.1149831792953222, 0.11927469445784311), (0.2444668838519597, 0.25391065515963973), (0.1827902244125621, 0.19218794824134555), (0.3698319690289136, 0.3821148968128103), (0.3075489483672875, 0.3202182522494077), (0.3675744038969153, 0.3778178390753144), (0.26498536578303344, 0.27247219781746423), (0.35097738641657195, 0.359734357178942), (0.16173954667756085, 0.16776453582619588), (0.2458965718139794, 0.2532556791586167), (0.16611500466456697, 0.17087433834706012), (0.15744929656400686, 0.16489338871784298), (0.2837714137486138, 0.2936032544427114), (0.15367938221318767, 0.15801676450941926), (0.18710790492263385, 0.1931819066665561), (0.30331019705173645, 0.3144693810650452), (0.16192871188237667, 0.17128068900597349), (0.12145381603180816, 0.12812447957031914), (0.3545193854481834, 0.3665085551241365), (0.2840071496511545, 0.29311243565380174), (0.34993061230100153, 0.3595377651823455), (0.347354331763265, 0.3574493343351701), (0.16491138703167785, 0.17202398771655147), (0.28005389912150314, 0.2885761914803826), (0.1489806549171444, 0.15746004606570274), (0.3064495556393023, 0.31501151800970995), (0.34505136122100843, 0.354503017490833), (0.22828351222859844, 0.23728207846687788), (0.30202778632840577, 0.30924931076005535), (0.2618305305709505, 0.26953137445014264), (0.35012701544012614, 0.3572110616635662), (0.0786704100236264, 0.082543467506909), (0.12290000760660647, 0.12974627365439773), (0.2486159239967929, 0.2568322297283725), (0.3512152326217226, 0.36198092964034245), (0.25096023628613295, 0.26076422519553333), (0.13971234526516696, 0.14663645883503307), (0.08370010288294871, 0.08914150166863344), (0.19418115614387332, 0.20240851737077037), (0.20394652854265694, 0.21037882422559484), (0.2860116507042836, 0.2938878200449988)]
# 2nd run: 50k imp draws

'''
util_df = pd.DataFrame({'DeptName':deptNames,'Bounds':deptallocbds,'Util_lo':util_lo, 'Util_lo_CI':util_lo_CI,
                        'Util_hi':util_hi_imp, 'Util_hi_CI':util_hi_CI_imp})

# todo: MODIFY HERE
util_df.to_csv(os.path.join('operationalizedsamplingplans', 'csv_utility', 'utilevals_SNemph.csv'), index=False)
