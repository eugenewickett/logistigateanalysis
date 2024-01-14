from logistigate.logistigate import utilities as util # Pull from the submodule "develop" branch
from logistigate.logistigate import methods
from logistigate.logistigate.priors import prior_normal_assort
from logistigate.logistigate import lossfunctions as lf
from logistigate.logistigate import samplingplanfunctions as sampf

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

    return dept_df, regcost_mat, testresults_df, regNames, manufNames

# Pull district-level Senegal data
# N, Y, SNnames, TNprovs, TNnames = GetSenegalDataMatrices(deidentify=False)
dept_df, regcost_mat, testresults_df, regNames, manufNames = GetSenegalCSVData()
deptNames = dept_df['Department'].sort_values().tolist()
numReg = len(regNames)
testdatadict = {'dataTbl':testresults_df.values.tolist(), 'type':'Tracked', 'TNnames':deptNames, 'SNnames':manufNames}
testdatadict = util.GetVectorForms(testdatadict)
N, Y, TNnames, SNnames = testdatadict['N'], testdatadict['Y'], testdatadict['TNnames'], testdatadict['SNnames']
(numTN, numSN) = N.shape # For later use

def GetRegion(dept_str, dept_df):
    """Retrieves the region associated with a department"""
    return dept_df.loc[dept_df['Department']==dept_str,'Region'].values[0]
def GetDeptChildren(reg_str, dept_df):
    """Retrieves the departments associated with a region"""
    return dept_df.loc[dept_df['Region']==reg_str,'Department'].values.tolist()

##############
### Print some data summaries
# Overall data
print('TNs by SNs: ' + str(N.shape) + '\nNumber of Obsvns: ' + str(N.sum()) + '\nNumber of SFPs: ' + str(Y.sum()) + '\nSFP rate: ' + str(round(
    Y.sum() / N.sum(), 4)))
# TN-specific data
print('Tests at TNs: ' + str(np.sum(N, axis=1)) + '\nSFPs at TNs: ' + str(np.sum(Y, axis=1)) + '\nSFP rates: ' + str(
    (np.sum(Y, axis=1) / np.sum(N, axis=1)).round(4)))

# Set up logistigate dictionary
lgdict = util.initDataDict(N, Y)
lgdict.update({'TNnames':TNnames, 'SNnames':SNnames})

##############
# Set up priors for SFP rates at nodes
# TODO: INSPECT CHOICE HERE LATER
# All SNs are `Moderate'
SNpriorMean = np.repeat(spsp.logit(0.1), numSN)
# TNs are randomly assigned risk, such that 5% are in the 1st and 7th levels, 10% are in the 2nd and 6th levels,
#   20% are in the 3rd and 5th levels, and 30% are in the 4th level
np.random.seed(15)
tempCategs = np.random.multinomial(n=1, pvals=[0.05,0.1,0.2,0.3,0.2,0.1,0.05], size=numTN)
riskMeans = [0.01,0.02,0.05,0.1,0.15,0.2,0.25]
randriskinds = np.mod(np.where(tempCategs.flatten()==1), len(riskMeans))[0]
TNpriorMean = spsp.logit(np.array([riskMeans[randriskinds[i]] for i in range(numTN)]))
# Concatenate prior means
priorMean = np.concatenate((SNpriorMean, TNpriorMean))
TNvar, SNvar = 2., 3.  # Variances for use with prior; supply nodes are wider due to unknown risk assessments
priorCovar = np.diag(np.concatenate((np.repeat(SNvar, numSN), np.repeat(TNvar, numTN))))
priorObj = prior_normal_assort(priorMean, priorCovar)
lgdict['prior'] = priorObj

# Set up MCMC
lgdict['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}

'''
# TODO: INSPECT CHOICE HERE LATER
numdraws = 5000
lgdict['numPostSamples'] = numdraws

np.random.seed(300) # For first 4 sets of 5k draws
np.random.seed(301) # For second 17 sets of 5k draws
np.random.seed(410) # For third 11 sets of 5k draws
np.random.seed(466) # For fourth XX sets of 5k draws

time0 = time.time()
lgdict = methods.GeneratePostSamples(lgdict, maxTime=5000)
print(time.time()-time0)

tempobj = lgdict['postSamples']
np.save(os.path.join('operationalizedsamplingplans', 'numpy_objects', 'draws40'),tempobj)

file_name = "operationalizedsamplingplans/numpy_objects/draws35.npy"
file_stats = os.stat(file_name)
print(f'File Size in MegaBytes is {file_stats.st_size / (1024 * 1024)}')
'''

# Load draws from files
tempobj = np.load(os.path.join('operationalizedsamplingplans', 'numpy_objects', 'draws1.npy'))
for drawgroupind in range(2, 41):
    newobj = np.load(os.path.join('operationalizedsamplingplans', 'numpy_objects', 'draws' + str(drawgroupind) +'.npy'))
    tempobj = np.concatenate((tempobj, newobj))
lgdict['postSamples'] = tempobj
# Print inference from initial data
util.plotPostSamples(lgdict, 'int90')

# Generate Q via bootstrap sampling of known traces
numvisitedTNs = np.count_nonzero(np.sum(lgdict['Q'],axis=1))
numboot = 20 # Average across each department in original data
SNprobs = np.sum(lgdict['N'], axis=0) / np.sum(lgdict['N']) # SN sourcing probabilities across original data
np.random.seed(44)
Qvecs = np.random.multinomial(numboot, SNprobs, size=numTN - numvisitedTNs) / numboot
# Only update rows with no observed traces
Qindcount = 0
tempQ = lgdict['Q'].copy()
for i in range(tempQ.shape[0]):
    if lgdict['Q'][i].sum() == 0:
        tempQ[i] = Qvecs[Qindcount]
        Qindcount += 1
lgdict.update({'Q':tempQ})

# Loss specification
# TODO: INSPECT CHOICE HERE LATER, ESP MARKETVEC
paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN))

# Set MCMC draws to use in fast algorithm
numtruthdraws, numdatadraws = 200000, 100
# Get random subsets for truth and data draws
np.random.seed(56)
truthdraws, datadraws = util.distribute_truthdata_draws(lgdict['postSamples'], numtruthdraws, numdatadraws)
paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
# Get base loss
paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)

util.print_param_checks(paramdict)  # Check of used parameters

def getUtilityEstimate(n, lgdict, paramdict, zlevel=0.95):
    """
    Return a utility estimate average and confidence interval for allocation array n
    """
    testnum = int(np.sum(n))
    des = n/testnum
    currlosslist = sampf.sampling_plan_loss_list(des, testnum, lgdict, paramdict)
    currloss_avg, currloss_CI = sampf.process_loss_list(currlosslist, zlevel=zlevel)
    return paramdict['baseloss'] - currloss_avg, (paramdict['baseloss']-currloss_CI[1], paramdict['baseloss']-currloss_CI[0])


'''
time0 = time.time()
utilavg, (utilCIlo, utilCIhi) = getUtilityEstimate(n, lgdict, paramdict)
print(time.time() - time0)

With numtruthdraws, numdatadraws = 10000, 500:
~160 seconds
utilavg, (utilCIlo, utilCIhi) =
0.4068438943300112, (0.3931478722114097, 0.42053991644861277)
0.42619338638365, (0.40593452427234133, 0.4464522484949587)
'''
##################
# Now set up functions for constraints and variables of our program
##################
# Set these parameters per the program described in the paper
# TODO: INSPECT CHOICES HERE LATER, ESP bigM
batchcost, batchsize, B, ctest = 0, 700, 700, 2
batchsize = B
bigM = B*ctest

dept_df_sort = dept_df.sort_values('Department')

FTEcostperday = 200
f_dept = np.array(dept_df_sort['DeptFixedCostDays'].tolist())*FTEcostperday
f_reg = np.array(regcost_mat)*FTEcostperday

optparamdict = {'batchcost':batchcost, 'budget':B, 'pertestcost':ctest, 'Mconstant':bigM, 'batchsize':batchsize,
                'deptfixedcostvec':f_dept, 'arcfixedcostmat': f_reg, 'reghqname':'Dakar', 'reghqind':0,
                'deptnames':deptNames, 'regnames':regNames, 'dept_df':dept_df_sort}

# What are the upper bounds for our allocation variables?
def GetUpperBounds(optparamdict):
    """Returns a numpy vector of upper bounds for an inputted parameter dictionary"""
    B, f_dept, f_reg = optparamdict['budget'], optparamdict['deptfixedcostvec'], optparamdict['arcfixedcostmat']
    batchcost, ctest, reghqind = optparamdict['batchcost'], optparamdict['pertestcost'], optparamdict['reghqind']
    deptnames, regnames, dept_df = optparamdict['deptnames'], optparamdict['regnames'], optparamdict['dept_df']
    retvec = np.zeros(f_dept.shape[0])
    for i in range(f_dept.shape[0]):
        regparent = GetRegion(deptnames[i], dept_df)
        regparentind = regnames.index(regparent)
        if regparentind == reghqind:
            retvec[i] = np.floor((B-f_dept[i]-batchcost)/ctest)
        else:
            regfixedcost = f_reg[reghqind,regparentind] + f_reg[regparentind, reghqind]
            retvec[i] = np.floor((B-f_dept[i]-batchcost-regfixedcost)/ctest)
    return retvec

deptallocbds = GetUpperBounds(optparamdict)
print(deptNames[np.argmin(deptallocbds)], min(deptallocbds))
print(deptNames[np.argmax(deptallocbds)], max(deptallocbds))

# TODO: INSPECT CHOICES HERE LATER
# Example set of variables to inspect validity
v_batch = B
n_alloc = np.zeros(numTN)
n_alloc[36] = 20 # Rufisque, Dakar
n_alloc[25] = 20 # Louga, Louga
n_alloc[24] = 20 # Linguere, Louga
n_alloc[2] = 20 # Bignona, Ziguinchor
n_alloc[32] = 20 # Oussouye, Ziguinchor
n_alloc[8] = 10 # Fatick, Fatick
n_alloc[9] = 10 # Foundiougne, Fatick
n_alloc[10] = 0 # Gossas, Fatick
z_reg = np.zeros(numReg)
z_reg[0] = 1 # Dakar
z_reg[7] = 1 # Louga
z_reg[13] = 1 # Ziguinchor
z_reg[2] = 1 # Fatick
z_dept = np.zeros(numTN)
z_dept[36] = 1 # Rufisque, Dakar
z_dept[25] = 1 # Louga, Louga
z_dept[24] = 1 # Linguere, Louga
z_dept[2] = 1 # Bignona, Ziguinchor
z_dept[32] = 1 # Oussouye, Ziguinchor
z_dept[8] = 1 # Fatick, Fatick
z_dept[9] = 1 # Foundiougne, Fatick
z_dept[10] = 0 # Gossas, Fatick

x = np.zeros((numReg, numReg))
x[0, 7] = 1 # Dakar to Louga
x[7, 13] = 1 # Louga to Ziguinchor
x[13, 2] = 1 # Ziguinchor to Fatick
x[2, 0] = 1 # Fatick to Dakar
# Generate a dictionary for variables
varsetdict = {'batch_int':v_batch, 'regaccessvec_bin':z_reg, 'deptaccessvec_bin':z_dept, 'arcmat_bin':x,
              'allocvec_int':n_alloc}
##########
# Add functions for all constraints; they return True if satisfied, False otherwise
##########
def ConstrBudget(varsetdict, optparamdict):
    """Indicates if the budget constraint is satisfied"""
    flag = False
    budgetcost = varsetdict['batch_int']*optparamdict['batchcost'] + \
        np.sum(varsetdict['deptaccessvec_bin']*optparamdict['deptfixedcostvec']) + \
        np.sum(varsetdict['allocvec_int'] * optparamdict['pertestcost']) + \
        np.sum(varsetdict['arcmat_bin'] * optparamdict['arcfixedcostmat'])
    if budgetcost <= optparamdict['budget']: # Constraint satisfied
        flag = True
    return flag

def ConstrRegionAccess(varsetdict, optparamdict):
    """Indicates if the regional access constraints are satisfied"""
    flag = True
    bigM = optparamdict['Mconstant']
    for aind, a in enumerate(optparamdict['deptnames']):
        parentreg = GetRegion(a, optparamdict['dept_df'])
        parentregind = optparamdict['regnames'].index(parentreg)
        if varsetdict['allocvec_int'][aind] > bigM*varsetdict['regaccessvec_bin'][parentregind]:
            flag = False
    return flag

def ConstrHQRegionAccess(varsetdict, optparamdict):
    """Indicates if the regional HQ access is set"""
    flag = True
    reghqind = optparamdict['reghqind']
    if varsetdict['regaccessvec_bin'][reghqind] != 1:
        flag = False
    return flag

def ConstrLocationAccess(varsetdict, optparamdict):
    """Indicates if the location/department access constraints are satisfied"""
    flag = True
    bigM = optparamdict['Mconstant']
    for aind, a in enumerate(optparamdict['deptnames']):
        if varsetdict['allocvec_int'][aind] > bigM*varsetdict['deptaccessvec_bin'][aind]:
            flag = False
    return flag

def ConstrBatching(varsetdict, optparamdict):
    """Indicates if the location/department access constraints are satisfied"""
    flag = True
    if optparamdict['batchsize']*varsetdict['batch_int'] < np.sum(varsetdict['allocvec_int']):
        flag = False
    return flag

def ConstrArcsLeaveOnce(varsetdict, optparamdict):
    """Each region can only be exited once"""
    flag = True
    x =  varsetdict['arcmat_bin']
    for rind in range(len(optparamdict['regnames'])):
        if np.sum(x[rind]) > 1:
            flag = False
    return flag

def ConstrArcsPassThruHQ(varsetdict, optparamdict):
    """Path must pass through the HQ region"""
    flag = True
    x =  varsetdict['arcmat_bin']
    reghqind = optparamdict['reghqind']
    reghqsum = np.sum(x[reghqind])*optparamdict['Mconstant']
    if np.sum(x) > reghqsum:
        flag = False
    return flag

def ConstrArcsFlowBalance(varsetdict, optparamdict):
    """Each region must be entered and exited the same number of times"""
    flag = True
    x =  varsetdict['arcmat_bin']
    for rind in range(len(optparamdict['regnames'])):
        if np.sum(x[rind]) != np.sum(x[:, rind]):
            flag = False
    return flag

def ConstrArcsRegAccess(varsetdict, optparamdict):
    """Accessed regions must be on the path"""
    flag = True
    x =  varsetdict['arcmat_bin']
    reghqind = optparamdict['reghqind']
    for rind in range(len(optparamdict['regnames'])):
        if (rind != reghqind) and varsetdict['regaccessvec_bin'][rind] > np.sum(x[rind]):
            flag = False
    return flag

def CheckSubtour(varsetdict, optparamdict):
    """Checks if matrix x of varsetdict has multiple tours"""
    x = varsetdict['arcmat_bin']
    tourlist = []
    flag = True
    if np.sum(x) == 0:
        return flag
    else:
        # Start from HQ ind
        reghqind = optparamdict['reghqind']
        tourlist.append(reghqind)
        nextregind = np.where(x[reghqind] == 1)[0][0]
        while nextregind not in tourlist:
            tourlist.append(nextregind)
            nextregind = np.where(x[nextregind] == 1)[0][0]
    if len(tourlist) < np.sum(x):
        flag = False
    return flag

def GetTours(varsetdict, optparamdict):
    """Return a list of lists, each of which is a tour of the arcs matrix in varsetdict"""
    x = varsetdict['arcmat_bin']
    tourlist = []
    flag = True
    tempx = x.copy()
    while np.sum(tempx) > 0:
        currtourlist = GetSubtour(tempx)
        tourlist.append(currtourlist)
        tempx[currtourlist] = tempx[currtourlist]*0
    return tourlist

def GetSubtour(x):
    '''
    Returns a subtour for incidence matrix x
    '''
    tourlist = []
    startind = (np.sum(x, axis=1) != 0).argmax()
    tourlist.append(startind)
    nextind = np.where(x[startind] == 1)[0][0]
    while nextind not in tourlist:
        tourlist.append(nextind)
        nextind = np.where(x[nextind] == 1)[0][0]
    return tourlist

def GetSubtourMaxCardinality(optparamdict):
    """Provide an upper bound on the number of regions included in any tour; HQ region is included"""
    mincostvec = [] # initialize
    dept_df = optparamdict['dept_df']
    ctest, B, batchcost = optparamdict['pertestcost'], optparamdict['budget'], optparamdict['batchcost']
    for r in range(len(optparamdict['regnames'])):
        if r != optparamdict['reghqind']:
            currReg = optparamdict['regnames'][r]
            currmindeptcost = np.max(optparamdict['deptfixedcostvec'])
            deptchildren = GetDeptChildren(currReg, dept_df)
            for currdept in deptchildren:
                currdeptind = optparamdict['deptnames'].index(currdept)
                if optparamdict['deptfixedcostvec'][currdeptind] < currmindeptcost:
                    currmindeptcost = optparamdict['deptfixedcostvec'][currdeptind]
            currminentry = optparamdict['arcfixedcostmat'][np.where(optparamdict['arcfixedcostmat'][:, r] > 0,
                                                                    optparamdict['arcfixedcostmat'][:, r],
                                                                    np.inf).argmin(), r]
            currminexit = optparamdict['arcfixedcostmat'][r, np.where(optparamdict['arcfixedcostmat'][r] > 0,
                                                                    optparamdict['arcfixedcostmat'][r],
                                                                    np.inf).argmin()]
            mincostvec.append(currmindeptcost + currminentry + currminexit + ctest)
        else:
            mincostvec.append(0) # HQ is always included
    # Now add regions until the budget is reached
    currsum = 0
    numregions = 0
    nexttoadd = np.array(mincostvec).argmin()
    while currsum + mincostvec[nexttoadd] <= B - batchcost:
        currsum += mincostvec[nexttoadd]
        numregions += 1
        _ = mincostvec.pop(nexttoadd)
        nexttoadd = np.array(mincostvec).argmin()

    return numregions


def GetTriangleInterpolation(xlist, flist):
    """
    Produces a concave interpolation for integers using the inputs x and function evaluations f_x.
    xlist should have three values: [x_0, x_0 + 1, x_max], and f_x should have evaluations corresponding to these
        three points.
    Returns x and f_x lists for the inclusive range x = [x_0, x_max], as well as intercept l, slope juncture k, and
        slopes m1 and m2
    """
    retx = np.arange(xlist[0], xlist[2]+1)
    # First get left line
    leftlineslope = (flist[1]-flist[0]) / (xlist[1]-xlist[0])
    leftline = leftlineslope * np.array([retx[i]-retx[0] for i in range(retx.shape[0])]) + flist[0]
    # Next get bottom line
    bottomlineslope = (flist[2]-flist[1]) / (xlist[2]-xlist[1])
    bottomline = bottomlineslope * np.array([retx[i] - retx[1] for i in range(retx.shape[0])]) + flist[1]
    # Top line is just the largest value
    topline = np.ones(retx.shape[0]) * flist[2]
    # Upper vals is minimum of left and top lines
    uppervals = np.minimum(leftline, topline)
    # Interpolation is midpoint between upper and bottom lines
    retf = np.average(np.vstack((uppervals, bottomline)),axis=0)
    retf[0] = flist[0]  # Otherwise we are changing the first value

    # Identify slope juncture k, where the line "bends", which is where leftline meets topline
    # k is the first index where the new slope takes hold
    k = leftline.tolist().index( next(x for x in leftline if x > topline[0]))
    # Slopes can be identified using this k
    # todo: WARNING: THIS MIGHT BREAK DOWN FOR EITHER VERY STRAIGHT OR VERY CURVED INTERPOLATIONS
    m1 = retf[k-1] - retf[k-2]
    m2 = retf[k+1] - retf[k]
    # l is the zero intercept, using m1
    l = retf[1] - m1

    return retx, retf, l, k, m1, m2


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
    currind = HQind
    currbesttup = 0
    for permuttuple in permutlist:
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

''' TEST TRIANGLE INTERPOLATION
xlist = [0,1,30]
flist = [0,1,5]

retx, retf = GetTriangleInterpolation(xlist, flist)

plt.plot(retx, retf)
plt.ylim([0,25])
plt.show()
'''


'''
# Here we obtain utility evaluations for 1 and n_bound tests at each department
deptallocbds = GetUpperBounds(optparamdict)
util_lo, util_lo_CI = [], []
util_hi, util_hi_CI = [], []
for i in range(len(deptNames)):
    currbd = int(deptallocbds[i])
    print('Getting utility for ' + deptNames[i] + ', at 1 test...')
    n = np.zeros(numTN)
    n[i] = 1
    currlo, currlo_CI = getUtilityEstimate(n, lgdict, paramdict)
    print(currlo, currlo_CI)
    util_lo.append(currlo)
    util_lo_CI.append(currlo_CI)
    print('Getting utility for ' + deptNames[i] + ', at ' + str(currbd) + ' tests...')
    n[i] = currbd
    currhi, currhi_CI = getUtilityEstimate(n, lgdict, paramdict)
    print(currhi, currhi_CI)
    util_hi.append(currhi)
    util_hi_CI.append(currhi_CI)

util_df = pd.DataFrame({'DeptName':deptNames,'Bounds':deptallocbds,'Util_lo':util_lo, 'Util_lo_CI':util_lo_CI,
                        'Util_hi':util_hi, 'Util_hi_CI':util_hi_CI})

util_df.to_pickle(os.path.join('operationalizedsamplingplans', 'numpy_objects', 'utilevals.pkl'))
'''

# Load previously calculated lower and upper utility evaluations
util_df = pd.read_pickle(os.path.join('operationalizedsamplingplans', 'numpy_objects', 'utilevals.pkl'))

''' RUNS 29-DEC
Bakel       0.03344590816593218 (0.03147292119474443, 0.03541889513711993), 
105         0.43105884100510217 (0.4231565417533414, 0.43896114025686295)
Bambey      0.03446700691774751 (0.031950294730139106, 0.03698371910535592)
269         0.7031888278193428 (0.65970263766636, 0.7466750179723256)
Bignona     0.030668690359265227 (0.02772583763711367, 0.033611543081416784)
140         0.37932139488912675 (0.3699972836620251, 0.3886455061162284)
Birkilane   0.03730134595455681 (0.034801214687281146, 0.03980147722183247)
238         0.4684099921669258 (0.4473053150611417, 0.4895146692727099)
Bounkiling  0.00453539247349255 (0.002933726625375499, 0.006137058321609601)
160         0.19396182550914354 (0.18843369234668472, 0.19948995867160235)
Dagana      0.004221436878848905 (0.003648644574480997, 0.004794229183216814)
195         0.14442075719252045 (0.14172686992181838, 0.14711464446322253)
Dakar       0.005222521081899245 (0.00429602215599445, 0.00614902000780404)
345         0.7332074114707208 (0.6865241407062364, 0.7798906822352052)
Diourbel    0.0023157423073154604 (0.0017679128534258126, 0.002863571761205108)
279         0.4526505991558132 (0.43015147401770193, 0.47514972429392444)
Fatick      0.033057486817286375 (0.030502717710007232, 0.03561225592456552)
273         0.7870853278437675 (0.7379427810124408, 0.8362278746750942)
Foundiougne 0.030612648885227856 (0.028847718182849036, 0.032377579587606675)
262         0.6110077584821685 (0.5739668037064192, 0.6480487132579178)

Gossas      0.04035724365910198 (0.03823470069321111, 0.042479786624992855)
257         0.7768547796500158 (0.7313417312562365, 0.8223678280437952)
Goudiry     0.022376695174909145 (0.0197621780011783, 0.02499121234863999)
152         0.29077391079703396 (0.28527535885379685, 0.2962724627402711)
Goudoump    0.03246600245761755 (0.029396877391432596, 0.0355351275238025)
124         0.3991144287361781 (0.3891039816752997, 0.4091248757970565)
Guediawaye  0.003474985346775483 (0.00289235462568449, 0.004057616067866476)
337         0.3502283652990581 (0.32466299834520385, 0.37579373225291235)
Guinguineo  0.02463908710868168 (0.022144656133576746, 0.027133518083786612)
249         0.3550276878136227 (0.34520036788485875, 0.36485500774238666)
Kaffrine    0.019184733052625802 (0.016111662656829395, 0.02225780344842221)
246         0.1849251993905554 (0.1779138024370006, 0.1919365963441102)
Kanel       0.005615564431058928 (0.004491138016824436, 0.00673999084529342)
175         0.3166100525107627 (0.30659025767436177, 0.3266298473471636)
Kaolack     0.008077416250422687 (0.006987441620362134, 0.00916739088048324)
260         0.7693792703784261 (0.7167354411006137, 0.8220230996562385)
Kebemer     0.010021691126443244 (0.008382621588113537, 0.011660760664772951)
244         0.21169123555459635 (0.2036658292848088, 0.21971664182438388)
Kedougou    0.004622782266665126 (0.003534646565491073, 0.00571091796783918)
117         0.18660775557903442 (0.1818373281647503, 0.19137818299331855)

Keur Massar 0.014243077724529485 (0.011471001293502425, 0.017015154155556544)
331         0.9200794423858429 (0.8648056849066883, 0.9753531998649976)
Kolda       0.002037553399144798 (0.0013381624829147398, 0.0027369443153748563)
112         0.21282807614512578 (0.20657330231725624, 0.21908284997299532)
Koumpentoum 0.001837844117927645 (0.0013401330073055107, 0.0023355552285497794)
155         0.16130278944481447 (0.15421666366154518, 0.16838891522808375)
Koungheul   0.005958355996412479 (0.0032371586407347053, 0.008679553352090252)
220         0.8893761338488027 (0.8415404583036743, 0.937211809393931)
Linguere    0.023307561121377773 (0.020134139803062112, 0.026480982439693435)
220         0.4654293821834443 (0.4388915848036472, 0.49196717956324143)
Louga       0.04462919926853459 (0.0434930674904237, 0.045765331046645485)
256         0.5828677042452295 (0.5479830671754176, 0.6177523413150414)
Malem Hoddar 0.036963849376093094 (0.03432851841861506, 0.03959918033357113)
223         0.47925973508820086 (0.4575227352029838, 0.5009967349734179)
Matam       0.0017134181987543684 (0.0012377717127680654, 0.0021890646847406714)
186         0.30697219152920674 (0.29423108868685155, 0.31971329437156193)
Mbacke      0.03962200876619093 (0.03740724300849685, 0.04183677452388501)
262         0.539087483444483 (0.5076969390261539, 0.5704780278628121)
Mbour       0.004014208828177601 (0.0032230176553138534, 0.004805400001041349)
266         0.2988236526257886 (0.2859534915793649, 0.31169381367221227)

Medina Yoro Foulah  0.031313789620442734 (0.029094008390853077, 0.03353357085003239)
70          0.3478566858681411 (0.3428632527540856, 0.35285011898219665)
Nioro du Rip    0.020612520831246428 (0.01698734956440795, 0.024237692098084906)
237         0.7957091962059106 (0.7609762725154461, 0.8304421198963752)
Oussouye    0.014096679242264543 (0.011656996567028344, 0.01653636191750074)
139         0.3441531228927115 (0.32833374427729645, 0.35997250150812654)
Pikine      0.04275233768036024 (0.04106823091015954, 0.04443644445056094)
336         0.6039066333258916 (0.5763165260770453, 0.6314967405747378)
Podor       0.009014961850546399 (0.006746567570228734, 0.011283356130864064)
164         0.36804361762671434 (0.36054774036625226, 0.3755394948871764)
Ranerou Ferlo  0.043110473939085736 (0.04176223564747694, 0.04445871223069453)
156         0.4522621316194151 (0.4394551485590643, 0.46506911467976586)
Rufisque    0.0014522699815096018 (0.001235590164801792, 0.0016689497982174117)
331         0.16007125628290986 (0.15167964313722138, 0.16846286942859834)
Saint-Louis 0.0020741220800317706 (0.001562028404910265, 0.002586215755153276)
236         0.357343950511666 (0.3264681997601997, 0.3882197012631323)
Salemata    0.012867779656367873 (0.010333547338657212, 0.015402011974078533)
88          0.3081576517058906 (0.3010332020937021, 0.3152821013180791)
Saraya      0.03405235123314121 (0.03179264426409212, 0.0363120582021903)
96          0.3619272721051523 (0.35467181345444665, 0.36918273075585795)

Sedhiou     0.013699743194225178 (0.010308593634334784, 0.017090892754115572)
180         0.446511053424965 (0.43037365667027494, 0.46264845017965506)
Tambacounda 0.002280675983080016 (0.0016242354636233358, 0.0029371165025366963)
184         0.258103204929494 (0.24579102463791713, 0.2704153852210709)
Thies       0.0017215048828003177 (0.0012758859177921522, 0.002167123847808483)
286         0.1784259311634564 (0.17022302576640413, 0.18662883656050866)
Tivaoune    0.008694595616617562 (0.007417164487003802, 0.009972026746231322)
273         0.2801971121810034 (0.2692235029128298, 0.291170721449177)
Velingara   0.009936752611013233 (0.00939940371292991, 0.010474101509096556)
69          0.2105445784493476 (0.20662048135186772, 0.21446867554682747)
Ziguinchor  0.0357728378865243 (0.034026750551074514, 0.037518925221974087)
155         0.4271828935412305 (0.41682005238472186, 0.43754573469773916)
'''

### GENERATE PATHS FOR CASE STUDY ###
# What is the upper bound on the number of regions in any feasible tour that uses at least one test?
maxregnum = GetSubtourMaxCardinality(optparamdict=optparamdict)

listinds1 = list(itertools.combinations(np.arange(1,numReg).tolist(),1))
listinds2 = list(itertools.combinations(np.arange(1,numReg).tolist(),2))
listinds3 = list(itertools.combinations(np.arange(1,numReg).tolist(),3))
listinds4 = list(itertools.combinations(np.arange(1,numReg).tolist(),4))
listinds5 = list(itertools.combinations(np.arange(1,numReg).tolist(),5))

mastlist = listinds1 + listinds2 + listinds3 + listinds4 + listinds5
len(mastlist)

# For storing best sequences and their corresponding costs
seqlist, seqcostlist = [], []

for tup in mastlist:
    tuplist = [tup[i] for i in range(len(tup))]
    tuplist.insert(0,0) # Add HQind to front of list
    bestseqlist, bestseqcost = FindTSPPathForGivenNodes(tuplist, f_reg)
    seqlist.append(bestseqlist)
    seqcostlist.append(bestseqcost)

# For each path, generate a binary vector indicating if each district is accessible on that path
# First get names of accessible districts
distaccesslist = []
for seq in seqlist:
    currdistlist = []
    for ind in seq:
        currdist = GetDeptChildren(regNames[ind],dept_df)
        currdistlist = currdistlist+currdist
    currdistlist.sort()
    distaccesslist.append(currdistlist)

# Next translate each list of district names to binary vectors
bindistaccessvectors = []
for distlist in distaccesslist:
    distbinvec = [int(i in distlist) for i in deptNames]
    bindistaccessvectors.append(distbinvec)

paths_df_all = pd.DataFrame({'Sequence':seqlist,'Cost':seqcostlist,'DistAccessBinaryVec':bindistaccessvectors})

# Remove all paths with cost exceeding budget - min{district access} - sampletest
paths_df = paths_df_all[paths_df_all['Cost'] < B].copy()
# Remaining paths require at least one district and one test in each visited region
boolevec = [True for i in range(paths_df.shape[0])]
for i in range(paths_df.shape[0]):
    rowseq, rowcost = paths_df.iloc[i]['Sequence'], paths_df.iloc[i]['Cost']
    mindistcost = 0
    for reg in rowseq:
        if reg != 0:
            mindistcost += f_dept[[deptNames.index(x) for x in GetDeptChildren(regNames[reg], dept_df)]].min()
    # Add district costs, testing costs, and path cost
    mincost = mindistcost + (len(rowseq)-1)*ctest + rowcost
    if mincost > B:
        boolevec[i] = False

paths_df = paths_df[boolevec]

# Update cost list and district access vectors to reflect these dropped paths
seqlist_trim = paths_df['Sequence'].copy()
seqcostlist_trim = paths_df['Cost'].copy()
bindistaccessvectors_trim = np.array(paths_df['DistAccessBinaryVec'].tolist())
seqlist_trim = seqlist_trim.reset_index()
seqlist_trim = seqlist_trim.drop(columns='index')
seqcostlist_trim = seqcostlist_trim.reset_index()
seqcostlist_trim = seqcostlist_trim.drop(columns='index')



# Save to avoid generating later
# paths_df.to_pickle(os.path.join('operationalizedsamplingplans', 'numpy_objects', 'paths.pkl'))

# paths_df = pd.read_pickle(os.path.join('operationalizedsamplingplans', 'numpy_objects', 'paths.pkl'))
###################################
###################################
###################################
# MAIN OPTIMIZATION BLOCK
###################################
###################################
###################################

# First need to obtain vectors of zero intercepts, junctures, and interpolation slopes for each of our Utilde evals
#   at each district
lvec, juncvec, m1vec, m2vec, = [], [], [], []
for ind, row in util_df.iterrows():
    currBound, loval, hival = row[1], row[2], row[4]
    # Get interpolation values
    _, _, l, k, m1, m2 = GetTriangleInterpolation([0, 1, currBound], [0, loval, hival])
    lvec.append(l)
    juncvec.append(k)
    m1vec.append(m1)
    m2vec.append(m2)
    
# What is the curvature, kappa, for our estimates?
kappavec = [1-m2vec[i]/m1vec[i] for i in range(len(m2vec))]
plt.hist(kappavec)
plt.title('Histogram of $\kappa$ curvature at each district')
plt.show()

# Make histograms of our interpolated values
plt.hist(lvec,color='orange')
plt.title('Histogram of zero intercepts ($l$ values)')
plt.xlabel('$l$')
plt.ylabel('Count')
plt.show()

plt.hist(juncvec,color='darkgreen')
plt.title('Histogram of slope junctures ($j$ values)')
plt.xlabel('$j$')
plt.ylabel('Count')
plt.show()

plt.hist(m1vec,color='crimson')
plt.title('Histogram of first slopes ($m_1$ values)')
plt.xlabel('$m_1$')
plt.ylabel('Count')
plt.xlim([0,0.025])
plt.show()

plt.hist(m2vec,color='pink')
plt.title('Histogram of second slopes ($m_2$ values)')
plt.xlabel('$m_2$')
plt.ylabel('Count')
plt.xlim([0,0.025])
plt.show()

# Now we construct our various program vectors and matrices per the scipy standards
numPath = paths_df.shape[0]

# Variable bounds
# Variable vectors are in form (z, n, x) [districts, allocations, paths]
lbounds = np.concatenate((np.zeros(numTN*3), np.zeros(numPath)))
ubounds = np.concatenate((np.ones(numTN),
                          np.array([juncvec[i]-1 for i in range(numTN)]),
                          np.array(util_df['Bounds'].tolist()) - np.array([juncvec[i] - 1 for i in range(numTN)]),
                          np.ones(numPath)))

optbounds = spo.Bounds(lbounds, ubounds)

# Objective vector; negated as milp requires minimization
optobjvec = -np.concatenate((np.array(lvec), np.array(m1vec), np.array(m2vec), np.zeros(numPath)))

### Constraints
# Build lower and upper inequality values
optconstrlower = np.concatenate(( np.ones(numTN*4+1) * -np.inf, np.array([1])))
optconstrupper = np.concatenate((np.array([B]), np.zeros(numTN*2), np.array(juncvec), np.zeros(numTN), np.array([1])))

# Build A matrix, from left to right
# Build z district binaries first
optconstraintmat1 = np.vstack((f_dept, -bigM*np.identity(numTN), np.identity(numTN), 0*np.identity(numTN),
                              np.identity(numTN), np.zeros(numTN)))
# n^' matrices
optconstraintmat2 = np.vstack((ctest*np.ones(numTN), np.identity(numTN), -np.identity(numTN), np.identity(numTN),
                              0*np.identity(numTN), np.zeros(numTN)))
# n^'' matrices
optconstraintmat3 = np.vstack((ctest*np.ones(numTN), np.identity(numTN), -np.identity(numTN), 0*np.identity(numTN),
                              0*np.identity(numTN), np.zeros(numTN)))
# path matrices
optconstraintmat4 = np.vstack((np.array(seqcostlist_trim).T, np.zeros((numTN*3, numPath)),
                               (-bindistaccessvectors_trim).T, np.ones(numPath)))

optconstraintmat = np.hstack((optconstraintmat1, optconstraintmat2, optconstraintmat3, optconstraintmat4))

optconstraints = spo.LinearConstraint(optconstraintmat, optconstrlower, optconstrupper)

# Define integrality for all variables
optintegrality = np.ones_like(optobjvec)

# Solve
spoOutput = milp(c=optobjvec, constraints=optconstraints, integrality=optintegrality, bounds=optbounds)
soln = spoOutput.x


# Make function for turning scipy output into our case study
def scipytoallocation(spo_x, eliminateZeros=False):
    z = np.round(spo_x[:numTN])
    n1 = np.round(spo_x[numTN:numTN * 2])
    n2 = np.round(spo_x[numTN * 2:numTN * 3])
    x = np.round(spo_x[numTN * 3:]) # Solver sometimes gives non-integer solutions
    path = seqlist_trim.iloc[np.where(x == 1)[0][0],0]
    # Print district name with
    for distind, distname in enumerate(deptNames):
        if not eliminateZeros:
            print(str(distname)+':', str(int(z[distind])), str(int(n1[distind])), str(int(n2[distind])))
        else: # Remove zeros
            if int(z[distind])==1:
                print(str(distname) + ':', str(int(z[distind])), str(int(n1[distind])), str(int(n2[distind])))
    pathstr = ''
    for regind in path:
        pathstr = pathstr + str(regNames[regind]) + ' '
    print('Path: '+ pathstr)
    return

scipytoallocation(spoOutput.x, eliminateZeros=True)

### Inspect our solution
# How does our utility value compare with the real utility?
n1 = soln[numTN:numTN * 2]
n2 = soln[numTN * 2:numTN * 3]
n_init = n1+n2


time0 = time.time()
u_init, u_init_CI = getUtilityEstimate(n_init, lgdict, paramdict)
time1 = time.time() - time0
print(time1)
''' 4-JAN
spoOutput*-1: 2.7690379399483853
150k/500 draws:
u_init, u_init_CI:  2.184450706985116, (2.0683527673839848, 2.300548646586247)
                    2.189141541015074, (2.063870236586828, 2.3144128454433197)
                    2.1243090313794664, (2.0040581266308806, 2.244559936128052)
Bound is about 27% above actual value
100k/500 draws:     (2.217041463121607, (2.098915436652165, 2.335167489591049))
200k/xxx draws:     ???????????????
losslist:           
'''
# This objective is our overall upper bound for the problem
UB = spoOutput.fun*-1

def getUtilityEstimateSequential(n, lgdict, paramdict, zlevel=0.95, datadrawsiter=50, eps=0.2, maxdatadraws=2000):
    """
    Return a utility estimate average and confidence interval for allocation array n that is epsperc of the estimate,
    by running data draws until the confidence interval is sufficiently small
    """
    testnum = int(np.sum(n))
    des = n/testnum

    # Modify paramdict to only have datadrawsiter data draws
    masterlosslist = []
    epsgap = 1.
    itercount = 0
    while len(masterlosslist) < maxdatadraws and epsgap > eps:
        itercount += 1
        print('Total number of data draws: ' + str(itercount*datadrawsiter))
        paramdictcopy = paramdict.copy()

        paramdictcopy.update({'datadraws':truthdraws[choice(np.arange(paramdict['truthdraws'].shape[0] ),
                                                            size=datadrawsiter, replace=False)]})
        util.print_param_checks(paramdictcopy)
        masterlosslist = masterlosslist + sampf.sampling_plan_loss_list(des, testnum, lgdict, paramdictcopy)
        currloss_avg, currloss_CI = sampf.process_loss_list(masterlosslist, zlevel=zlevel)
        # Get current gap
        epsgap = (currloss_CI[1]-currloss_CI[0])/(paramdict['baseloss'] -currloss_avg)
        print('New utility: ' + str(paramdict['baseloss'] - currloss_avg))
        print('New utility range: ' + str(epsgap))

    return paramdict['baseloss'] - currloss_avg, \
           (paramdict['baseloss']-currloss_CI[1], paramdict['baseloss']-currloss_CI[0]), masterlosslist


time0 = time.time()
u_init, u_init_CI, _ = getUtilityEstimateSequential(n_init, lgdict, paramdict)
runtime = time.time()-time0
print(runtime)


################################
################################
# COMPARATIVE CASE: GENERATE RANDOM PATHS AND ALLOCATIONS AND COMPARE WITH THE UTILITY OF OUR INITIAL FEASIBLE SOLUTION
################################
################################
comparepathsdict = {}
# Choose 2|D|+1 feasible paths
np.random.seed(55893)
numcomparepaths = 2*len(deptNames)+277
compare_pathinds = np.random.choice(np.arange(numPath),size=numcomparepaths,replace=False)
compare_pathinds.sort()
comparepathsdict.update({'pathinds':compare_pathinds})
# Iterate through each path and designate visited districts
compare_visiteddistinds = []
compare_allocvecs = []
pathstoadd = 0 # For ensuring we end up with 93 feasible paths
for pathind in comparepathsdict['pathinds'].tolist():
    curr_distaccess = [0 for x in range(numTN)]
    curr_regs = paths_df.iloc[pathind]['Sequence']
    for r in curr_regs:
        if r == 0: # Flip coin for HQ region
            possDists = GetDeptChildren(regNames[r], dept_df)
            possDistsInds = [deptNames.index(x) for x in possDists]
            for distInd in possDistsInds:
                curr_distaccess[distInd] = np.random.binomial(n=1,p=0.25)
        else:
            # Guarantee one district is visited
            possDists = GetDeptChildren(regNames[r], dept_df)
            defVisitDist = possDists[np.random.choice(np.arange(len(possDists)))]
            curr_distaccess[deptNames.index(defVisitDist)] = 1
            possDists.remove(defVisitDist)
            possDistsInds = [deptNames.index(x) for x in possDists]
            for distInd in possDistsInds:
                curr_distaccess[distInd] = np.random.binomial(1, 0.25)
    compare_visiteddistinds.append(curr_distaccess)
    # Add one test to each visited district
    curr_n = np.array(curr_distaccess)
    # Check if budget is feasible
    budgetcost = np.sum(np.array(curr_distaccess) * f_dept) + paths_df.iloc[pathind]['Cost'] + curr_n.sum()*ctest
    if budgetcost > B:
        pathstoadd += 1
    else: # Expend rest of budget on tests at random locations
        teststoadd = int(np.floor((B-budgetcost)/ctest))
        multinom_num = curr_n.sum()
        multinom_vec = np.random.multinomial(n=teststoadd,pvals=np.ones(multinom_num)/multinom_num)
        curraddind = 0
        for t_ind in range(curr_n.shape[0]):
            if curr_n[t_ind] > 0:
                curr_n[t_ind] += multinom_vec[curraddind]
                curraddind += 1
    compare_allocvecs.append(curr_n)
comparepathsdict.update({'visiteddistinds':compare_visiteddistinds})
comparepathsdict.update({'allocvecs':compare_allocvecs})
print(numcomparepaths-pathstoadd) # Target is 93

# Save comparative paths dictionary
'''
comparepathsdict.update({'lossevals':[[] for x in range(numcomparepaths)]})
with open(os.path.join('operationalizedsamplingplans', 'numpy_objects', 'comparepaths.pkl'), 'wb') as fp:
    pickle.dump(comparepathsdict, fp)
'''

#########
# Load previous runs and append to those
with open(os.path.join('operationalizedsamplingplans', 'numpy_objects', 'comparepaths.pkl'), 'rb') as fp:
    comparepathsdict = pickle.load(fp)

# Now loop through feasible budgets and get loss evaluations
start_i = 201 # Denote which comparison path to begin with
for temp_i, pathind in enumerate(comparepathsdict['pathinds'].tolist()):
    budgetcost = (np.array(np.array(comparepathsdict['visiteddistinds']).tolist()[temp_i])*f_dept).sum() +\
                 paths_df['Cost'].tolist()[pathind] +\
                 np.array(np.array(comparepathsdict['allocvecs']).tolist()[temp_i]).sum()*ctest
    if budgetcost <= B and temp_i >= start_i: # Get utility
        print('Getting utility for comparative path '+str(temp_i)+'...')
        curr_n = comparepathsdict['allocvecs'][temp_i]
        currlosslist = sampf.sampling_plan_loss_list(curr_n/curr_n.sum(), curr_n.sum(), lgdict, paramdict)
        comparepathsdict['lossevals'][temp_i] = comparepathsdict['lossevals'][temp_i] + currlosslist

with open(os.path.join('operationalizedsamplingplans', 'numpy_objects', 'comparepaths.pkl'), 'wb') as fp:
    pickle.dump(comparepathsdict, fp)

# Plot a histogram
utilhistvals = []
for i in range(len(comparepathsdict['pathinds'].tolist())):
    utilhistvals.append(paramdict['baseloss'] - np.average(comparepathsdict['lossevals'][i]))
fig, ax = plt.subplots()
plt.hist(utilhistvals, color='darkgreen')
plt.title('Histogram of utilities for random paths and allocations')
plt.xlabel('Utility')
plt.ylabel('Count')
plt.axvline(x=2.7690379399483853, color='magenta') # Approximation
plt.axvline(x=2.184450706985116, color='black') # Evaluated utility
ax.text(1.8,20,'$U(n^{(0)})$',fontsize=12)
ax.text(2.26,20,'$\sum_{d}\hat{U}_{d}(n^{(0)})$',fontsize=12,color='magenta')
plt.show()


###################
# HEURISTIC TO AVOID CORRELATED DISTRICTS
###################
def GetConstraintsWithDistrictCut(numVar, distIndsList):
    """
    Returns constraint object for use with scipy optimize, where each district in distIndsList must be 0
    """
    newconstraintmat = np.zeros((len(distIndsList), numVar)) # size of new constraints matrix
    for rowInd, distInd in enumerate(distIndsList):
        newconstraintmat[rowInd, distInd] = 1.
    return spo.LinearConstraint(newconstraintmat, np.zeros(len(distIndsList)), np.zeros(len(distIndsList)))

cutconstraints = GetConstraintsWithDistrictCut(numPath+numTN*3,[3])
numVar = numPath + numTN*3
solTuple = np.round(spoOutput.x)
solUtil = 2.184450706985116


def AvoidCorrelatedDistricts(solTuple, solUtil, constrToAdd=None):
    """
    Improves a solution by reducing district correlation in the utility
    """
    # Initialize return values
    retTuple = solTuple.copy()
    retUtil = solUtil

    # Initialize list of cut districts
    distCutIndsList = []

    # Loop until the utility drops
    utilDropped = False
    while not utilDropped:
        # Identify most correlated pair of districts
        eligDistInds = [ind for ind, x in enumerate(retTuple[:numTN].tolist()) if x>0]
        eligDistQ = tempQ[eligDistInds]
        # Initialize low pair
        lowpair = (0, 1)
        lownorm = np.linalg.norm(eligDistQ[0]-eligDistQ[1])
        # Identify smallest sourcing vector norm
        for i in range(len(eligDistInds)):
            for j in range(len(eligDistInds)):
                if j > i:
                    currnorm = np.linalg.norm(eligDistQ[i]-eligDistQ[j])
                    if currnorm < lownorm:
                        lownorm = currnorm
                        lowpair = (i, j)
        # Identify district providing least to objective
        ind1, ind2 = eligDistInds[lowpair[0]], eligDistInds[lowpair[1]]
        print('Most correlated pair: ' + str((ind1, ind2)) + ' (' + deptNames[ind1] + ', ' + deptNames[ind2] + ')' )
        nprime1_1 = retTuple[numTN + ind1]
        nprime1_2 = retTuple[numTN * 2 + ind1]
        nprime2_1 = retTuple[numTN + ind2]
        nprime2_2 = retTuple[numTN * 2 + ind2]
        obj1 = lvec[ind1] + m1vec[ind1] * nprime1_1 + m2vec[ind1] * nprime1_2
        obj2 = lvec[ind2] + m1vec[ind2] * nprime2_1 + m2vec[ind2] * nprime2_2
        if obj2 < obj1: # Drop ind2
            print('Cut district: ' + str(ind2) + ' (' + str(deptNames[ind2]) + ')')
            distCutIndsList.append(ind2)
        else: # Drop ind1
            print('Cut district: ' + str(ind1) + ' (' + str(deptNames[ind1]) + ')')
            distCutIndsList.append(ind1)
        # Generate constraints for cut districts
        cutconstraints = GetConstraintsWithDistrictCut(numVar, distCutIndsList)
        if constrToAdd == None:
            curr_spoOutput = milp(c=optobjvec,
                              constraints=(optconstraints, cutconstraints),
                              integrality=optintegrality, bounds=optbounds)
        else:
            curr_spoOutput = milp(c=optobjvec,
                                  constraints=(optconstraints, cutconstraints, constrToAdd),
                                  integrality=optintegrality, bounds=optbounds)
        scipytoallocation(curr_spoOutput.x, eliminateZeros=True)
        curr_n1 = curr_spoOutput.x[numTN:numTN * 2]
        curr_n2 = curr_spoOutput.x[numTN * 2:numTN * 3]
        curr_n = curr_n1 + curr_n2

        # Get utility oracle estimate
        curr_u, curr_u_CI, curr_losslist = getUtilityEstimateSequential(curr_n, lgdict, paramdict, maxdatadraws=1000,
                                                                        eps=0.1)
        print('New utility: ' + str(curr_u))
        print('Utility CI: ' + str(curr_u_CI))
        print('Loss list:')
        print(curr_losslist)
        # Exit if utility dropped
        if curr_u < retUtil:
            utilDropped = True
            print('Done trying to improve; new utility is: ' + str(retUtil))
        else:
            retTuple = curr_spoOutput.x.copy()
            retUtil = curr_u
    # END WHILE LOOP

    return retTuple, retUtil, distCutIndsList


# bestSol, LB, distCutIndsList = AvoidCorrelatedDistricts(solTuple, solUtil)
bestSol, LB = solTuple.copy(), solUtil
'''
AFTER 1 RUN OF AVOIDCORR ON INITIAL FEASIBLE SOLUTION
curr_u, curr_u_CI = 1.8521507685488476, (1.7256166704778373, 1.978684866619858)
distCutIndsList = [9]
'''
####################
# PART 2: TRY SOME RANDOM PATHS AND CHECK THEIR UTILITY
####################
UB = spoOutput.fun*-1

# Solve RP while setting each path to 1
def GetConstraintsWithPathCut(numVar, pathInd):
    """
    Returns constraint object for use with scipy optimize, where each district in distIndsList must be 0
    """
    newconstraintmat = np.zeros((1, numVar)) # size of new constraints matrix
    newconstraintmat[0, numTN*3 + pathInd] = 1.
    return spo.LinearConstraint(newconstraintmat, np.ones(1), np.ones(1))

# Prep a new paths dataframe

# Or load
# phase2paths_df = pd.read_pickle(os.path.join('operationalizedsamplingplans', 'numpy_objects', 'phase2paths.pkl'))

phase2paths_df = paths_df.copy()
phase2paths_df.insert(3, 'RPobj', np.zeros(numPath).tolist(), True)
phase2paths_df.insert(4, 'DistCost', np.zeros(numPath).tolist(), True) # Add column to store RP district costs
phase2paths_df.insert(5, 'Uoracle', np.zeros(numPath).tolist(), True) # Add column for oracle evals
phase2paths_df.insert(6, 'UoracleCIlo', [0 for i in range(numPath)], True) # Add column for oracle eval CIs
phase2paths_df.insert(7, 'UoracleCIhi', [0 for i in range(numPath)], True) # Add column for oracle eval CIs


# List of eligible path indices
eligPathInds = []

# RP for each path
for pathind in range(numPath):
    pathconstraint = GetConstraintsWithPathCut(numPath+numTN*3, pathind)
    curr_spoOutput = milp(c=optobjvec, constraints=(optconstraints, pathconstraint),
                          integrality=optintegrality, bounds=optbounds)
    phase2paths_df.iloc[pathind, 3] = curr_spoOutput.fun*-1
    phase2paths_df.iloc[pathind, 4] = (curr_spoOutput.x[:numTN] * f_dept).sum()
    if curr_spoOutput.fun*-1 > LB:
        eligPathInds.append(pathind)
        #scipytoallocation(np.round(curr_spoOutput.x), True)
        #print('Path cost: ' + str(phase2paths_df.iloc[pathind, 1]))
        #print('Path RP utility: ' + str(phase2paths_df.iloc[pathind, 3]))

# Save to avoid generating later
phase2paths_df.to_pickle(os.path.join('operationalizedsamplingplans', 'numpy_objects', 'phase2paths.pkl'))
'''
len(eligPathInds) = 30
eligPathInds = [1, 13, 14, 15, 18, 25, 26, 29, 31, 32, 33, 34, 35, 91, 92, 95, 97, 100, 157, 160, 165, 169, 174, 195, 376, 384, 388, 393, 409, 412, 
'''

# Sort 30 remaining eligible paths by transit/collection trade-off distance from initial solution tradeoff
initPathInd = np.where(soln[numTN * 3:] == 1)[0][0]
eligPathInds.remove(initPathInd) # Don't need to reevaluate this

initPathCost = phase2paths_df.iloc[initPathInd, 1] + phase2paths_df.iloc[initPathInd, 4]
initSol_transittestPerc = initPathCost / B

# Sort 29 eligible paths by distance from initial transit-testing ratio
eligPathRatioDists, eligPathRatioDists_forHist = [], []
for currpathind in eligPathInds:
    # Get ratio
    currPathCost = phase2paths_df.iloc[currpathind, 1] + phase2paths_df.iloc[currpathind, 4]
    currSol_transittestPerc = currPathCost / B
    eligPathRatioDists_forHist.append(initSol_transittestPerc-currSol_transittestPerc)
    eligPathRatioDists.append(np.abs(initSol_transittestPerc-currSol_transittestPerc))
# Histogram of differences in transit/testing ratio
plt.hist(eligPathRatioDists_forHist, color='darkblue')
plt.title('Histogram of difference of transit-testing ratios for Phase II paths\nSubtracted from initial solution ratio (lower=more transit)')
plt.xlabel('Difference')
plt.ylabel('Count')
plt.show()

# arg sort
eligPathInds_sort = [eligPathInds[x] for x in np.argsort(eligPathRatioDists).tolist()]

#####
# Loop through each eligible path according to the list sorted by testing ratio
# Get utility from oracle; then run AVOIDCORR to attempt to improve
#####

# Load from pickle if needed
# phase2paths_df = pd.read_pickle(os.path.join('operationalizedsamplingplans', 'numpy_objects', 'phase2paths.pkl'))

for currpathind in eligPathInds_sort[3:]:
    print('On path index: '+ str(currpathind)+'...')
    pathconstraint = GetConstraintsWithPathCut(numPath + numTN * 3, currpathind)
    currpath_spoOutput = milp(c=optobjvec, constraints=(optconstraints, pathconstraint),
                          integrality=optintegrality, bounds=optbounds)
    curr_n = currpath_spoOutput.x[numTN:numTN*2] + currpath_spoOutput.x[numTN*2:numTN*3]
    curr_u, curr_u_CI, curr_losslist = getUtilityEstimateSequential(curr_n, lgdict, paramdict, maxdatadraws=1000,
                                                                    eps=0.1)
    # Run AVOIDCORR to see if we can get a better fit
    print('Current utility: ' + str(curr_u))
    print('Current loss list:')
    print(curr_losslist)
    print('Seeing if we can improve the solution for path index: ' + str(currpathind) + '...')
    curr_impSol, curr_impUtil, curr_cutDists = AvoidCorrelatedDistricts(currpath_spoOutput.x, curr_u,
                                                                        constrToAdd=pathconstraint)
    if curr_impUtil > curr_u:
        print('Improved path index ' + str(currpathind) + 'by cutting districts ' + str(curr_cutDists[:-1]))

    # Save to data frame
    phase2paths_df.iloc[currpathind, 5] = curr_u
    phase2paths_df.iloc[currpathind, 6] = curr_u_CI[0]
    phase2paths_df.iloc[currpathind, 7] = curr_u_CI[1]

    # Save to avoid generating later
    phase2paths_df.to_pickle(os.path.join('operationalizedsamplingplans', 'numpy_objects', 'phase2paths.pkl'))

###########
# For plotting AVOIDCORR results
avoidcorrlist_noGain = [] # Each item is a list of [ind, util, utilCIlo, utilCIhi]
# 'ind' corresponds to order in eligPathInds_sort, AFTER our initial feasible solution
avoidcorrlist_noGain.append([2, 1.1365323217042338, 1.0810216850767471, 1.1920429583317205])
avoidcorrlist_noGain.append([1, 1.8521507685488476, 1.7256166704778373, 1.978684866619858])
avoidcorrlist_noGain.append([3, 1.2764127360273676, 1.216187066780611, 1.3366384052741243])
avoidcorrlist_noGain.append([4, 1.0850895475972102, 1.0344833165187381, 1.1356957786756823])


avoidcorrlist_Gain = [] # When improvements occur
#todo: DUMMY VALUE
avoidcorrlist_Gain.append([5, 2.1365323217042338, 2.0810216850767471, 2.1920429583317205])

'''
path 0
init_u, CI: 1.212514932297184, (1.1539925129541722, 1.2710373516401958)
losslist: [10.681744857240075, 10.21411113157396, 10.487961677590919, 10.361215232801364, 10.899210720569435, 10.844743534397294, 11.134234829940576, 10.772396440485092, 10.529399756661414, 10.66887299092486, 10.701501637110177, 10.620471454030882, 10.735231569195385, 10.582197878249533, 10.640929296888146, 10.69126825677781, 10.547009887726649, 10.688236200907504, 10.679932260924083, 10.683219540133848, 10.84055474641147, 10.85729999307392, 10.954838507869658, 10.885191823733868, 10.362159433629909, 10.08311185172482, 10.629668208710015, 10.712629585479608, 10.893457742498908, 10.707357105796087, 10.821566803009269, 10.877370300906192, 10.802840669305937, 10.698602985657402, 10.529160458845542, 10.467933900723061, 10.888670201597993, 10.473681563972107, 11.023817399117048, 10.74242634372481, 11.073911763310745, 9.14840518031635, 10.84432441885106, 10.628041116188033, 10.525376117359409, 10.86912103565599, 11.003318683269379, 10.572411529166473, 10.607455153292388, 10.806760576922501, 10.812986940900876, 10.76760484941331, 10.797252630940575, 10.81466504044724, 10.620281001215687, 10.875760538687064, 10.827326013721427, 10.858746823681832, 10.868589281710861, 10.703406035273078, 10.866800309139014, 10.465803992861442, 10.689044474312512, 10.274567805075923, 10.501197712273939, 10.997457272671861, 10.735773263932808, 10.876392806700354, 10.230548005375388, 10.814606476957275, 10.877634993198596, 10.574657193725752, 9.834057101979077, 10.78340910797319, 10.771132070998213, 10.782562966901455, 10.777103531180654, 10.683098718035424, 9.559627509906074, 10.797076754241985, 10.813407608222146, 10.675560906307208, 10.06883267179899, 5.416742076826827, 10.450333477313217, 10.634199066608383, 10.86789492323875, 10.715837834234751, 10.329585584676055, 10.510547376586288, 10.81108480160887, 10.790965154883475, 10.889755577878905, 10.729766696660823, 10.458977331930589, 10.719085610847323, 8.002634811622737, 10.649517828871678, 10.634551795360535, 10.501729977017337, 10.54349869607913, 8.088589271828019, 10.752110265573211, 10.75096665150001, 10.758750614604757, 10.231619085667509, 10.991005542422254, 10.71610897815246, 10.593652705991833, 10.538827117678009, 10.754841767656412, 10.982824529200053, 11.201989983167184, 10.672438891943047, 11.023356189638672, 10.87177581614556, 10.80088409458191, 11.01610993888989, 10.638213466665306, 10.86338423014831, 10.797668009777572, 10.222994430466198, 10.602633102486196, 10.498002664319797, 11.045031951170412, 10.3980523023545, 10.61706170105595, 10.924938077009266, 10.71278123972833, 10.686809748558215, 10.371263221655672, 10.829707158127338, 10.915449543531224, 10.845656458636569, 10.56015269531351, 10.323937072298435, 10.623645737961581, 10.895063759422904, 10.26579525601776, 10.573589939707277, 10.822130326250225, 10.917699231152087, 10.87719644532816, 10.751818625643494, 10.558804344472163, 10.839932232065133, 10.750157712039377, 10.86986594664586, 10.594544643421807, 10.676295852146971, 10.963623229615955, 10.380497029769863, 10.558587540158058, 10.791770919777774, 10.726394532732654, 10.908688969499167, 10.787812425104311, 10.766887915344068, 10.448551517134309, 9.80535857435934, 10.721622611938166, 10.354451251142985, 10.731124738489497, 10.526704755463847, 10.182774576364398, 10.357016437385747, 10.629000176197207, 10.653892740130543, 10.487406317418822, 10.82370830093555, 10.813114742292635, 10.881564602207366, 10.534990664232936, 10.625217657159583, 10.810825853438102, 10.986236433807756, 10.174074549768132, 10.787368536083807, 10.671318981067818, 10.778609988402055, 9.229154429181989, 10.859135300584, 10.696450515946877, 10.30330009089967, 10.705056611178986, 10.323157590429691, 10.460032842084253, 10.73420244424768, 9.892703648846618, 10.496704654314621, 10.233944070332734, 10.450268897826707, 10.723254113334304, 10.652461238777253, 10.643575060456275, 10.877730373656636, 10.545105469066963, 10.836604458116284, 10.765995363436618, 10.565162393035406, 10.772489207626634, 10.735262874627852, 10.922463738913597, 10.718189812755753, 10.465214516721064, 10.573324373286383, 10.50611390596604, 10.644779733664368, 10.667811877693325, 10.942956178532908, 10.761482954402636, 10.733211262082724, 10.708899087354343, 10.677645773637728, 10.861364972521898, 10.940315158334332, 10.768101925325741, 10.66452390613911, 10.649705969191253, 10.577207290968406, 10.766361220841578, 10.58534199699945, 10.601763459576018, 10.548111541115434, 10.647933551583874, 10.846444348069086, 11.055529042692497, 11.062584976478538, 10.83647501662963, 10.698606825718704, 9.522941517126263, 10.970828697959675, 10.873150123731728, 10.554686138403103, 10.45802082281694, 10.639191973830098, 10.653123366158075, 10.938669218270686, 10.877196699664793, 10.72720200025526, 10.812555005880924, 10.793143483665753, 10.151450937908486, 10.812504902422594, 10.838567610175584, 10.909544686188513, 10.802203005497997, 10.762461707486105, 10.823421833714143, 10.879710154300886, 10.323621550399015, 10.62785986979706, 10.748075530705512, 10.959163048317178, 10.823589120390405, 10.814175549447349, 10.671360692112657, 10.168178283370032, 10.819711508907666, 10.990567802177257, 10.816624674157179, 10.754026366729786, 10.844302369607737, 10.889548834433848, 10.919160109085569, 10.588604184929595, 10.709804163218086, 10.493778085433098, 10.503978649801052, 10.771589394305975, 10.789813307779722, 10.32953818622898, 10.682731674554809, 10.917803701194808, 11.05932200393881, 10.773303584842841, 10.507321818204444, 10.905929414464646, 10.677095070399393, 10.910482352601406, 10.95880947787257, 10.525570184736257, 10.718768639739514, 10.684759661886138, 10.792824458499803, 10.520756910225415, 10.576273429669067, 10.64842425370564, 10.772504433496465, 10.803818034967763, 10.215420198835279, 10.748398372448474, 10.3723454878518, 10.629623010204966, 1.0916583077391562, 10.758099062571056, 10.919080010264691, 10.765543865499716, 10.904523980891415, 10.314185814823873, 10.844872233051001, 10.840631792861107, 9.622888894257867, 10.89419682166742, 10.635739073506205, 10.774902341247705, 10.731877713066567, 10.906655931023288, 10.763369682056002, 10.532624377046362, 10.2961213087843, 9.296558887872628, 10.82900411506746, 10.691965210297502, 10.790564942924403, 10.613880975644669, 10.318119624271086, 10.37925727502707, 10.338869868245316, 10.840601647887505, 10.784144768457292, 10.277027962856952, 10.866093064888018, 10.28879338968132, 10.41031642028981, 10.603745900634925, 10.53371526503839, 8.961462304999179, 10.726626788609554, 10.710159686287906, 10.669020964249114, 11.004658433914479, 10.654155133266292, 9.711190705564137, 10.654020892077043, 10.664159998499388, 10.77961652189708, 10.849455512913487, 10.57439012219213, 10.738158655149011, 10.269201086405918, 10.898573631160005, 9.945441006558417, 10.719310442768247, 10.608811158528033, 10.644331750388133, 10.688453858316741, 10.831011383793182, 10.602691271904943, 10.859521797574537, 10.599345102240733, 10.471070055074309, 9.726870424105568, 10.694572889769985, 10.883346589547216, 10.68862650489962, 10.85141973562212, 10.866492158999911, 10.571218878441238, 10.58590036754764, 10.849673082584136, 10.805421684574027, 10.297850131707662, 10.845580538434676, 10.844582786408056, 10.441145725599325, 10.466550874454228, 10.779672279168215, 10.617380744890447, 10.88610906920497, 10.870209506685928, 10.402582209120986, 10.716557690332516, 11.027505440698942, 10.876588754813701, 10.83876840372418, 10.752032884381611, 10.878998152182877, 10.908966519472454, 10.581619165057425, 10.319607810615635, 10.557111217706069, 10.60072463840731, 11.000226596508565, 9.593346298453813, 10.70884161138944, 10.747816288279123, 10.653306810058751, 10.868805080626862, 10.824822067625869, 10.909485843852936, 10.864147815261374, 10.925409388365553, 10.617651247591434, 11.155481762492249, 8.636089382676179, 10.837912350214843, 10.649935097100151, 10.558442695333262, 10.792982672347033, 10.967571662508192, 10.750179077789515, 10.472711294990763, 10.588698225871841, 10.954775454378455, 10.778621523904373, 10.271357371237984, 10.748997813073297, 10.784083667797393, 10.980455382815355, 10.729958215554529, 10.627560802603824, 10.504682197666295, 10.661606993017946, 10.280676928518574, 10.777732665155959, 10.795960596935023, 10.682675945398922, 10.74954643505878, 10.766753979799073, 10.713365098603356, 10.948125193389666, 10.946193081177434, 10.661773428954657, 9.373673146198529, 10.652566779275741, 10.86102414005541, 10.885477976305307, 8.672403888695692, 10.518019035275454, 11.016745913827812, 10.759496253782276, 10.862634245596988, 10.786467357659543, 10.681949945950999, 10.766191580160193, 10.799234459353116, 10.659056706277322, 10.812449657456224, 10.782735939710633, 10.447818871517962, 10.392920099434475, 10.60428884624283, 10.900654839736957, 10.16284382860183, 10.375903486549587, 9.933497921537729, 9.209833188366934, 10.87441011042951, 10.740239561577903]
AVOIDCORR
Most correlated pair: (28, 33) (Mbacke, Pikine)
Cut district: 28 (Mbacke)
u, CI: 1.1365323217042338, (1.0810216850767471, 1.1920429583317205)
losslist: [10.752445034879933, 10.835257448460734, 10.777430125329246, 10.725844200620871, 10.963953136271776, 10.704735484907163, 10.900031196135608, 10.773532299458774, 8.117971011722004, 10.73076910962207, 10.644297778244432, 11.002631688699667, 10.638978184248499, 10.765617281417715, 10.82231072897174, 10.63782555044295, 10.741457467113733, 11.411850540325167, 10.755227140787708, 10.977035720139853, 10.820247829282982, 10.472114587796012, 10.741290454884876, 10.845730240843595, 10.98454909372515, 10.910434066821287, 10.65398660969911, 10.62656388735804, 10.83114459298432, 11.05181048832887, 11.036774874779052, 10.947335774133293, 10.372451399967963, 10.375996734376288, 9.193757550141816, 11.026478170495222, 10.787536553180493, 11.008992147822774, 10.858013205802743, 10.4305521049892, 10.71414949275473, 10.366466925432348, 10.960615273405141, 10.983340366921908, 10.87667714103726, 10.618693910907771, 10.799678813822897, 10.68000422602526, 10.927281191178903, 10.710630624002263, 10.847283715439984, 11.006686396063605, 10.817197303299551, 10.778757804960449, 10.79897756091195, 10.893502356721196, 11.116310831938065, 9.963488258354214, 10.2702272085015, 10.967420949019772, 11.005605152986012, 10.183576552118877, 10.83841990871393, 11.138385056799237, 10.765417046158445, 11.03063193071821, 10.990150913438658, 10.711954280756382, 10.881207697426111, 10.950139785093798, 5.0785568323420005, 10.78110476399267, 11.152429916094743, 10.777574067627773, 10.701277351564512, 10.093493149004194, 8.40843117971384, 10.722722263730976, 10.604754207109748, 10.790373029750304, 10.983528403985247, 10.789328696329983, 10.898704286109997, 10.538942858149383, 10.796398576316108, 10.816554708018405, 10.697630421446629, 10.421011750557744, 11.103341517558327, 10.633250331518632, 10.890182419329257, 10.916554218213953, 10.431536035670392, 10.688929255069883, 10.952485639472336, 10.723981679861154, 10.816256796297692, 10.862567502555223, 10.778782763209206, 10.575438682955243, 10.803432965509744, 10.752843361401618, 9.662602999114544, 10.94080923211365, 10.865364515914903, 10.695252521119826, 10.902248651411673, 10.590066828183154, 10.803253493083691, 10.801755440131947, 10.701183808163918, 10.990116911019376, 11.113610518203602, 10.89718305633854, 9.812164227522148, 10.5733012953841, 10.807300579489086, 10.92549965471767, 10.968912137689914, 11.03051601185149, 10.986552342200689, 11.165884283403415, 10.744837561358217, 11.008186915772125, 10.108750065269051, 10.069446749275274, 10.580916055665151, 11.00239793172488, 10.661607173611339, 10.589173255784866, 10.808664347453393, 10.634430433734273, 10.770444318619372, 10.816328819913169, 10.96936097412077, 10.7700554648014, 10.74077164903993, 10.629726651355332, 10.725014697813771, 10.980703684693085, 10.750438567568134, 10.713305123836752, 10.498220001237677, 10.616658293644043, 10.977600584167934, 10.773200495972285, 10.855211693853297, 11.006284409634418, 11.07265419603349, 10.990015490253626, 10.4565381104539, 10.948271075905568, 10.729114605370338, 10.810952958732926, 11.148405128593584, 10.449664165318788, 10.877396489533066, 10.896527709501495, 10.730885794364756, 10.68127405753398, 10.394454904048661, 10.821413748748679, 11.042589004654026, 10.693347233348343, 10.893829163748288, 10.6941878479696, 10.911185254363787, 10.756541669442365, 11.029675656123018, 10.791770014103014, 10.812049122121689, 10.770416349693509, 10.817288913474039, 10.659254871467137, 10.934636859121147, 10.685222222808445, 10.844707382420296, 10.830990463284868, 10.855645739517954, 10.756815887701004, 10.83990605044456, 10.63709790505334, 10.696598093871067, 10.999474483901398, 10.746395906701427, 10.821883124443435, 9.209736841853719, 11.137650200737085, 10.584728967019766, 11.123163834253164, 10.79956939603559, 10.81442409015532, 11.02283614585439, 10.695129322314104, 10.545742320215778, 8.406916644777407, 10.889429410534971, 11.123700357630556, 10.770226678789442, 10.82729735979732, 10.916789420266097, 10.77702737330608, 10.973785148791347, 10.837814208529343, 10.727582564744758, 10.33490996518923, 10.933412647594494, 10.899636771527629, 11.180853108015983, 10.745072606973066, 10.83716034264716, 11.097918787643755, 10.776120581673727, 10.63977237351497, 10.99899630265443, 10.90458706490881, 10.864030037571398, 10.650691235167896, 10.959958874430361, 10.66689564917612, 10.819611391732057, 10.076181938852157, 10.498590894730805, 10.85369997944389, 10.578906647092857, 10.971651923321097, 11.016186271717721, 11.053882661245963, 10.762472735894518, 10.999884923442254, 10.706510889841518, 10.438430478393592, 10.741468333621786, 10.884511324934635, 10.993662652976951, 10.860234248470649, 10.868517532962123, 10.625786167322493, 10.416451357686759, 10.814883186975726, 10.781712394891526, 10.936508380968409, 11.107407104608884, 10.840221529492842, 10.874980600860145, 10.527741114647888, 10.875208949923502, 10.542068715565737, 11.109710774186627, 9.955580360363676, 10.923774373511238, 9.690241419580172, 10.782117106885053, 10.885084394360847, 10.712485823584066, 10.734639379708415, 10.704695983532224, 10.215309866138622, 11.081161940407384, 10.941046252423462, 10.39226032289298, 11.227591856924384, 10.811962921784664, 10.7781936263524, 10.871580928049475, 10.911303446839447, 9.655221883364323, 11.036993936823901, 10.10381152303221, 10.848697240418758, 10.928819929544348, 10.633846349896045, 10.722560717088319, 1.26344343725458, 10.525288866611607, 11.017886749288076, 10.845511542605395, 11.018998879122991, 11.068254847124292, 10.909143634835504, 10.885838501131827, 10.797644240864328, 10.832088333911242, 10.865059564011572, 10.665662332528605, 10.950718554477568, 10.693960251364002, 11.034745886547917, 10.914876550926607, 10.767850334321423, 10.192148521142688, 10.657263732596084, 10.876940128226007, 10.840846323643909, 10.913476500737488, 10.92423211792346, 10.969489955998306, 10.814159330552895, 10.898125607669785, 10.758911512041891, 10.934561643123219, 11.039790746538591, 10.540301804687044, 10.865636981040938, 10.404135707600446, 10.510102121113652, 10.95211686534141, 10.725384661298573, 10.467542839463997, 10.712941557753885, 11.005320346280632, 11.206014704347218, 10.902658731950458, 10.707935367740937, 10.588993701275683, 10.884297311161415, 10.95626128957081, 9.731772842918735, 10.660799184995263, 10.830637365102664, 10.834609280692204, 10.515880472168314, 10.879760967055907, 10.790289063501787, 10.728251714305026, 9.90052004131104, 10.70456061186316, 11.034568940625867, 11.00546015258699, 10.967801935040988, 11.030407199284198, 10.775193684337712, 10.473206962812789, 10.34830775759783, 10.857766065282975, 10.789436317982505, 10.922183322572375, 7.80332044436921, 10.869537230022082, 10.370260101731676, 10.708584305808428, 10.415535423094278, 11.169632974047232, 10.955203061237093, 4.518929990588789, 10.899992785966036, 10.896880631629053, 10.685817754332312, 10.753510296516863, 10.532106925984307, 10.900179768441912, 10.86839284677599, 10.508827898708118, 10.801507263253242, 10.03307567115132, 11.1712426906259, 11.126927552507713, 10.846648028739436, 10.784576431886453, 10.462189364000876, 10.822773470152688, 11.029146113501676, 10.747055327060433, 11.1040917982339, 11.180105116449106, 10.801722045072394, 10.9021123851759, 10.811687944425339, 10.840948085848286, 10.81006196473889, 10.510788552578045, 10.666032386949007, 10.995492400414632, 10.8189380633364, 10.801499562690744, 11.014934240056276, 11.00792499165026, 10.882787164409923, 10.636602789495518, 11.006949931053482, 10.966681870342871, 10.820517004837283, 10.881730129382538, 10.721080290737293, 10.769852051249524, 10.82522267324515, 9.836637686759964, 10.801711002656322, 10.885238444127607, 10.849460398140375, 10.693995674610049, 10.673295822093129, 10.816432482836841, 10.966562248203093, 10.935174500974538, 10.873177958258802, 10.838316992871672, 10.364865250679845, 10.782889604700426, 11.024460772839268, 10.69853025437182, 10.808218308950027, 10.672746148619465, 10.945769099284178, 10.916736898283267, 9.175981823910861, 10.408899268925174, 10.568703482597614, 10.642079477698415, 10.88693841443145, 10.94218292306778, 10.044608867682802, 10.990393813901727, 11.146608279804695, 10.971627707440934, 10.822397228354363, 10.93583839487874, 10.932259521568595, 10.358585649774097, 11.156326323259742, 10.77302928811361, 10.669059852131246, 10.530101788228537, 10.642270574605092, 10.639866961764085, 10.76033981485372, 10.781331285484232, 10.867978254567637, 10.91088668907421, 11.507272845892887, 10.452346377465245, 10.822885116261485, 10.811533785315868, 11.02835961255088, 10.85661499528384, 10.719645900428006, 10.561128981405469, 11.005919584834, 10.97100769711338, 10.882463464177013, 10.912160585961427, 10.6393910230084, 10.115421938057622, 10.529057452765873, 10.981254428830901, 10.723420006161069, 10.654138001532317, 10.958348637078867, 11.016776188643998, 10.74751850933765, 10.938802747364255, 10.02830239675553, 10.858806489128915, 10.57474667005049, 10.724765097891414, 9.85102686623751, 10.96440500264437, 10.705141841511779, 10.73567398135383, 10.803888981458991, 10.594937541073298, 10.970616832887956, 11.145799149210394, 10.642457087726195, 10.983978821050774, 10.788847849598119, 11.048180556560812, 10.209427859906365, 10.98692524045911, 10.702636111924663, 10.605170716284785, 10.54395287245114, 10.83994381943336, 11.082495455171916, 10.384922604770038, 10.829104146157146, 10.658666815617805, 10.973100273639588, 10.654625630122776, 10.777971539277592, 10.83155036581773, 10.725146580409236, 2.1763052930255773, 10.925276505874681, 10.889049695825019, 10.68701067849515, 10.98653288789768, 10.763877321388271, 10.78277069089826, 10.901095287544232, 11.098438422810316, 10.68619358914317, 10.699605507947625, 3.1515088885663514, 10.945926962369938, 10.936387111527822, 10.134681916804844, 10.568735681354331, 10.83770677352116, 10.610969687041877, 10.969524032703404, 10.853856056170795, 10.897726204108046, 11.186145140448877, 10.380689245179804, 10.064199014495724, 10.739374834612903, 10.56587815156541, 10.748243282473737, 10.814477536756568, 10.916529885692887, 10.815306710786412, 10.994519849893841, 10.68354012892506, 11.038078478901014, 10.826989290782072, 11.144480206469902, 8.706921921393606, 10.786198388960841, 10.930234999869175, 10.795710706186194, 11.186415721626748, 10.990903445057821, 10.889349716978822, 10.966358691719327, 10.92396929008553, 10.644555577032653, 10.864197489666942, 10.861474657437988, 10.9430470797389, 11.054402816974058, 10.86872134431484, 11.011460282604604, 10.753815352633593, 11.064326638700775, 10.788955327371477, 10.859730861506613, 10.820109282780432, 10.893441127237706, 10.593679774654909, 10.792528079495495, 10.753979955751086, 10.91456248262346, 10.644653129828322, 10.994224771973478, 10.982054055706067, 10.723700568760703, 10.321738386389582, 10.826770614147797, 10.992533460103644, 10.66034793963373, 10.626840597058147, 10.911691658010405, 10.73195120000621, 10.77508628971632, 10.684702401786833, 10.988267467949102, 10.634491459841723, 10.715793162587067, 10.85255063869418, 11.086113925611315, 10.678836524281888, 10.89195491968007, 10.94537828745827, 10.803608575053564, 10.343888651584443, 11.087723448953476, 10.878258302095634, 11.004722702755027, 10.620761933345111, 10.814412931184446, 10.933996061651701, 11.015736337743133, 10.793608820006295, 10.7077163927331, 10.67376425321501, 10.575627037167731, 10.821259294870178, 10.841865914735312, 10.786831996339762, 10.680936465059453, 10.491921740081889, 10.696321071115305, 10.314421739481533, 10.952726719997463, 11.022789867398684, 10.658994149712344, 10.744913986865793, 10.883510230508957, 10.720590126890484, 10.952065658692817, 10.810170226799073, 10.80489958540341, 10.697218149396095, 10.649756608997706, 10.912514612213549, 10.707087006569022, 11.015967382901756, 10.620460626152049, 10.937977622286335, 10.686228156918787, 10.751706785127332, 10.832217205345664, 10.435310119946394, 10.77133896150428, 10.876708875857968, 8.348571027909815, 10.862431312939448, 10.862510583334275, 10.946641071859485, 10.924430682800155, 11.01275725936122, 10.71039683528317, 10.986327605286416, 11.017365295076283, 10.796756181814676, 10.85293417103602, 10.676186124297866, 10.66895169416955, 10.800526299765501, 10.8119685886012, 10.811261687206851, 10.779847030294585, 11.028001720251691, 10.725714395963672, 11.08493253767573, 10.70266336331905, 10.969713254000172, 10.437368127747634, 11.144903903701362, 10.792890754291758, 10.806882739584497, 10.794332747454499, 10.833141964909235, 5.455950668259359, 10.894125362176183, 10.941577183957575, 10.922366064312149, 10.856383622647297, 10.785888823050222, 11.036649245493289, 10.916282542922952, 10.495920001941132, 10.694426235538156, 10.651781396303416, 10.832528043480835, 10.990575644281481, 9.942089336922031, 10.867626861453168, 11.147013922150444, 10.841309233485102, 10.693545343690891, 10.579043171256874, 10.822011141525525, 10.806041195126076, 10.841514691910243, 10.971924154102338, 10.692400092044684, 11.179022695959091, 10.662351112515902, 10.830106796195423, 10.696770296424651, 10.898237890323848, 9.498262521323722, 10.912171571816607, 10.551242670669666, 10.937216949853568, 10.914439588730772, 10.895549313865237, 10.748318894977785, 11.025622850087403, 11.096921739862168, 10.886150280584468, 10.706973434312394, 10.944125406753926, 10.479272696732759, 10.829335735650849, 10.509052307348151, 8.356943781316929, 10.810904414862732, 10.739216231875123, 11.006307249399596, 10.672094416858014, 10.06911761194887, 10.909228593694163, 10.80565562787574, 11.067997070658556, 10.807266355525863, 9.634535861544512, 10.776665369500638, 10.834529268072892, 10.83394271878505, 10.990914992537562, 11.16806561886337, 10.927506678869886, 10.817395206202411, 10.78783900223109, 10.704685758703917, 10.78850705742691, 8.047283341075484, 11.070556901289697, 8.812491941907155, 10.657556373031467, 10.882633419257106, 10.365284111070395, 11.12656064743213, 10.948647282504075, 10.931902793010503, 10.434266673438792, 10.878245818494534, 10.813423763642392, 10.608966646935585, 10.473207619015541, 9.907223690873936, 5.856589754442581, 10.754283780802826, 10.87685910810369, 11.148356182330463, 10.58310976344597, 10.885940997105022, 10.680111954605993, 11.11604339388412, 11.050757452353299, 10.793100911448413, 10.874643785018538, 10.876707021698925, 10.866026973810206, 10.79360643762932, 10.637829500364944, 10.460893019203777, 11.006067482677834, 10.601177808304561, 10.550394903450403, 10.758672613878577, 10.796966818001751, 10.526886847263569, 11.039117342642593, 10.8858402364727, 10.793457771290159, 10.810091048157709, 10.76411074924851, 10.150849868969674, 10.72490715053014, 9.222615512412974, 11.06768938172057, 10.957407541380391, 10.806959987230519, 10.674305108229264, 10.541888258530806, 10.414581985561501]

path 1
init_u, CI: 1.5988931545712113, (1.5226730983699461, 1.6751132107724764)
losslist: [10.491170232343984, 10.52049623554677, 10.614299200181863, 9.587172963183423, 10.59847246379077, 10.703858920683174, 10.267646442964587, 10.334307507857162, 10.213556111759846, 10.539070126404859, 10.687771005677732, 10.783608162978469, 10.289573787480903, 10.68385532018124, 10.530837355160907, 9.517413961684358, 10.611435126784, 10.59837423242315, 2.222673196745074, 10.413109248785103, 10.505647115431321, 10.058606278228474, 10.626358520237321, 10.366987777425422, 10.378503821086502, 10.57847278613923, 10.922619757858167, 10.370903733787474, 10.456547154920072, 9.820909309594859, 10.586331389349791, 10.66818435914896, 10.650502714998336, 10.727580407814006, 8.256370298380528, 9.899198214891417, 10.453953329387968, 10.15936644824418, 10.601127342774872, 10.58281032901425, 9.70722577873802, 10.252296084669451, 10.464891267509488, 10.643008227466053, 10.053248454026576, 9.807550733975729, 10.421783990691868, 10.531800748534767, 10.557979277676083, 10.057586483944895, 9.210272578635182, 10.706626300377001, 10.749987927434066, 5.313688943170628, 10.581674996731158, 10.4075226020731, 10.272795562410355, 10.457991069846013, 10.380958197333687, 10.104649984545615, 10.70369129421439, 10.364449901113488, 10.360684151155981, 10.510729392876673, 10.606997686197547, 10.06110999398455, 9.971683938394703, 10.670917851795515, 10.461288576506291, 10.366498017033647, 10.633043323007508, 10.726241666479403, 10.063023643672395, 10.14006776618874, 10.520795657163095, 10.55188432000338, 9.534378576627665, 10.352759980655797, 10.468599708509611, 10.136056060473543, 10.438767639046562, 10.39479431304509, 10.489685341009235, 10.783396272683394, 10.541922474330415, 10.682682367349756, 10.051551667752832, 10.598476134984962, 10.33697396624151, 9.18043902932502, 10.451786291975086, 10.616773344411692, 10.491436627077835, 10.555044418861806, 10.773045594047717, 10.724194751614602, 10.657806153229034, 10.599976743493327, 10.288028180902979, 10.180706103267754, 7.849442765683893, 9.486638234693707, 10.443594651591257, 10.603344281355852, 10.567618186679411, 10.73268520465391, 9.783124532209193, 10.557162166584506, 10.25967387620969, 10.632213479006188, 10.417115730057182, 9.915366994886318, 10.911513081128495, 10.583707769833108, 10.339550818316713, 10.648982554671784, 10.7628071138821, 10.145198043302084, 10.59781666168473, 10.502750594844601, 10.944741988787095, 10.455383688575871, 10.631537246792025, 10.449831085384659, 10.68458732367271, 10.930060105081695, 10.863821825167717, 10.57586130634305, 10.504338071483083, 10.635696722036645, 10.725630636995467, 9.150041355928717, 10.655936564132636, 10.863605946657179, 10.65778587163059, 10.402849500331646, 10.278440142675693, 10.524978863939632, 10.364913645591596, 10.29851847835767, 10.577271098942388, 10.14146475883753, 10.577757452535549, 10.580780322450819, 10.776425772085414, 9.858721875808346, 9.643173268886693, 9.8383076727273, 10.538635592224693, 10.14638067681281, 11.046393860892048, 10.525133480916718, 10.597845578819136, 8.87586496964588, 10.784785720612616, 9.158803048951187, 10.360173283121028, 10.929282559784626, 10.420889962612073, 10.605583989901112, 10.766723461360005, 10.206724684671594, 10.842535527699194, 10.68555265408451, 10.55131912176874, 10.032580894084546, 10.605644954412885, 10.20590820543371, 10.423653422034567, 10.326073920819445, 10.722389107510052, 10.191227430146268, 10.602387534347768, 10.360281520397358, 10.21855592775649, 10.819461712368367, 10.344836043766803, 10.596404922795815, 10.547557036687833, 10.415533611098981, 10.626019971456428, 10.571092068811353, 9.54623384526936, 10.49420271756442, 5.042745051206892, 10.194219211747122, 10.650210858836898, 9.868468659703439, 10.512445647217078, 10.5214822557146, 10.50548521080916, 10.126480472850929, 10.659741885208193, 10.561641397326994, 10.527223924221797, 10.37440544309829, 10.653902903994906, 10.603573457808693, 9.93495046278038, 10.193434237350097, 9.368182101500281, 10.546702376593014, 10.872435730649903, 10.599658556034795, 10.495709989263158, 10.324058310970587, 10.385865926748046, 10.032244874896053, 10.566062938119925, 9.978389981584208, 10.574757587294068, 10.737256562763605, 10.562953753336432, 10.124759436207208, 10.696952808587316, 9.944292531472428, 10.280457357051421, 10.68247710710123, 10.39638330840038, 10.591796170348868, 10.547259455304047, 10.641914549568888, 10.61085409062435, 10.54351442327459, 10.49451988920725, 10.33604907686636, 10.145081255224797, 10.74183163611493, 10.503896172133274, 10.56407433960438, 6.983742671999248, 10.592415700498748, 10.515505132940731, 10.299593306056208, 10.593521005671331, 10.88172941116636, 10.793265280008779, 10.135450869716472, 9.341102757318902, 10.364072099955616, 0.2720511343027302, 10.881225922737723, 10.881337727235538, 10.677776236001318, 10.544266445599874, 10.707483773549079, 10.512647020416095, 10.687459823793569, 9.669598025996635, 10.2352166918818, 10.258364651386755, 10.639684135094063, 7.671906827554593, 10.62982790886337, 10.618475602840304, 6.892359800107155, 10.678003401414275, 10.593186959564994, 10.417423020988698, 10.508552639844208, 10.36329496809645, 10.649000260723923, 10.443217919265877, 7.669480367959954, 10.288420897388157, 8.177194399048997, 10.504503944924544, 9.7311872053705, 10.72403618620674, 10.405771544218796, 10.843568173959666, 10.260308068684608, 9.395855001842449, 10.507655324544244, 10.598161447343374, 10.529004873227093, 10.238681483525875, 10.467638379461873, 10.309147200947448, 10.081423217407897, 8.45498657642207, 9.74691829357984, 10.56536191389195, 10.826722084950347, 10.2832761455411, 10.620491250691488, 4.666205506843427, 10.923776334726458, 10.369451611748827, 1.0618979440384149, 10.122804010764298, 10.565838350469365, 10.453327158719668, 10.722392374258156, 10.586747387800065, 10.711182571511436, 9.624339179814731, 10.50027291869185, 10.478536677826407, 10.336903341520964, 10.710796369199285, 3.669874875023079, 10.40907649809808, 10.155171994565421, 10.49377829065147, 8.404503806614084, 10.565671636191778, 8.237746909714879, 10.38795929669997, 10.622119268989774, 10.705564164293108, 6.4132601353677225, 10.62527901061057, 10.704995923989314, 10.114260456729038, 9.679552451804648, 10.15161295255484, 10.808270925044873, 10.532527566707172, 10.408227561789156, 10.436210141115838, 10.811402255260438, 10.496176138514748, 10.531755586064072, 10.454404001998336, 10.582203715469669, 10.524333267715699, 9.753475880658872, 10.58074660620522, 10.469313707591413, 10.48749614498464, 8.783507163152974, 10.343410195313961, 10.342134064624906, 9.864734776584164, 10.557352336639768, 10.563360152008444, 10.395454805570521, 9.410068539752197, 10.771515050969109, 10.648573764381783, 2.3178162475635116, 10.531352881374083, 10.467887899938237, 10.58019849995732, 10.352234926793653, 10.710780237090415, 10.521439753854667, 10.532395426061735, 10.904910231640438, 10.257303683517724, 9.647725238231171, 10.827414885027217, 10.537630697702868, 10.203270946610486, 10.35199711845496, 8.202542031987598, 10.599284616451884, 10.689537915504395, 10.161447902750616, 10.628142079847043, 10.7848027736213, 10.456184675624563, 10.156852730005712, 10.428323936995424, 10.642126926102208, 10.53707669939944, 10.54699974511679, 10.404381325805469, 8.353178118374588, 10.014468780504759, 10.612202798868775, 10.807359978690517, 10.593338283961751, 10.373541954055169, 10.668969852948887, 10.739846438930948, 10.186447121208289, 10.32141587584117, 10.584142845817581, 10.571325788305368, 10.39215988637448, 10.273812010392296, 10.516614557432991, 10.459456790608526, 10.09488550869054, 10.468064052057109, 10.802135523660382, 10.566435281105088, 10.64754561457447, 10.469742790081968, 10.507412555577188, 10.557231470151395, 10.732513594387552, 10.520579152065476, 10.538638039296197, 10.2178865799621, 10.87870754806696, 10.190335132079676, 10.368121953354112, 10.444151367624082, 10.682720975839729, 10.724602372144554, 10.430812090107976, 10.836573448263366, 10.57059606942305, 10.64040371554317, 10.676089122236826, 10.47177546861519, 10.703625415219253, 10.398603458267118, 10.454570714447723, 10.309498540453491, 10.43793599880477, 10.626517207672523, 10.63794505640611, 9.655811508980047, 10.551614563908885, 10.119907603479342, 10.48732815520845, 10.745113581456842, 10.55577049503172, 10.400585535219488, 9.779055984999903, 10.520697598426917, 10.434609741572933, 10.314201117564593, 10.581769125424264, 10.487175894930576, 10.155652168186286, 10.691747458295437, 10.488298847255432, 10.476844076138763, 8.412529497967274, 10.678830633567825, 10.325588785216231, 10.301983309124067, 10.190016823783367, 10.529209566645886, 10.716998748569079, 10.340050158021594, 9.934867169887571, 7.482737029209926, 10.591485368967982, 10.453098379665448, 10.557687712566556, 10.30711854265697, 8.707554824396018, 10.408100051915579, 10.693009558806297, 9.933945152646366, 10.344863398561262, 10.755334704625838, 10.374864543791004, 10.520932264902662, 10.51927120409802, 10.50325614425612, 10.618121749149354, 9.878237861746534, 10.742192379910156, 10.519583609956719, 10.401162057737276, 9.940103733008167, 9.413489844400127, 10.39545995840261, 10.378142605295498, 10.72092794444762, 10.134561536006787, 10.565405817691104, 10.512698965966445, 10.569771753289944, 10.687982250779047, 10.473618262591172, 10.621779917139255, 10.193871580759552, 10.562293827563845, 9.758723167338834, 10.589570080624702, 10.437606731938324, 10.385877890339648, 10.758573731637863, 10.495499145817107, 8.5953822226573, 10.632446167895091, 9.612985000753369, 10.768812756315018, 10.378843932944553, 10.317945493116788, 10.642571050667698, 10.501703983890353, 9.815501225030642, 10.647581793325761, 10.477416177908372, 6.813539366375018, 10.628889847745992, 10.590018749141402, 10.620702256863773, 6.601578386746357, 7.68162391123149, 10.482496574388138, 10.477370137337752, 10.710121267304023, 10.773127158746702, 10.433409146796595, 8.912267079682573, 10.092610500865185, 10.670648892157107, 8.488586547479297, 9.922435540223148, 9.292882712007923, 7.895387610425133, 10.323945195702546, 10.627750507515652, 10.372874704827057, 10.80215017571385, 10.643806783809534, 10.508638752889272, 10.688189073838487, 10.18583737996633, 9.87489246158517, 10.536693601566865, 10.20961527401712, 11.050824030638974, 10.117733268608422, 10.611051100289275, 10.336367230839485, 10.595720253606483, 10.78267764555272, 10.492802293894611, 10.56312766815587, 10.446857565085669, 10.58663154683606, 10.4457846078264, 10.51738368305721, 10.341666197246179, 10.701661598450903, 8.499289502962293, 10.576034738310517, 11.025400823710948, 8.96550396651486, 10.382893330119563, 10.442998313807944, 10.382730738255455, 10.550067097635711, 10.38414053417196, 10.497834513709636, 10.457155968419592, 8.76617495213104, 10.833342723516786, 10.547713103907787, 7.517968906002295, 10.459293996903126, 10.354716845773428, 10.209995580941477, 9.775577658199923, 10.578139341630767, 10.487774587705708, 10.65222168445245, 10.40945488142041, 10.607262806567777, 10.59782254794743, 9.61768111033957, 10.360426893078275, 10.534961931024824, 10.367031403332449, 9.805014904577439, 10.38571719462955, 10.640011182632382, 10.642219520828196, 10.493703534917689, 9.853841601397866, 10.532357063695502, 10.47916289670418, 10.947128240236003, 10.530952617807152, 10.627358595442118, 10.726881109662807, 10.662960404919428, 4.758879363148023, 10.45738187370151, 10.050893672569186, 10.577817815852999, 10.089886591877566, 10.720058574530814, 10.22682229125708, 10.453743584673104, 10.641897574511269, 10.463511161476456, 10.814735747089696, 10.427016780047055, 10.6099701836667, 10.492916629389498, 10.451100489726802, 10.58912288301407, 10.738674325255595, 11.069650594045914, 10.246660092527472, 10.606988345194909, 8.626225703844268, 10.518769124881548, 10.317073587879875, 10.496398282463456, 10.628874201278729, 10.612521567299142, 10.453575402406583, 10.453890825575915, 10.67728411406185, 10.230820017894757, 10.686395531057299, 7.415348658042253, 10.557655910066426, 10.277509258713161, 10.256173985369186, 10.247012379229895, 10.488385887358701, 10.477824318538016, 10.597925165281803, 9.257101033403236, 10.616878088481734, 10.560097362534915, 10.806098978427224, 9.717604113972117, 10.211902970015032, 0.21475392783767755, 10.827870436907649, 10.757789245333472, 10.392812085333953, 10.539083076237803, 10.581299225911938, 10.667781141506074, 10.460373204708425, 10.249615522285447, 10.418480355004483, 10.874663837887262, 10.460712928079056, 10.582345122119136, 10.53578750822741, 9.199567265904774, 10.426637450835035, 10.914307733447718, 10.517653776640412, 10.195968065441502, 10.489239220213069, 9.540782297501009, 10.331775795255941, 10.441109985064328, 10.91222232912103, 10.230177986989986, 10.498688704053265, 10.650280484782956, 9.477222959239688, 2.927268578915126, 10.376533349671355, 10.692843268731057, 10.455435711877513, 10.713042451344222, 10.528797205877801, 10.420872334351351, 10.345773410429059, 10.43808159097643, 10.209276053868486, 10.407133075479972, 10.02320135703749, 10.203648894191396, 10.531718839986466, 10.3880835617859, 10.62350943410736, 10.341750861295054, 9.983628947999076, 10.167816363297035, 10.564605235436705, 10.423334150042399, 10.516579144640817, 10.387919948135615, 10.718086459892827, 10.623596694577985, 10.255263629352825, 10.43480341178346, 10.644586056743965, 10.715013476311938, 10.552190730106188, 10.628614132253986, 10.657145503070543, 10.834341232172907, 10.900589511460383, 10.579850058013838, 7.750260134486241, 10.413578422554222, 10.379424332772185, 10.580762392644862, 10.745074178308274, 10.793744018666347, 9.487270083044596, 10.431738484669685, 8.319197203951298, 4.576554478713939, 10.912288672788506, 9.801043454085177, 10.781725864693964, 10.857059749386114, 10.61534990659439, 10.668099754052177, 10.45409106640308, 9.34734735814392, 10.471209256004174, 5.669846719647225, 10.496465434496429, 10.319020162463223, 9.950952672838975, 10.356841173320591, 10.525954345954451, 10.673379743538202, 10.636955700359724, 10.510381715624845, 10.39059486555018, 10.460193173450886, 10.560081847691006, 9.63086673148408, 10.72489909739979, 10.300546351643675, 10.553878716554497, 10.497719608735409, 10.177667119526308, 10.539895116593165, 10.556309197409197, 10.362194141178513, 10.750963671342276, 11.039329874151566, 10.573565117307606, 10.580415483901563, 10.34697791910632, 10.623828269779354, 10.655437823737893, 10.486456963889834, 10.59774776168444, 10.628772743728248, 10.61713844093872, 10.706110032574093, 10.778408286931759, 10.695347419542728, 10.55838377459736, 10.248792008679484, 10.490922954172742, 10.49127864113987, 10.537449351578095, 10.60177543659457, 10.816398706896656, 10.653609660837018, 10.054357595355118, 10.70589500350066, 9.942378946438955, 9.895815105727548, 10.443210975383483, 8.829508420828558, 10.569900309265314, 10.628759247290919, 10.843038010749268, 10.628630119854646, 10.471869581553086, 10.595324578488896, 10.370439629689963, 10.5984688020181, 10.522849328144778, 10.368484859832138, 10.177021056010572, 9.87596935459396, 10.646969068865005, 10.620118225034854, 10.29855093439762, 10.525455246095937, 10.579420929761733, 10.671882182926245, 10.399458770400715, 10.216723969201485, 9.890761142518778, 10.54921937404258, 10.760599234743772, 10.632971414286061, 10.553832281882713, 10.630098389795814, 10.318797070998329, 10.853354928077998, 10.66482769953744, 10.663910711909054, 9.822456242719227, 10.174929115057632, 10.352135582721141, 10.337927980440318, 10.614612152272441, 10.500097912407607, 8.411246492722725, 8.063898269006028, 9.818584269856453, 10.38369839032423, 10.40446322197636, 10.72488143776222, 10.555647280193662, 10.49646829872395, 10.166449494746065, 10.035521531368225]
AVOIDCORR
Most correlated pair: (8, 9) (Fatick, Foundiougne)
Cut district: Cut district: 9 (Foundiougne)
u, CI: 1.2764127360273676, (1.216187066780611, 1.3366384052741243)
losslist: [10.921207559894384, 10.765174460945092, 10.154396609489988, 10.718091551856395, 10.926780601548394, 10.980029297172676, 10.227593044461658, 10.101678104069524, 9.974776209516223, 10.923016382433575, 10.99856080568358, 10.938376098177104, 10.739921330963217, 10.760857655054856, 10.882569783067797, 10.69586270135527, 10.726732384028349, 10.169665095615338, 10.644799182306313, 10.887477747567333, 10.799178236045961, 10.711198639747982, 10.719247627179092, 10.535102033020381, 10.063191136400466, 10.406971973292734, 10.883976171559274, 10.785351503254658, 9.029571468192978, 10.536723231458561, 10.641528435732818, 10.476843183973688, 8.646697538816872, 10.773583360685668, 10.694475135104224, 10.560725499123672, 10.54651555050408, 10.132637735717648, 10.116538462439264, 10.917317813536403, 10.574455388240466, 10.688785807651634, 10.478832232987369, 10.842820630062032, 10.572850583377843, 10.86680242080006, 10.578659414179251, 10.221210400768149, 10.515678022299042, 10.659398020608768, 10.57586928495374, 10.674698582738632, 10.471748564391518, 10.5566613188369, 10.874712200884375, 10.65565690094569, 10.380570120378424, 10.67161844074281, 11.21608304267016, 10.447450433562713, 10.459515845012383, 10.555344431594122, 10.725405189675477, 10.877782541815513, 10.43530428224467, 10.85187629315849, 10.594289351563317, 10.585169931991667, 10.311255562297031, 10.652236036053964, 10.59990502020141, 10.783865829850658, 9.801885650662747, 10.63225837958999, 10.3126187792598, 10.323579306520816, 10.51030027167221, 10.141001682027873, 10.716692098458703, 10.646992496872425, 10.72858255182288, 10.705963520231874, 10.32960621998939, 10.672794731591791, 10.912454609089759, 10.534569453513534, 10.653922795278023, 10.51230439264589, 10.549651789822565, 10.686233938030313, 10.664493839029667, 8.734211114451417, 10.61031288020779, 10.695325767740236, 10.713224447761446, 10.837623540551922, 10.88957321624909, 10.57450156549772, 10.915907973093656, 10.598218876777327, 10.68521377599964, 9.939249432902129, 10.419046127923288, 10.300433611411034, 10.31560991943568, 10.557442927896744, 10.615981008832998, 10.13569423936358, 10.619509888429326, 10.778121222621788, 10.529383572782342, 10.602137091724428, 10.410763879782213, 10.588254548611976, 10.78753319688546, 10.773948715953098, 10.829559772298385, 10.618815242179753, 10.529423342291553, 10.786532657590632, 10.405331999498067, 11.02449236874665, 9.80989589278274, 10.586297808457234, 10.280068275519294, 10.532752034902582, 8.514491556604888, 10.292109837399286, 10.224276679690082, 10.799660952324059, 10.594692080928379, 9.92730249912709, 10.671612750743295, 10.533469185511926, 10.75695312299199, 7.589491063718638, 10.759645536434629, 10.788220151744834, 10.502489777846574, 10.77276461007273, 10.663576106229012, 10.742022707488049, 9.690334240004253, 10.742901414613735, 10.441649925938023, 10.54808174476807, 10.97140607158512, 10.837519657847638, 10.977224503375595, 11.04635201299915, 10.214295873630974, 10.484358001779645, 10.777958080271874, 10.955476083137492, 9.856800181019512, 10.812501188040947, 9.655497140530034, 10.69423335615276, 10.753135864243621, 10.657894408793284, 10.746760504247472, 10.763375075428764, 10.760362415202572, 10.588339281950509, 10.717008048449715, 9.7249981593366, 10.361604023790656, 10.631025178574573, 10.377915779330976, 10.84416943023592, 10.53802744476685, 10.875637559587506, 10.020079708028067, 10.769453167805056, 10.327512527267578, 10.383410355360589, 10.787848296406759, 10.635213576635309, 10.85112064856565, 10.648491315566543, 10.81690448936204, 10.814048686496413, 10.374593108579704, 10.436876591130833, 10.48478860059757, 11.01024786681714, 10.407340630847495, 10.53034213933064, 10.510209364809127, 10.797253809849721, 10.943927138293647, 10.422955204196125, 10.774192715220828, 10.689782516409217, 10.27041423649399, 10.757243887634466, 10.567483088921511, 10.66386578746177, 10.66602408581255, 10.84276137838845]

path 2
init_u, CI: 1.1616119907975282, (1.1045601027555385, 1.2186638788395179)
losslist: [11.090545965565976, 10.684749917309334, 10.687530902854634, 10.990602986825525, 10.69329848007258, 10.690241672450895, 10.888733750206459, 10.769708797261938, 10.732306989050908, 10.75689944978663, 10.66024080102558, 10.580114671890561, 10.9076605823383, 5.216991887970009, 10.623394402664545, 10.692295388193505, 10.95029098779358, 10.725734040303259, 10.755198821392023, 10.884721515394045, 10.625167481320572, 10.79851346284922, 10.92279724816807, 10.79321936431187, 10.555274023869819, 10.61207296776714, 10.66956844502148, 10.72682817162523, 10.134896131868652, 10.826431065040126, 10.689120341815357, 10.818100084000104, 9.857443492769423, 10.938324644136022, 10.623988106116652, 10.804753492985867, 10.883325758243044, 10.667753647638843, 10.909003676586334, 10.658601230280048, 10.632540940773806, 10.711988146709988, 10.736422266639932, 10.696921427285288, 10.734565572431682, 10.320385655959434, 10.93117130720091, 10.589570693336514, 10.783362495711867, 10.754470426743001, 10.874570449463073, 10.683009963727043, 10.773439400840513, 10.323422324040676, 10.782898087299516, 10.970469348170814, 10.811017359725266, 10.81561405674407, 10.351156459934229, 10.743823897852142, 10.7646505388545, 10.94562289875044, 10.592139183226987, 10.55472485451957, 10.713103849814074, 8.703191254804397, 11.048751943575095, 10.717618275034333, 10.788853933334853, 10.566019754800028, 10.778912283011715, 10.97522337047457, 10.890205599690582, 10.888910578812853, 10.750803028021942, 10.68227737637479, 10.699169664216958, 10.71739369466621, 10.646897661854052, 11.039593413033785, 10.781909862392155, 10.752990584483689, 10.955760789808048, 10.78370742274363, 5.399771057387333, 10.851768350975178, 10.772120113347285, 10.837067871208145, 10.922498043578505, 10.897757899848205, 10.692523020315786, 10.941109911967297, 10.747732292729456, 10.84341032998254, 10.837545704739155, 10.533001935786162, 10.840689160796972, 10.642507086175813, 10.74307593385904, 10.566868315583248, 10.883024737109485, 10.991720970387444, 10.525014741903693, 10.659685521690326, 10.61493248789484, 10.643104459700309, 10.718736045791514, 10.961071705339348, 10.78350606421239, 10.64368191672065, 10.687002611545102, 10.69121104800978, 10.692830685836011, 10.698365236545714, 10.893112269088473, 10.503157702655736, 10.733274481378386, 10.832025354076245, 10.947222655815729, 10.69178961355448, 10.61096137318715, 10.77641570204047, 10.044072907687832, 10.870730518958146, 10.647048544919105, 10.703441074031213, 11.026819682088032, 9.481995536498518, 10.895458606106919, 10.766350619442811, 10.942967482708958, 10.65764268488336, 10.829961404138206, 10.827015714469411, 10.54867005120905, 10.875431169816672, 10.734772972756122, 10.592750872028274, 10.964618007435568, 10.619075541519267, 10.864728794904305, 10.508741435975919, 10.697767092208263, 11.146498001610244, 11.058229432864596, 10.678783162384772, 11.066984854644447, 10.597722893069372, 10.487836221434303, 10.62252377947809, 10.838394433714607, 10.753471356497887, 10.741110609682973, 10.82622533362051, 10.901011434419823, 10.647801987972498, 10.592437581412874, 10.604518711879667, 10.673493931403899, 10.651909308425068, 10.601943527758234, 10.87484296830792, 10.968034366588322, 10.972299360546678, 10.8101551354866, 10.69316896178437, 10.2483758801511, 10.910153940246131, 10.722477960873768, 9.370829202489208, 10.944305536396563, 10.759410759428311, 10.412985508822429, 10.70451991548074, 10.888476757778221, 11.092322457951209, 10.72427300052998, 10.719184536745784, 10.835363368447037, 10.484679076834237, 10.832305679078031, 10.752857010766627, 10.985551527217343, 10.581962626908489, 10.295706222708953, 10.350274791707305, 10.678745826610392, 10.910229107222523, 10.706037218101905, 10.756372863826789, 10.623553521171168, 10.841102426781625, 10.716944765566211, 10.837381034930305, 10.8658093892142, 11.068655970335245, 10.802180725670436, 10.257168163854917, 10.89598803941567, 10.579476371609703, 11.218123951445996, 10.731980862087731, 10.990317616395819, 10.57026027728632, 10.876750424741878, 10.885683566893942, 11.04525435104995, 10.70895109489298, 10.700314135337882, 10.785176332178562, 10.760680166573263, 10.778344744136092, 10.834276456908771, 10.534771678689893, 10.369491779983697, 9.770640824525337, 10.80167439286053, 10.215493114505588, 10.716033395950808, 10.47077822610393, 10.790518290607798, 10.62621968281134, 10.768402758142777, 10.769931195083108, 10.938118873423265, 10.161969474768421, 10.445301463121446, 10.96488870210868, 10.874072841103935, 10.841350954579168, 10.643522034051383, 10.836761257615628, 9.757245175621263, 10.734652651302005, 10.580976185953665, 10.850862721782478, 10.759345657722424, 10.768184009450536, 10.833422032588391, 10.437557060975431, 9.654609690239418, 10.656393835150618, 10.68997493202706, 10.731564289567126, 11.001453901590711, 10.671264205655465, 10.762726098793198, 10.742723044168644, 10.47341384851646, 11.02501241274095, 10.406030589454275, 10.087164579490796, 10.818120488574145, 10.835535249764344, 10.998272350388168, 10.496384725727745, 10.808585856560804, 10.672738139199394, 10.589344250580687, 10.78850422655036, 10.76018914062614, 11.065088149829355, 10.660322415130404, 10.822192737632324, 10.840477478637203, 10.984386194152808, 10.849397707082673, 10.46814577245018, 10.43008445875439, 11.089690022257779, 10.620939895708773, 10.595449428836883, 10.601645320332748, 10.891584615379319, 10.96805183875852, 10.908385500193956, 10.543253014951295, 10.543794744972063, 10.246920949838593, 10.589634536046376, 10.86014480667619, 10.439920187997922, 10.838179047721239, 10.64421963109433, 10.826013987269734, 10.622342459107449, 10.834496959864447, 10.653203457784533, 10.85235357193895, 10.908368304745151, 9.487814237148777, 10.770014026520048, 10.743243587981926, 10.6142762717158, 10.8324548496657, 10.788373472035314, 10.89786781311028, 10.771465884276873, 10.948372543540712, 10.98763085959491, 10.76436693078766, 10.74100573940127, 10.732015237860084, 10.66456495907938, 10.836924503040228, 10.895767776505997, 10.526184271270415, 10.853142409928495, 10.393467826750165, 10.857091831143714, 10.2285255036157, 10.53932271192463, 10.790726778685139, 10.710304184655351, 10.595062198202834, 10.549965848760923, 10.724886498862611, 10.664859094615164, 10.77058800031603, 10.821110338691014, 1.450699459130054, 10.559399256986795, 10.6141213910533, 10.51415650971516, 10.917287023605079, 10.773890900629974, 10.902779526000717, 10.739132906414785, 11.163670120403422, 10.68629533864847, 10.733897952313168, 10.780394453461064, 10.74822410861257, 10.497002955904817, 10.908817397851406, 10.581319188246608, 10.464258110357694, 10.842902390381662, 8.231333577181701, 10.714347228994505, 10.891226176991148, 10.827459307097184, 10.624620698005867, 10.95227889487437, 10.832377221465428, 10.864305927653975, 11.089286627822576, 10.542092230960275, 9.73724780047912, 10.31486700136033, 10.638424708745427, 10.624973499965138, 10.706838486705903, 10.85319501635971, 10.763332155859207, 10.864053843734638, 10.933189055709247, 10.78536146713394, 10.902662376129644, 10.88193401119001, 10.957761483829154, 10.825380362341312, 10.843662129700968, 10.776554857841646, 10.884034916606302, 10.950777247606112, 10.980721361032638, 10.823900197001219, 10.747721660744265, 10.575576965911923, 10.406007032112468, 10.480334188402658, 9.555344363334314, 10.809002855838735, 10.763882182254106, 10.83564439397518, 10.674164094645846, 9.395255351704117, 10.902401844474088, 11.142808488849061, 10.877124395271094, 11.07169891732756, 10.82398793051172, 11.086541067716782, 10.671666898112665, 10.741370259035971, 10.804269494302472, 10.782948455307315, 11.06213072802745, 10.861308232237025, 10.885979270718256, 10.892251387805668, 10.930271783884072, 10.74636973684985, 10.73211248850514, 10.709068651334372, 9.7075051170658, 11.092274873259433, 10.983151786211044, 10.742354108278528, 10.696836546198284, 6.005525948517048, 10.494129024868114, 10.454470606366145, 10.763020961128722, 11.004796801976406, 10.785196844845396, 10.846759263622761, 10.837318504763516, 10.829311746023155, 10.862554058246106, 10.895002945229814, 10.689995166821621, 10.885154118609679, 10.697040985743259, 10.921286149078648, 10.250820253891586, 10.819890466586049, 10.498327170052, 10.607767770639631, 10.82602828630965, 10.691112054996646, 11.030644834198458, 10.82192551623827, 10.93158277327017, 10.31231095643552, 10.629319685347015, 10.775161366559288, 10.73265199447707, 10.686535073519142, 10.62614171633978, 10.401950971623727, 10.84602190305134, 9.015514521150402, 10.94290060880157, 10.579775343191546, 10.805209674277707, 10.804447894482733, 10.008871129235096, 11.055089057650914, 10.953027621431168, 10.550440380464925, 10.74174676549517, 10.595602502762649, 10.69234853417696, 10.66342382951929, 9.611041466774086, 10.70165407984501, 10.530413514327993, 10.27412706434316, 10.739899433089994, 10.437797211361575, 10.671161688749718, 10.568555256459346, 10.80748164009928, 10.825405130191063, 10.75661627507033, 11.006259511990649, 10.693502181560039, 10.776994091703774, 10.71122467613078, 10.705134666839173, 10.928301182895611, 10.781253494035278, 11.056599666779057, 10.961916288976454, 10.913339739499998, 10.57783824013107, 10.738706390127781, 10.43675558648665, 10.670467031444538, 10.717042818827675, 10.5963117930683, 10.75441648923901, 10.638113475894516, 10.836848697188147, 10.42944220244708, 10.80371573862081, 10.440035661056546, 11.009432898125535, 10.627462373909337, 10.336760656411817, 10.984805756049184, 10.363684011026228, 11.001608673727254, 10.59403922668964, 10.734390923656377, 11.235632287503245, 10.95551919083364, 10.73575739874609, 10.671661461603605, 9.949832547151612, 10.885444354720658, 10.533799184495804, 10.71551177702351, 10.90156950168708, 10.530468598011666, 10.931807283391318, 10.897913617828406, 10.864825303062664]
AVOIDCORR
Most correlated pair: (6, 20) (Dakar, Keur Massar)
Cut district: 6 (Dakar)
u, CI: 1.0850895475972102, (1.0344833165187381, 1.1356957786756823)
losslist: [10.687613149469069, 10.796616775932348, 10.644189715584483, 10.988230036024074, 11.097291650249565, 10.702924644515733, 10.4015051981588, 10.72743693150009, 10.661182511246794, 10.389316598588014, 10.73414398600746, 10.70957315807797, 10.574253476832114, 10.562315977687172, 10.801086964213802, 11.08963711819561, 10.709254901907283, 10.630737157635279, 10.655679356651413, 10.73306786794378, 10.881457131246068, 10.803101845183713, 10.656517827244492, 10.885107672526956, 10.852392035280673, 10.573968589879986, 10.343752298830648, 10.7212961718947, 11.134759437945599, 10.562407846606282, 10.703160731147223, 10.903691148632417, 11.01757600284999, 10.705244425626743, 10.598775577268379, 11.016395775888116, 10.67017962287135, 10.917588122182517, 10.531689306596686, 10.831143595028163, 10.826245242112146, 10.661892631573261, 10.74633539601213, 10.693969569547171, 10.593927468127905, 10.63450532977814, 10.867677022796764, 10.762923786899735, 10.626583906935927, 10.426276251966417]

path 3
init_u, CI: 1.5054090994868403, (1.4349911746252015, 1.575827024348479)
losslist: [10.339353069265863, 10.342504754346221, 10.596360248820977, 10.634553451704981, 10.225192366463059, 10.544405150534695, 10.62103141155875, 10.26103289789526, 10.031120540493495, 10.513766359691793, 10.06109354521619, 10.365324826120599, 10.041156276538397, 10.674624898790785, 10.92870363380845, 10.689069509213432, 10.59758841671925, 10.354081142899036, 10.438514730488826, 10.729706324628069, 10.580366372993643, 10.147365477061651, 10.350588461199145, 10.81750430284497, 10.31751840176173, 10.41884933412241, 10.807192413981396, 10.48420679471945, 10.632606340813439, 10.984966361173871, 10.350899846617748, 10.56468902600252, 10.155078186002275, 10.035663729398037, 10.664407491899457, 10.339794986411256, 10.183232855376291, 10.892492391103634, 10.15143012140376, 10.479183226588237, 10.49441640544185, 10.460251954197842, 9.503708475223936, 10.533302765140931, 10.562478683262437, 9.82616769698435, 10.335676515639001, 9.934812325755749, 10.511740542898643, 10.08313801231646, 10.357782476786241, 10.507466112453955, 10.463891413268161, 9.54222092595047, 10.64126709735877, 9.442296206528265, 9.925906498457673, 10.386749309316293, 10.629046346478134, 10.279884771318, 8.950552296890992, 10.53311168422728, 10.608090607942769, 10.462816437239669, 10.227314281304142, 10.397304608172036, 10.442963801339962, 10.60158992558387, 10.577557774429566, 10.66777444348583, 10.517883982830808, 10.537633596179145, 9.907890553172988, 10.670480704440097, 10.7694984224367, 10.436544362269323, 9.003643063140174, 10.675379598462499, 10.454313712039802, 10.295388060284862, 9.264183908854575, 10.549865350676072, 10.367389360049915, 10.197895946840664, 10.437061697073208, 10.042908650712521, 10.647910779465917, 10.5669038114679, 10.569469027599173, 10.264733095449632, 9.90979009711611, 10.280354839541962, 7.071746414693325, 10.595877707601497, 10.405602650606637, 10.154471690245161, 10.514588225134492, 10.36965375264673, 10.18976320852473, 6.655158199852548, 10.395612000937854, 10.645980713103219, 10.634900216839544, 10.528166837793394, 10.497747785829251, 10.012262800663157, 10.362260414721414, 10.602072719412147, 9.752522375546134, 10.724328553184748, 10.520556099366999, 10.61217678292842, 10.480645476992693, 10.084282488653209, 10.37377859725673, 10.868813709153871, 10.75647113203348, 10.505509962045135, 10.299431724956438, 10.682784319473019, 10.626511477236395, 10.303428611184048, 10.58792574448635, 10.751522927598133, 10.207948794450063, 10.717792324687274, 10.679532183362111, 4.073611451235816, 10.619026678951927, 10.655173237352711, 10.640674805245476, 8.555339336140385, 10.213955486383828, 10.435412426191444, 10.499478543895076, 10.727963181555397, 10.625450450523113, 10.444345905284571, 10.683139524191827, 10.14556937376089, 10.34166034173526, 10.528906502075936, 10.21900538949975, 10.49821642072285, 10.326148518730303, 8.301709667750492, 10.717914392191588, 10.922532406567283, 10.448183435242903, 10.578262996392057, 9.778907187219179, 10.464595023195269, 10.41811689068368, 10.221621699390504, 9.887027023569608, 10.861544683219085, 10.616038888722507, 9.824635684458972, 10.493515308342097, 10.692787852142148, 10.384626445834094, 10.504611535288715, 10.136921933763231, 10.665324504802067, 10.544599856500424, 10.33052210814281, 7.812968592295921, 10.748682041046452, 10.209047429740854, 10.78090554048528, 10.46653240993476, 10.349680191635525, 10.449999677522502, 10.502972615866843, 10.586144318652115, 10.683351801679393, 10.802861397743737, 9.956953673255079, 10.049423502998707, 10.365965336533442, 10.247170501637909, 10.032516919916823, 10.467409488890322, 10.44903933984028, 10.572854081050998, 10.689069234708146, 10.429182284847922, 9.710997516204689, 10.679941736152623, 10.519358693549961, 10.797833020368518, 10.428443114133307, 8.55250114833705, 10.47075663563138, 9.83836472782091, 10.603251372557065, 10.041213146547427, 10.558832186528118, 10.853884205404242, 10.652206323331121, 10.62280196001305, 10.669419829716462, 10.405478193396021, 9.940975716457551, 10.614313753401184, 9.903618786414896, 10.531530093980225, 10.456841832164056, 10.6339900264846, 10.666124061487917, 10.714470801666517, 10.695442442031966, 10.37419401758647, 10.692125934659604, 10.24857450418374, 10.16070212112302, 10.382498657991585, 10.563704614149014, 10.711477308131634, 10.4694251118689, 10.059681167521717, 10.611317155671038, 10.486716820776062, 10.526597638336591, 9.23175344898095, 10.526310046818972, 10.794693839814403, 10.577684859824222, 10.67985634324581, 10.702354903809532, 10.609994958219305, 10.457203571149629, 10.612181045074827, 10.435168412094159, 10.98847422072331, 10.567094565099882, 3.970539992643121, 10.811362398479169, 10.336423025883409, 10.666405122105287, 10.607639276238649, 10.65159876409467, 10.666914156913053, 10.505334885682931, 9.992491620115862, 10.712375071061647, 10.748966057551742, 10.546239184423252, 9.419015108590912, 8.248123167749805, 10.243692768259773, 10.04743784339736, 10.469346769613733, 9.956677145630557, 10.342683018267731, 10.458499896163447, 10.780733702621825, 10.198112123836367, 9.961882455606277, 10.68716194148809, 8.454301685918827, 10.562859406339205, 9.564986782363354, 10.521845991725455, 10.517701399976106, 10.881006734659229, 10.457462190701735, 10.718829838728889, 9.22110583566427, 9.910822420586673, 10.313085987877761, 9.793828423050961, 10.199274841676162, 10.533630365003864, 10.550016868308386, 10.281363066897923, 10.676135388169147, 10.625238504916483, 10.580961191639453, 10.573802514773682, 10.572437126229904, 10.411796659787118, 10.841640322558849, 10.601834823541449, 10.910970780046693, 10.625290507114899, 10.071199334353414, 10.128385811719953, 10.524293851406155, 10.480354360417218, 10.292319747516952, 10.36429375738393, 10.416158445788112, 10.610376145827772, 8.454457026997684, 10.471804903932075, 10.59682887770841, 10.453670903248613, 10.541315934231815, 7.066247926802079, 10.296776343789203, 10.381365660814179, 10.226337724226406, 10.392557287294947, 10.371343583839327, 10.567834594255714, 10.591650901618884, 7.964148617793135, 10.467287859603614, 10.139249011314238, 10.85493770349203, 10.615943946874216, 10.019135136625728, 10.536465673986374, 10.639389713799618, 10.36191121789187, 10.475150110843977, 10.479674726771249, 10.48751386384983, 10.756506513046205, 10.30895001137325, 10.195498840532935, 10.669597518182657, 10.623538999123324, 10.295896683840686, 10.574002275643652, 10.18560666949898, 10.596992598006901, 10.609169577959403, 10.309616395410778, 8.232356384157429, 9.983935035319716, 10.565534616095702, 10.320898618620742, 10.234037598637755, 10.45562017229685, 11.031439561152721, 10.761187302042606, 10.267224244971421, 10.369965480752496, 10.373222218956089, 10.515594309697713, 10.505548246074799, 9.060782603519982, 10.853687278499276, 10.682263452209655, 7.599363809635251, 10.069387624644891, 10.758791156767083, 10.409628672087065, 10.431023855215344, 10.893918884590773, 11.11248233066323, 10.786648246337313, 10.68747461478545, 10.582137543568225, 10.418053897472351, 10.625379117012193, 10.494889489877595, 10.525675885549926, 10.78000049999666, 10.258766019237312, 10.576909144956817, 10.661438030625668, 10.977591037869429, 10.79341967474815, 10.240687328464814, 10.514537832487557, 9.986737576127119, 10.275838164067041, 9.953843572314991, 10.490609510246607, 10.523267641443987, 10.716246122160019, 10.177147403821149, 9.299440097737744, 9.83038303301331, 10.602728587487574, 10.203518389209304, 10.658410265941498, 10.514301748983225, 10.857339911592144, 10.43174396448991, 10.726267798670738, 10.284302508200762, 10.41460676048477, 10.3704308166277, 9.02917255756758, 10.723018341483023, 10.678824752057276, 10.469834007141507, 10.573283500135574, 10.539278772173077, 10.068328049279724, 10.541717149949285, 10.57614609082683, 10.58439853956498, 10.437891942193692, 10.359170725939174, 10.247963372329993]
AVOIDCORR
Most correlated pair: (8, 9) (Fatick, Foundiougne)
Cut district: 9 (Foundiougne)
u, CI: 
losslist:

path 4
init_u, CI:
losslist:
AVOIDCORR
Most correlated pair:
Cut district:
u, CI:
losslist:  

path 5
init_u, CI:
losslist:
AVOIDCORR
Most correlated pair:
Cut district:
u, CI:
losslist: 
'''


''' RUN BEFORE DOING PLOTTING BELOW
phase2paths_df.iloc[eligPathInds_sort[0], 5] = 1.212514932297184
phase2paths_df.iloc[eligPathInds_sort[0], 6] = 1.1539925129541722
phase2paths_df.iloc[eligPathInds_sort[0], 7] = 1.2710373516401958
phase2paths_df.iloc[eligPathInds_sort[1], 5] = 1.5988931545712113
phase2paths_df.iloc[eligPathInds_sort[1], 6] = 1.5226730983699461
phase2paths_df.iloc[eligPathInds_sort[1], 7] = 1.6751132107724764
phase2paths_df.iloc[eligPathInds_sort[2], 5] = 1.1616119907975282
phase2paths_df.iloc[eligPathInds_sort[2], 6] = 1.1045601027555385
phase2paths_df.iloc[eligPathInds_sort[2], 7] = 1.2186638788395179
phase2paths_df.iloc[eligPathInds_sort[3], 5] = 1.5054090994868403
phase2paths_df.iloc[eligPathInds_sort[3], 6] = 1.4349911746252015
phase2paths_df.iloc[eligPathInds_sort[3], 7] = 1.575827024348479


'''

# Generate a plot for our improvement runs
xvals = np.arange(1,len(eligPathInds)+2) # Add 1 for our initial feasible solution
UBvals = [phase2paths_df.iloc[i, 3] for i in eligPathInds_sort]
UBvals.insert(0, UB)
utilvals = [phase2paths_df.iloc[i, 5] for i in eligPathInds_sort]
utilvals.insert(0, solUtil)
utilvalCIs = [(phase2paths_df.iloc[i, 7]-phase2paths_df.iloc[i, 6])/2 for i in eligPathInds_sort]
utilvalCIs.insert(0, (2.300548646586247-2.0683527673839848)/4)

# Plot
fig, ax = plt.subplots()
# yerr should be HALF the length of the total error bar
ax.plot(xvals,UBvals,'^',color='deeppink',alpha=0.4)
ax.errorbar(xvals, utilvals, yerr=utilvalCIs, fmt='o',
            color='black', linewidth=0.5, capsize=1.5)
for lst in avoidcorrlist_Gain:
    ax.errorbar(lst[0],lst[1], yerr=[(lst[3]-lst[2])/2], fmt='x', markersize=5,
                color='darkgreen',linewidth=0.5,capsize=1.5,alpha=0.4)
for lst in avoidcorrlist_noGain:
    ax.errorbar(lst[0],lst[1], yerr=[(lst[3]-lst[2])/2], fmt='x', markersize=5,
                color='crimson',linewidth=0.5,capsize=1.5,alpha=0.4)
ax.set(xticks=xvals, ylim=(1., 3.))
plt.xticks(fontsize=8)
plt.ylabel('Utility', fontsize=12)
plt.xlabel('Path index', fontsize=12)
plt.title('Phase II utility evaluations', fontsize=14)
plt.legend(['RP obj.', 'Init. Utility', 'AVOIDCORR: Gain', 'AVOIDCORR: No Gain'])
plt.show()
