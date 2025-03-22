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
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.family"] = "serif"
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
    dept_df = pd.read_csv('orienteering/senegal_csv_files/deptfixedcosts.csv', header=0)
    regcost_mat = pd.read_csv('orienteering/senegal_csv_files/regarcfixedcosts.csv', header=None)
    regNames = ['Dakar', 'Diourbel', 'Fatick', 'Kaffrine', 'Kaolack', 'Kedougou', 'Kolda', 'Louga', 'Matam',
                'Saint-Louis', 'Sedhiou', 'Tambacounda', 'Thies', 'Ziguinchor']
    # Get testing results
    testresults_df = pd.read_csv('orienteering/senegal_csv_files/dataresults.csv', header=0)
    manufNames = testresults_df.Manufacturer.sort_values().unique().tolist()

    return dept_df, regcost_mat, testresults_df, regNames, manufNames

dept_df, regcost_mat, testresults_df, regNames, manufNames = GetSenegalCSVData()
deptNames = dept_df['Department'].sort_values().tolist()
numReg = len(regNames)
testdatadict = {'dataTbl':testresults_df.values.tolist(), 'type':'Tracked', 'TNnames':deptNames, 'SNnames':manufNames}
testdatadict = util.GetVectorForms(testdatadict)
N, Y, TNnames, SNnames = testdatadict['N'], testdatadict['Y'], testdatadict['TNnames'], testdatadict['SNnames']
# Drop all nontested districts
keepinds = np.where(np.sum(N,axis=1)>0)
N, Y = N[keepinds], Y[keepinds]
# Drop supply nodes with less than 5 tests
keepinds = np.where(np.sum(N, axis=0)>4)
N, Y = N[:, keepinds[0]], Y[:, keepinds[0]]

(numTN, numSN) = N.shape # (20, 14)

# Set up logistigate dictionary
lgdict = util.initDataDict(N, Y)
lgdict.update({'TNnames':TNnames, 'SNnames':SNnames})

# Set up priors
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
lgdict['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 2000, 'delta': 0.4}

'''
numdraws = 20000
lgdict['numPostSamples'] = numdraws

for i in range(1, 5):
    np.random.seed(500 + i)
    print('On draw set', i, '...')
    time0 = time.time()
    lgdict = methods.GeneratePostSamples(lgdict, maxTime=5000)
    print('Time:',time.time()-time0)
    tempobj = lgdict['postSamples']
    np.save(os.path.join('studies', 'importancedraws_14MAR24', 'draws'+str(i+1)), tempobj)
'''

# Load draws from files
tempobj = np.load(os.path.join('studies', 'importancedraws_14MAR24', 'draws1.npy'))
for drawgroupind in range(2, 6):
    newobj = np.load(os.path.join('studies', 'importancedraws_14MAR24', 'draws' + str(drawgroupind) +'.npy'))
    tempobj = np.concatenate((tempobj, newobj))
lgdict['postSamples'] = tempobj
# Print inference from initial data
util.plotPostSamples(lgdict, 'int90')

# TODO: GENERATE A NEW Q?

# Build parameter dictionary for utility estimation
paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.13, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN))

def getUtilityEstimate(n, lgdict, paramdict, zlevel=0.95):
    """
    Return a utility estimate average and confidence interval for allocation array n
    """
    testnum = int(np.sum(n))
    des = n/testnum
    currlosslist = sampf.sampling_plan_loss_list(des, testnum, lgdict, paramdict)
    currloss_avg, currloss_CI = sampf.process_loss_list(currlosslist, zlevel=zlevel)
    return paramdict['baseloss'] - currloss_avg, (paramdict['baseloss']-currloss_CI[1], paramdict['baseloss']-currloss_CI[0])

truthdrawslist = [1000, 5000, 10000, 20000, 40000, 80000]
numdatadraws = 100 # Same for all truth draws
numtruthdraws = 80000

# Do 20 tests at 4 nodes and inspect the utility under different numbers of truth draws
n = np.zeros(numTN)
# np.sum(N, axis=0)
n[15], n[1], n[2], n[3], n[9], n[11], n[16] = 50, 40, 50, 30, 50, 60, 30

util_list, utilCI_list = [], []
for numtruthdraws in truthdrawslist:
    truthdraws, datadraws = util.distribute_truthdata_draws(lgdict['postSamples'], numtruthdraws, numdatadraws)
    paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
    paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict) # Get base loss
    util.print_param_checks(paramdict)  # Check of used parameters
    time0 = time.time()
    utilavg, (utilCIlo, utilCIhi) = getUtilityEstimate(n, lgdict, paramdict)
    util_list.append(utilavg), utilCI_list.append((utilCIlo, utilCIhi))
    print(time.time() - time0)
    print(utilavg, (utilCIlo, utilCIhi))

'''18-MAR-24
util_list = [1.0097718208469475,  0.5312661714581854,
 0.5643099023142, 0.4798562562811095,
 0.46251948888548355, 0.43693906930026793]
utilCI_list = [(0.8713481039999795, 1.1481955376939155),  (0.4936725341583901, 0.5688598087579808),
 (0.4943228829455939, 0.634296921682806), (0.44657517296957483, 0.5131373395926442),
 (0.4270171720101965, 0.4980218057607706), (0.4143334143791533, 0.4595447242213826)]
util_list2 = [0.9603319830270349,  0.6643353895719342,
 0.626035598447455,  0.46986298569528584,
 0.4724290016766588,  0.44825931109091455]
utilCI_list2 = [(0.842242948212069, 1.0784210178420008),  (0.5585907279654458, 0.7700800511784225),
 (0.537830726986563, 0.7142404699083471),  (0.4414494273097662, 0.49827654408080546),
 (0.4376701304279651, 0.5071878729253525),  (0.4232933005606201, 0.473225321621209)]
'''

# Plot of estimates under current method
utilerrs = np.array([utilCI_list[i][1] - util_list[i] for i in range(len(truthdrawslist))])
plt.errorbar(truthdrawslist, util_list, yerr=utilerrs, ms=15, mfc='red', mew=1, mec='red', capsize=2)
plt.ylim([0, 1.1])
plt.xlim([0, 80000])
plt.title('Utility estimates vs. truth draws\nCurrent method')
plt.xlabel('Truth Draws')
plt.ylabel('Utility Estimate')
plt.show()

#################################
#################################
# Now generate new estimation with MCMC importance draws
#################################
#################################

design = n/n.sum()
numtests = n.sum()
priordatadict = lgdict.copy()
numdatadrawsforimportance = 10
numimportdraws = 10000

def importance_method_loss_list(design, numtests, priordatadict, paramdict, numdatadrawsforimportance, numimportdraws):
    """
    Produces a list of sampling plan losses for a test budget under a given data set and specified parameters, using
    the fast estimation algorithm with direct optimization (instead of a loss matrix).
    design: sampling probability vector along all test nodes/traces
    numtests: test budget
    priordatadict: logistigate data dictionary capturing known data
    paramdict: parameter dictionary containing a loss matrix, truth and data MCMC draws, and an optional method for
        rounding the design to an integer allocation
    """
    if 'roundalg' in paramdict: # Set default rounding algorithm for plan
        roundalg = paramdict['roundalg'].copy()
    else:
        roundalg = 'lo'
    # Initialize samples to be drawn from traces, per the design, using a rounding algorithm
    sampMat = util.generate_sampling_array(design, numtests, roundalg)
    #todo: KEY NEW STUFF ADDED HERE
    (numTN, numSN), Q, s, r = priordatadict['N'].shape, priordatadict['Q'], priordatadict['diagSens'], priordatadict['diagSpec']
    importancedatadrawinds = np.random.choice(np.arange(paramdict['datadraws'].shape[0]),
                                              size = numdatadrawsforimportance, replace=False)
    importancedatadraws = paramdict['datadraws'][importancedatadrawinds]
    numtruthdraws = paramdict['truthdraws'].shape[0]
    zMatTruth = util.zProbTrVec(numSN, paramdict['truthdraws'], sens=s,
                                spec=r)  # Matrix of SFP probabilities, as a function of SFP rate draws
    zMatData = util.zProbTrVec(numSN, importancedatadraws, sens=s, spec=r)  # Probs. using data draws
    NMat = np.moveaxis(np.array([np.random.multinomial(sampMat[tnInd], Q[tnInd], size=numdatadrawsforimportance)
                                 for tnInd in range(numTN)]), 1, 0).astype(int)
    YMat = np.random.binomial(NMat, zMatData)
    # Get 'average' rounded data set from these few draws
    NMatAvg, YMatAvg = np.round(np.average(NMat, axis=0)).astype(int), np.round(np.average(YMat, axis=0)).astype(int)
    # Add these data to a new data dictionary and generate a new set of MCMC draws
    tempdict = priordatadict.copy()
    tempdict['N'], tempdict['Y'] = priordatadict['N'] + NMatAvg, priordatadict['Y'] + YMatAvg
    # Generate a new MCMC importance set
    tempdict['numPostSamples'] = numimportdraws
    tempdict = methods.GeneratePostSamples(tempdict, maxTime=5000)
    importancedraws = tempdict['postSamples'].copy()
    '''
    for nodeind in range(5):
        plt.hist(priordatadict['postSamples'][:,nodeind],color='black',density=True,alpha=0.6)
        plt.hist(importancedraws[:,nodeind],color='orange',density=True,alpha=0.6)    
        plt.title('Histogram comparison for node '+ str(nodeind))
        plt.show()
        plt.close()
    '''
    # Get weights matrix
    Wimport = sampf.build_weights_matrix(importancedraws, paramdict['datadraws'], sampMat, priordatadict)
    # Get risk matrix
    Rimport = lf.risk_check_array(importancedraws, paramdict['riskdict'])
    # Get critical ratio
    q = paramdict['scoredict']['underestweight'] / (1 + paramdict['scoredict']['underestweight'])
    # Get likelihood weights WRT original data set
    zMatImport = util.zProbTrVec(numSN, importancedraws, sens=s, spec=r)  # Matrix of SFP probabilities along each trace
    NMatPrior, YMatPrior = priordatadict['N'], priordatadict['Y']
    Vimport = np.zeros(shape = numimportdraws)
    for snInd in range(numSN):  # Loop through each SN and TN combination; DON'T vectorize as resulting matrix can be too big
        for tnInd in range(numTN):
            if NMatPrior[tnInd, snInd] > 0:
                bigZtemp = np.transpose(
                    np.reshape(np.tile(zMatImport[:, tnInd, snInd], 1), (1, numimportdraws)))
                bigNtemp = np.reshape(np.tile(NMatPrior[tnInd, snInd], numimportdraws), (numimportdraws, 1))
                bigYtemp = np.reshape(np.tile(YMatPrior[tnInd, snInd], numimportdraws), (numimportdraws, 1))
                combNYtemp = np.reshape(np.tile(spsp.comb(NMatPrior[tnInd, snInd], YMatPrior[tnInd, snInd]),
                                                numimportdraws), (numimportdraws, 1))
                Vimport += np.squeeze( (bigYtemp * np.log(bigZtemp)) + ((bigNtemp - bigYtemp) * np.log(1 - bigZtemp)) + np.log(
                    combNYtemp))
    Vimport = np.exp(Vimport)

    '''
    def cand_obj_val_importance(x, truthdraws, Wvec, paramdict, riskmat):
        """Objective for optimization step"""
        # scoremat stores the loss (ignoring the risk) for x against the draws in truthdraws
        scoremat = lf.score_diff_matrix(truthdraws, x.reshape(1, truthdraws[0].shape[0]), paramdict['scoredict'])[0]
        return np.sum(np.sum(scoremat * riskmat * paramdict['marketvec'], axis=1) * Wvec)
    '''

    # Compile list of optima
    minslist = []
    for j in range(Wimport.shape[1]):
        est = sampf.bayesest_critratio(importancedraws, Wimport[:, j]*Vimport, q)
        minslist.append(sampf.cand_obj_val(est, importancedraws, Wimport[:, j], paramdict, Rimport))
    return minslist

def getImportanceUtilityEstimate(n, lgdict, paramdict, numdatadrawsforimportance, numimportdraws, zlevel=0.95):
    """
    Return a utility estimate average and confidence interval for allocation array n, using a second MCMC set of
    'importance' draws
    """
    testnum = int(np.sum(n))
    des = n/testnum
    # TODO: FOCUS IS HERE
    currlosslist = importance_method_loss_list(des, testnum, lgdict, paramdict, numdatadrawsforimportance,
                                               numimportdraws)
    # TODO: END OF FOCUS
    currloss_avg, currloss_CI = sampf.process_loss_list(currlosslist, zlevel=zlevel)
    return paramdict['baseloss'] - currloss_avg, (paramdict['baseloss']-currloss_CI[1], paramdict['baseloss']-currloss_CI[0])


datadrawsforimportancelist = [10]
numimportdrawslist = [100, 500, 1000, 5000, 10000, 20000, 40000]

utilavgstore_mast, utilCIstore_mast = [], []
for rep in range(1):
    utilavgstore, utilCIstore = [], []
    for numimportdraws in numimportdrawslist:
        currutilavg, (currutilCIlo, currutilCIhi) = getImportanceUtilityEstimate(n, lgdict, paramdict,
                                                                                 numdatadrawsforimportance=10,
                                                                         numimportdraws=numimportdraws)
        print('Utility: ',currutilavg)
        utilavgstore.append(currutilavg)
        utilCIstore.append((currutilCIlo, currutilCIhi))
    utilavgstore_mast.append(utilavgstore)
    utilCIstore_mast.append(utilCIstore)

    # Plot of estimates under current method AND new method
    utilerrs = np.array([utilCI_list[i][1] - util_list[i] for i in range(len(truthdrawslist))])
    plt.errorbar(truthdrawslist, util_list, yerr=utilerrs, ms=15, color='black', mew=1, mec='black', capsize=2)
    utilerrs2 = np.array([utilCI_list2[i][1] - util_list2[i] for i in range(len(truthdrawslist))])
    plt.errorbar(truthdrawslist, util_list2, yerr=utilerrs2, ms=15, color='black', mew=1, mec='black', capsize=2)
    for j in range(len(utilavgstore_mast)):
        utilerrs_new = np.array([utilCIstore_mast[j][i][1] - utilavgstore_mast[j][i] for i in range(len(numimportdrawslist))])
        plt.errorbar(numimportdrawslist, utilavgstore_mast[j], yerr=utilerrs_new, ms=15, mew=1, capsize=2)
    plt.ylim([0, 1.1])
    plt.xlim([0, 80000])
    plt.title('Utility estimates vs. truth draws\nCurrent method (black) and new method (colors)')
    plt.xlabel('Truth Draws')
    plt.ylabel('Utility Estimate')
    plt.show()
    plt.close()

'''19-MAR-24
utilavgstore_mast=[[0.42535895930162093,   0.28920989692718635,  0.19057572244712784,  0.31087961947853815,  0.414815473636164,  0.40610788339433945,  0.3785537026209447],
 [0.3481568238380479,   0.21585755628272985,  0.30615563740333185,  0.4273459335607055,   0.29414752242760045,  0.4662333261877345,  0.40106783867239804],
 [0.014504936719855976,  0.5148634473989988,  0.3057251658124476,  0.36266387444146186,  0.4637119840994939,  0.38378246357960677,  0.4592355310321059],
 [0.41788807038456444,  0.12123940724281557,  0.17877479088845183,  0.39902373768814536,  0.34250762154605985,  0.47284372430861143,  0.3749373426617808],
 [0.7770560665220749,  0.4099297943638782,  0.09128133105268121,  0.4120272223038164,  0.39310601799966216,  0.4018829969406523,  0.4505716324245417]]
utilCIstore_mast=[[(0.2374300893740151, 0.6132878292292268),  (0.17515749371161426, 0.40326230014275843),  (0.09877660612746153, 0.28237483876679415),  (0.2546275903292967, 0.3671316486277796),  (0.37761585732481207, 0.45201508994751594),  (0.36934737540525164, 0.44286839138342726),  (0.3506068298223881, 0.40650057541950124)], 
                [(0.13321160814695965, 0.5631020395291362),  (0.057305101085721066, 0.3744100114797386),  (0.21621904593761077, 0.39609222886905293),  (0.37815624062884146, 0.47653562649256953),  (0.2521437725245952, 0.3361512723306057),  (0.44122551981195635, 0.49124113256351265),  (0.3757227295369532, 0.4264129478078429)],
                 [(-0.20952746782336407, 0.23853734126307602),  (0.4180519177252018, 0.6116749770727958),  (0.22446949562547047, 0.3869808359994247),  (0.3214956967102536, 0.40383205217267015),  (0.43262364280893184, 0.4948003253900559),  (0.35231798418880356, 0.41524694297041),  (0.43854615356411086, 0.47992490850010094)],
                 [(0.22764926976791955, 0.6081268710012093),  (-0.020003875641656954, 0.2624826901272881),  (0.09357788120888788, 0.2639717005680158),  (0.352287259457182, 0.44576021591910875),  (0.30327101232753106, 0.38174423076458863),  (0.438577221596947, 0.5071102270202759),  (0.33754275903351916, 0.41233192629004245)],
                 [(0.5986644466096522, 0.9554476864344976),  (0.31048833688594524, 0.5093712518418112),  (-0.02344780484974196, 0.2060104669551044),  (0.36810597457288496, 0.4559484700347478),  (0.36232614852532974, 0.4238858874739946), (0.36816672579918563, 0.435599268082119),   (0.43130958664805874, 0.4698336782010246)]]  
'''

# Does increasing the number of data draws help?
numdatadraws = 1000 # Same for all truth draws
truthdraws, datadraws = util.distribute_truthdata_draws(lgdict['postSamples'], numtruthdraws, numdatadraws)
paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict) # Get base loss
util.print_param_checks(paramdict)

numimportdrawslist2 = [1000, 5000, 10000, 20000, 40000]
utilavgstore_mast2, utilCIstore_mast2 = [], []
for rep in range(4):
    utilavgstore, utilCIstore = [], []
    for numimportdraws in numimportdrawslist2:
        currutilavg, (currutilCIlo, currutilCIhi) = getImportanceUtilityEstimate(n, lgdict, paramdict,
                                                                                 numdatadrawsforimportance=10,
                                                                         numimportdraws=numimportdraws)
        print('Utility:',currutilavg)
        print('Range:', (currutilCIlo, currutilCIhi))
        utilavgstore.append(currutilavg)
        utilCIstore.append((currutilCIlo, currutilCIhi))
    utilavgstore_mast2.append(utilavgstore)
    utilCIstore_mast2.append(utilCIstore)

    # Plot of estimates under current method AND new method
    utilerrs = np.array([utilCI_list[i][1] - util_list[i] for i in range(len(truthdrawslist))])
    plt.errorbar(truthdrawslist, util_list, yerr=utilerrs, ms=15, color='black', mew=1, mec='black', capsize=2)
    utilerrs2 = np.array([utilCI_list2[i][1] - util_list2[i] for i in range(len(truthdrawslist))])
    plt.errorbar(truthdrawslist, util_list2, yerr=utilerrs2, ms=15, color='black', mew=1, mec='black', capsize=2)
    for j in range(len(utilavgstore_mast)):
        utilerrs_new = np.array([utilCIstore_mast[j][i][1] - utilavgstore_mast[j][i] for i in range(len(numimportdrawslist))])
        plt.errorbar(numimportdrawslist, utilavgstore_mast[j], yerr=utilerrs_new, ms=15, mew=1, capsize=2, alpha=0.3)
    for j in range(len(utilavgstore_mast2)):
        utilerrs_new = np.array([utilCIstore_mast2[j][i][1] - utilavgstore_mast2[j][i] for i in range(len(numimportdrawslist2))])
        plt.errorbar(numimportdrawslist2, utilavgstore_mast2[j], yerr=utilerrs_new, ms=15, mew=1, capsize=2)
    plt.ylim([0, 1.1])
    plt.xlim([0, 80000])
    plt.title('Utility estimates vs. truth draws\nCurrent method (black) and new method (colors)')
    plt.xlabel('Truth Draws')
    plt.ylabel('Utility Estimate')
    plt.show()
    plt.close()

'''19-MAR-24
utilavgstore_mast2 = [[0.3046378685745492,  0.3589733386919298,  0.4269627807813796,  0.3138032225856384,  0.3841857583387087],
 [0.26550867675053214,  0.32015040262360017,  0.3946170864785161,  0.3519793384153598,  0.38707439922085696],
 [0.33858755401230756,  0.4038045925138989,  0.27375132307792294,  0.37100505717146604,  0.43872570234429054],
 [0.3553993344034696,  0.2749817493465505,  0.4293970202141377,  0.3798610321912963,  0.3902590174736158],
 [0.21314121984821144,  0.32901735601705884,  0.3126866297096669,  0.4697617820825424,  0.33484491178271636]]
utilCIstore_mast2 = [[(0.2654116326688052, 0.34386410448029325),
  (0.33134648219557095, 0.3866001951882887),
  (0.41107375016539605, 0.4428518113973632),
  (0.29110400111224966, 0.3365024440590272),
  (0.37226188081769873, 0.3961096358597187)],
 [(0.213361693280254, 0.31765566022081027),
  (0.29721875640404694, 0.3430820488431534),
  (0.37710044116565067, 0.4121337317913816),
  (0.3391396942311413, 0.36481898259957823),
  (0.3742327193721753, 0.39991607906953863)],
 [(0.2975745191081187, 0.3796005889164964),
  (0.374279937763085, 0.4333292472647128),
  (0.2470145652393403, 0.3004880809165056),
  (0.3509581918466207, 0.3910519224963114),
  (0.42760477843805234, 0.44984662625052874)],
 [(0.314385854578934, 0.39641281422800523),
  (0.249278519913668, 0.300684978779433),
  (0.40899908636262294, 0.4497949540656525),
  (0.3625795847291209, 0.3971424796534717),
  (0.3723266356375281, 0.4081913993097035)],
 [(0.17097444859107647, 0.2553079911053464),
  (0.30506841809032803, 0.35296629394378964),
  (0.2921662501768272, 0.33320700924250657),
  (0.45529920213492225, 0.4842243620301625),
  (0.32116573119958947, 0.34852409236584325)]]
'''

# What impact does the number of callibration data draws have?
# Increase to 100 from 10
utilavgstore_mast3, utilCIstore_mast3 = [], []
for rep in range(5):
    utilavgstore, utilCIstore = [], []
    for numimportdraws in numimportdrawslist2:
        currutilavg, (currutilCIlo, currutilCIhi) = getImportanceUtilityEstimate(n, lgdict, paramdict,
                                                        numdatadrawsforimportance=100,
                                                        numimportdraws=numimportdraws)
        print('Utility:',currutilavg)
        print('Range:', (currutilCIlo, currutilCIhi))
        utilavgstore.append(currutilavg)
        utilCIstore.append((currutilCIlo, currutilCIhi))
    utilavgstore_mast3.append(utilavgstore)
    utilCIstore_mast3.append(utilCIstore)

    # Plot of estimates under current method AND new method
    utilerrs = np.array([utilCI_list[i][1] - util_list[i] for i in range(len(truthdrawslist))])
    plt.errorbar(truthdrawslist, util_list, yerr=utilerrs, ms=15, color='black', mew=1, mec='black', capsize=2)
    utilerrs2 = np.array([utilCI_list2[i][1] - util_list2[i] for i in range(len(truthdrawslist))])
    plt.errorbar(truthdrawslist, util_list2, yerr=utilerrs2, ms=15, color='black', mew=1, mec='black', capsize=2)
    for j in range(len(utilavgstore_mast)):
        utilerrs_new = np.array([utilCIstore_mast[j][i][1] - utilavgstore_mast[j][i] for i in range(len(numimportdrawslist))])
        plt.errorbar(numimportdrawslist, utilavgstore_mast[j], yerr=utilerrs_new, ms=15, mew=1, capsize=2, alpha=0.1)
    for j in range(len(utilavgstore_mast2)):
        utilerrs_new = np.array([utilCIstore_mast2[j][i][1] - utilavgstore_mast2[j][i] for i in range(len(numimportdrawslist2))])
        plt.errorbar(numimportdrawslist2, utilavgstore_mast2[j], yerr=utilerrs_new, ms=15, mew=1, capsize=2, alpha=0.3)
    for j in range(len(utilavgstore_mast3)):
        utilerrs_new = np.array([utilCIstore_mast3[j][i][1] - utilavgstore_mast3[j][i] for i in range(len(numimportdrawslist2))])
        plt.errorbar(numimportdrawslist2, utilavgstore_mast3[j], yerr=utilerrs_new, ms=15, mew=1, capsize=2)
    plt.ylim([0, 1.1])
    plt.xlim([0, 80000])
    plt.title('Utility estimates vs. truth draws\nCurrent method (black) and new method (colors)')
    plt.xlabel('Truth Draws')
    plt.ylabel('Utility Estimate')
    plt.show()
    plt.close()
'''19-MAR-24
utilavgstore_mast3 = [[0.13349953844582396,  0.34626552680299394,  0.3365276708126128,  0.36730266420040003,  0.39449208016785375],
 [0.27315157056492545,  0.33999477412152634,  0.27417519733093076,  0.367044331523668,  0.3885877864335314],
 [0.39685785388420625,  0.38902104430112905,  0.40958972890874223,  0.31398880883353897,  0.3247408997336594], 
  [0.11955953644289607,  0.26003606322257244,  0.39404900112853536,  0.41118294952548373,  0.40511777724497966],
 [0.24767831599234658,  0.3948834220126334,  0.2595706987973938,  0.37464331580066723,  0.40517316801806036]]
utilCIstore_mast3 = [[(0.08839849082683982, 0.1786005860648081),   (0.32559358646777303, 0.36693746713821485),
  (0.3165452630733756, 0.35651007855185),   (0.34520264905960607, 0.389402679341194),
  (0.38224681221727375, 0.40673734811843376)],
 [(0.23516754997882083, 0.31113559115103007),
  (0.31360819957752684, 0.36638134866552585),  (0.25012277570742025, 0.2982276189544413),
  (0.35421098940966145, 0.37987767363767455),  (0.3785578805867149, 0.39861769228034793)],
 [(0.3574587757644454, 0.4362569320039671),  (0.36384786055741225, 0.41419422804484585),
  (0.3890201136887299, 0.43015934412875456),  (0.2992015795043037, 0.32877603816277423),
  (0.3140229306365634, 0.3354588688307554)],
 [(0.05897907433914629, 0.18013999854664586),  (0.23649767929946286, 0.28357444714568203),
  (0.3773895109843943, 0.4107084912726764),  (0.39813565950189167, 0.4242302395490758),
  (0.3905142354317195, 0.41972131905823984)],
 [(0.20564772289844058, 0.2897089090862526),  (0.36739156787425165, 0.4223752761510151),
  (0.24092886169373884, 0.2782125359010488),  (0.3600364764492996, 0.38925015515203487),
  (0.3946136114860139, 0.4157327245501068)]]
'''

# How long does it take for the average data set to converge?
# For 20 reps, generate data sets and plot cumulative averages of each trace
numdatadrawsforimportance = 2000
for rep in range(20):
    importancedatadrawinds = np.random.choice(np.arange(paramdict['datadraws'].shape[0]),
                                              size=numdatadrawsforimportance, replace=False)
    importancedatadraws = paramdict['datadraws'][importancedatadrawinds]
    zMatData = util.zProbTrVec(numSN, importancedatadraws)  # Probs. using data draws
    NMat = np.moveaxis(np.array([np.random.multinomial(n[tnInd], lgdict['Q'][tnInd], size=numdatadrawsforimportance)
                                 for tnInd in range(numTN)]), 1, 0).astype(int)
    YMat = np.random.binomial(NMat, zMatData)
    NMat_cumsum = np.cumsum(NMat, axis=0)
    temp = np.resize(np.transpose(np.reshape(np.repeat(np.arange(1,numdatadrawsforimportance+1),numTN*numSN),(2000,20,14)),
                        (0,1,2)), (2000,20,14))
    NMat_avg = NMat_cumsum/temp
    for tnind in range(numTN):
        for snind in range(numSN):
            if NMat_avg[-1, tnind, snind]>0:
                plt.plot(NMat_avg[:,tnind,snind],linewidth=0.3,color='black')
    plt.title('Running average allocation at tested traces vs. number of data draws')
    plt.xlabel('Number of data draws')
    plt.ylabel('Average trace allocation')
    plt.show()
    plt.close()
# 1000 data draws appears sufficient

#################################
#################################
# Slightly different estimation method
#################################
#################################
def importance_method_loss_list2(design, numtests, priordatadict, paramdict, numimportdraws,
                                 numdatadrawsforimportance=1000, impweightoutlierprop=0.01):
    """
    Produces a list of sampling plan losses, a la sampling_plan_loss_list(). This method uses the importance
    sampling approach, using numdatadrawsforimportance draws to produce an 'average' data set. An MCMC set of
    numimportdraws is produced assuming this average data set; this MCMC set should be closer to the important region
    of SFP rates for this design.

    design: sampling probability vector along all test nodes/traces
    numtests: test budget
    priordatadict: logistigate data dictionary capturing known data
    paramdict: parameter dictionary containing a loss matrix, truth and data MCMC draws, and an optional method for
        rounding the design to an integer allocation
    """
    if 'roundalg' in paramdict: # Set default rounding algorithm for plan
        roundalg = paramdict['roundalg'].copy()
    else:
        roundalg = 'lo'
    # Initialize samples to be drawn from traces, per the design, using a rounding algorithm
    sampMat = util.generate_sampling_array(design, numtests, roundalg)
    #todo: KEY NEW STUFF ADDED HERE
    (numTN, numSN), Q, s, r = priordatadict['N'].shape, priordatadict['Q'], priordatadict['diagSens'], priordatadict['diagSpec']
    importancedatadrawinds = np.random.choice(np.arange(paramdict['datadraws'].shape[0]),
                                              size = numdatadrawsforimportance,
                                              replace=paramdict['datadraws'].shape[0] < numdatadrawsforimportance)
    importancedatadraws = paramdict['datadraws'][importancedatadrawinds]
    #numtruthdraws = paramdict['truthdraws'].shape[0]
    #zMatTruth = util.zProbTrVec(numSN, paramdict['truthdraws'], sens=s, spec=r)  # Matrix of SFP probabilities, as a function of SFP rate draws
    zMatData = util.zProbTrVec(numSN, importancedatadraws, sens=s, spec=r)  # Probs. using data draws
    NMat = np.moveaxis(np.array([np.random.multinomial(sampMat[tnInd], Q[tnInd], size=numdatadrawsforimportance)
                                 for tnInd in range(numTN)]), 1, 0).astype(int)
    YMat = np.random.binomial(NMat, zMatData)
    # Get 'average' rounded data set from these few draws
    NMatAvg, YMatAvg = np.round(np.average(NMat, axis=0)).astype(int), np.round(np.average(YMat, axis=0)).astype(int)
    # Add these data to a new data dictionary and generate a new set of MCMC draws
    tempdict = priordatadict.copy()
    tempdict['N'], tempdict['Y'] = priordatadict['N'] + NMatAvg, priordatadict['Y'] + YMatAvg
    # Generate a new MCMC importance set
    tempdict['numPostSamples'] = numimportdraws
    tempdict = methods.GeneratePostSamples(tempdict, maxTime=5000)
    importancedraws = tempdict['postSamples'].copy()
    '''
    for nodeind in range(5):
        plt.hist(priordatadict['postSamples'][:,nodeind],color='black',density=True,alpha=0.6)
        plt.hist(importancedraws[:,nodeind],color='orange',density=True,alpha=0.6)    
        plt.title('Histogram comparison for node '+ str(nodeind))
        plt.show()
        plt.close()
    '''
    # Get weights matrix - don't normalize


    numdatadraws =  paramdict['datadraws'].shape[0]
    zMatTruth = util.zProbTrVec(numSN, importancedraws, sens=s, spec=r)  # Matrix of SFP probabilities, as a function of SFP rate draws
    zMatData = util.zProbTrVec(numSN, paramdict['datadraws'], sens=s, spec=r)  # Probs. using data draws
    NMat = np.moveaxis(np.array([np.random.multinomial(sampMat[tnInd], Q[tnInd], size=numdatadraws)
                                 for tnInd in range(numTN)]), 1, 0).astype(int)
    YMat = np.random.binomial(NMat, zMatData)
    tempW = np.zeros(shape=(numimportdraws, numdatadraws))
    for snInd in range(numSN):  # Loop through each SN and TN combination; DON'T vectorize as resulting matrix can be too big
        for tnInd in range(numTN):
            if sampMat[tnInd] > 0 and Q[tnInd, snInd] > 0:  # Save processing by only looking at feasible traces
                # Get zProbs corresponding to current trace
                bigZtemp = np.transpose(
                    np.reshape(np.tile(zMatTruth[:, tnInd, snInd], numdatadraws), (numdatadraws, numimportdraws)))
                bigNtemp = np.reshape(np.tile(NMat[:, tnInd, snInd], numimportdraws), (numimportdraws, numdatadraws))
                bigYtemp = np.reshape(np.tile(YMat[:, tnInd, snInd], numimportdraws), (numimportdraws, numdatadraws))
                combNYtemp = np.reshape(np.tile(spsp.comb(NMat[:, tnInd, snInd], YMat[:, tnInd, snInd]), numimportdraws),
                                        (numimportdraws, numdatadraws))
                tempW += (bigYtemp * np.log(bigZtemp)) + ((bigNtemp - bigYtemp) * np.log(1 - bigZtemp)) + np.log(
                    combNYtemp)
    Wimport = np.exp(tempW)

    # Get risk matrix
    Rimport = lf.risk_check_array(importancedraws, paramdict['riskdict'])
    # Get critical ratio
    q = paramdict['scoredict']['underestweight'] / (1 + paramdict['scoredict']['underestweight'])
    # Get likelihood weights WRT original data set
    zMatImport = util.zProbTrVec(numSN, importancedraws, sens=s, spec=r)  # Matrix of SFP probabilities along each trace
    NMatPrior, YMatPrior = priordatadict['N'], priordatadict['Y']
    Vimport = np.zeros(shape = numimportdraws)
    for snInd in range(numSN):  # Loop through each SN and TN combination; DON'T vectorize as resulting matrix can be too big
        for tnInd in range(numTN):
            if NMatPrior[tnInd, snInd] > 0:
                bigZtemp = np.transpose(
                    np.reshape(np.tile(zMatImport[:, tnInd, snInd], 1), (1, numimportdraws)))
                bigNtemp = np.reshape(np.tile(NMatPrior[tnInd, snInd], numimportdraws), (numimportdraws, 1))
                bigYtemp = np.reshape(np.tile(YMatPrior[tnInd, snInd], numimportdraws), (numimportdraws, 1))
                combNYtemp = np.reshape(np.tile(spsp.comb(NMatPrior[tnInd, snInd], YMatPrior[tnInd, snInd]),
                                                numimportdraws), (numimportdraws, 1))
                Vimport += np.squeeze( (bigYtemp * np.log(bigZtemp)) + ((bigNtemp - bigYtemp) * np.log(1 - bigZtemp)) + np.log(
                    combNYtemp))
    Vimport = np.exp(Vimport)
    #Vimport_norm = Vimport / np.sum(Vimport)
    ####
    # Get likelihood weights WRT average data set
    #zMatImport = util.zProbTrVec(numSN, importancedraws, sens=s, spec=r)  # Matrix of SFP probabilities along each trace
    NMatPrior, YMatPrior = tempdict['N'].copy(), tempdict['Y'].copy()
    Uimport = np.zeros(shape=numimportdraws)
    for snInd in range(
            numSN):  # Loop through each SN and TN combination; DON'T vectorize as resulting matrix can be too big
        for tnInd in range(numTN):
            if NMatPrior[tnInd, snInd] > 0:
                bigZtemp = np.transpose(
                    np.reshape(np.tile(zMatImport[:, tnInd, snInd], 1), (1, numimportdraws)))
                bigNtemp = np.reshape(np.tile(NMatPrior[tnInd, snInd], numimportdraws), (numimportdraws, 1))
                bigYtemp = np.reshape(np.tile(YMatPrior[tnInd, snInd], numimportdraws), (numimportdraws, 1))
                combNYtemp = np.reshape(np.tile(spsp.comb(NMatPrior[tnInd, snInd], YMatPrior[tnInd, snInd]),
                                                numimportdraws), (numimportdraws, 1))
                Uimport += np.squeeze(
                    (bigYtemp * np.log(bigZtemp)) + ((bigNtemp - bigYtemp) * np.log(1 - bigZtemp)) + np.log(
                        combNYtemp))
    Uimport = np.exp(Uimport)

    # Importance likelihood ratio for importance draws
    VoverU = (Vimport / Uimport)

    # Compile list of optima
    minslist = []
    for j in range(Wimport.shape[1]):
        tempwtarray = Wimport[:, j] * VoverU * numimportdraws / np.sum(Wimport[:, j] * VoverU)
        #plt.plot(j, np.quantile(tempwtarray, 0.5), 'x', color='black')
        #plt.plot(j, np.quantile(tempwtarray, 0.999), 'o', color='blue')
        #plt.plot(j, np.max(tempwtarray), '^', color='red')
    #plt.title('Importance Weight Quantiles\nBlack x=Median, Blue Circle=0.999, Red Triangle=Max')
    #plt.xlabel('Data simulation index')
    #plt.ylim([0,2000])
    #plt.show()
    #plt.close()
        # Remove inds for top impweightoutlierprop of weights
        tempremoveinds = np.where(tempwtarray>np.quantile(tempwtarray, 1-impweightoutlierprop))
        tempwtarray = np.delete(tempwtarray, tempremoveinds)
        tempwtarray = tempwtarray/np.sum(tempwtarray)
        tempimportancedraws = np.delete(importancedraws, tempremoveinds, axis=0)
        tempRimport = np.delete(Rimport, tempremoveinds, axis=0)
        est = sampf.bayesest_critratio(tempimportancedraws, tempwtarray, q)
        minslist.append(sampf.cand_obj_val(est, tempimportancedraws, tempwtarray, paramdict, tempRimport))

    return minslist

def getImportanceUtilityEstimate2(n, lgdict, paramdict, numimportdraws, numdatadrawsforimportance=1000,
                                  impweightoutlierprop=0.01, zlevel=0.95):
    """
    Return a utility estimate average and confidence interval for allocation array n, using a second MCMC set of
    'importance' draws
    """
    testnum = int(np.sum(n))
    des = n/testnum
    # TODO: FOCUS IS HERE
    currlosslist = importance_method_loss_list2(des, testnum, lgdict, paramdict, numdatadrawsforimportance,
                                               numimportdraws)
    # TODO: END OF FOCUS
    currloss_avg, currloss_CI = sampf.process_loss_list(currlosslist, zlevel=zlevel)
    return paramdict['baseloss'] - currloss_avg, (paramdict['baseloss']-currloss_CI[1], paramdict['baseloss']-currloss_CI[0])

utilavgstore_mast4, utilCIstore_mast4 = [], []
for rep in range(5):
    utilavgstore, utilCIstore = [], []
    for numimportdraws in numimportdrawslist2:
        currutilavg, (currutilCIlo, currutilCIhi) = getImportanceUtilityEstimate2(n, lgdict, paramdict,
                                                        numdatadrawsforimportance=1000,
                                                        numimportdraws=numimportdraws)
        print('Utility:',currutilavg)
        print('Range:', (currutilCIlo, currutilCIhi))
        utilavgstore.append(currutilavg)
        utilCIstore.append((currutilCIlo, currutilCIhi))
    utilavgstore_mast4.append(utilavgstore)
    utilCIstore_mast4.append(utilCIstore)

    # Plot of estimates under current method AND new method
    utilerrs = np.array([utilCI_list[i][1] - util_list[i] for i in range(len(truthdrawslist))])
    plt.errorbar(truthdrawslist, util_list, yerr=utilerrs, ms=15, color='black', mew=1, mec='black', capsize=2)
    utilerrs2 = np.array([utilCI_list2[i][1] - util_list2[i] for i in range(len(truthdrawslist))])
    plt.errorbar(truthdrawslist, util_list2, yerr=utilerrs2, ms=15, color='black', mew=1, mec='black', capsize=2)
    '''for j in range(len(utilavgstore_mast)):
        utilerrs_new = np.array([utilCIstore_mast[j][i][1] - utilavgstore_mast[j][i] for i in range(len(numimportdrawslist))])
        plt.errorbar(numimportdrawslist, utilavgstore_mast[j], yerr=utilerrs_new, ms=15, mew=1, capsize=2, alpha=0.1)
    for j in range(len(utilavgstore_mast2)):
        utilerrs_new = np.array([utilCIstore_mast2[j][i][1] - utilavgstore_mast2[j][i] for i in range(len(numimportdrawslist2))])
        plt.errorbar(numimportdrawslist2, utilavgstore_mast2[j], yerr=utilerrs_new, ms=15, mew=1, capsize=2, alpha=0.2)
    for j in range(len(utilavgstore_mast3)):
        utilerrs_new = np.array([utilCIstore_mast3[j][i][1] - utilavgstore_mast3[j][i] for i in range(len(numimportdrawslist2))])
        plt.errorbar(numimportdrawslist2, utilavgstore_mast3[j], yerr=utilerrs_new, ms=15, mew=1, capsize=2, alpha=0.3)'''
    for j in range(len(utilavgstore_mast4)):
        utilerrs_new = np.array([utilCIstore_mast4[j][i][1] - utilavgstore_mast4[j][i] for i in range(len(numimportdrawslist2))])
        plt.errorbar(numimportdrawslist2, utilavgstore_mast4[j], yerr=utilerrs_new, ms=15, mew=1, capsize=2)
    plt.ylim([0, 1.1])
    plt.xlim([0, 80000])
    plt.title('Utility estimates vs. truth draws\nCurrent method (black) and new method (colors)')
    plt.xlabel('Truth Draws')
    plt.ylabel('Utility Estimate')
    plt.show()
    plt.close()

#########################################
#########################################
### ANALYSIS OF PARAMETERS FOR IMPORTANCE SAMPLING
#########################################
#########################################
# Want to inspect sensitivity to:
#   - % outliers removed
#   - # importance draws
#   - # data draws
n = np.zeros(numTN)
# np.sum(N, axis=0)
n[15], n[1], n[2], n[3], n[9], n[11], n[16] = 50, 40, 50, 30, 50, 60, 30

def getImportanceUtilityEstimate2(n, lgdict, paramdict, numimportdraws, numdatadrawsforimportance=1000,
                                  impweightoutlierprop=0.01, zlevel=0.95):
    """
    Return a utility estimate average and confidence interval for allocation array n, using a second MCMC set of
    'importance' draws
    """
    testnum = int(np.sum(n))
    des = n/testnum
    currlosslist = sampf.sampling_plan_loss_list_importance(des, testnum, lgdict, paramdict, numimportdraws,
                                                            numdatadrawsforimportance, impweightoutlierprop)
    currloss_avg, currloss_CI = sampf.process_loss_list(currlosslist, zlevel=zlevel)
    return paramdict['baseloss'] - currloss_avg, (paramdict['baseloss']-currloss_CI[1], paramdict['baseloss']-currloss_CI[0])

outlierRemoveList = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]

mastUtilavgstore, mastUtilCIstore = [], []
for rep in range(10): # 10 reps
    print('Rep:', rep)
    utilavgstore, utilCIstore = [], []
    for currOutlierRemove in outlierRemoveList:
        currutilavg, (currutilCIlo, currutilCIhi) = getImportanceUtilityEstimate2(n, lgdict, paramdict,
                                                        numimportdraws=20000,
                                                        numdatadrawsforimportance=1000,
                                                        impweightoutlierprop=currOutlierRemove)
        print('Utility:', currutilavg)
        print('Range:', (currutilCIlo, currutilCIhi))
        utilavgstore.append(currutilavg)
        utilCIstore.append((currutilCIlo, currutilCIhi))
    mastUtilavgstore.append(utilavgstore)
    mastUtilCIstore.append(utilCIstore)
    '''29-MAR
    mastUtilavgstore = [[0.4869442991632269, 0.4587847610308162, 0.4601644478730842, 0.44860577814476965, 0.46896436181891854, 0.4733897229246806], [0.47795072669177774, 0.4865628841586376, 0.4608773144045504, 0.4627668067902988, 0.4507244642258552, 0.4855995657304666], [0.48560188729767484, 0.4632263435088344, 0.4543215266885281, 0.4633003228131902, 0.4581622634263032, 0.4902303641277266], [0.4740258506300381, 0.47059867878742834, 0.4547884282621837, 0.4561502251136784, 0.46895181065459646, 0.504248466187045], [0.4722950899823779, 0.465398644124829, 0.47034861044936394, 0.46799276335272166, 0.45718660178213044, 0.4585473696640645], [0.49446082394380353, 0.48604358879636944, 0.4589785758869751, 0.4439987825906413, 0.4776784334711799, 0.4796556621970467], [0.47213726037214343, 0.49087419462275905, 0.4665623081163206, 0.45106954910122843, 0.4561792528622939, 0.48353286603685364], [0.4816954978829462, 0.4769766017442292, 0.4554954898016481, 0.47829687218605654, 0.47426284500032834, 0.46214883902417103], [0.4656706906334338, 0.4610874148382611, 0.4566997497588092, 0.4726125603328204, 0.4943643025070181, 0.4919283399940313], [0.4776889336815642, 0.47148551738808964, 0.46292304406492457, 0.4691117617316589, 0.47231512009592924, 0.47380611476676204]]
    mastUtilCIstore = [[(0.4825248946712355, 0.4913637036552183), (0.451048911381724, 0.46652061067990847), (0.4507930539413869, 0.4695358418047815), (0.4387787279490958, 0.4584328283404435), (0.4541093353121255, 0.4838193883257116), (0.45736804513443907, 0.4894114007149222)], [(0.47244072017809025, 0.48346073320546523), (0.48065045236624737, 0.4924753159510278), (0.45203615873958647, 0.4697184700695143), (0.45211201055386585, 0.47342160302673175), (0.4391819116904907, 0.46226701676121973), (0.47096658075881725, 0.5002325507021159)], [(0.4807432284574391, 0.49046054613791057), (0.4562553430194578, 0.470197343998211), (0.44491364969518177, 0.46372940368187443), (0.45058695461744236, 0.4760136910089381), (0.44395826661617255, 0.4723662602364338), (0.4744316609180763, 0.5060290673373768)], [(0.4693283334243441, 0.4787233678357321), (0.46359301799632213, 0.47760433957853454), (0.44469075848463513, 0.46488609803973224), (0.44739893314933443, 0.4649015170780224), (0.45511122156722683, 0.4827923997419661), (0.48797610521920864, 0.5205208271548813)], [(0.4673913206074123, 0.4771988593573435), (0.45860318843671255, 0.47219409981294547), (0.4609659900554548, 0.47973123084327307), (0.4587080380623494, 0.47727748864309394), (0.443376809484219, 0.4709963940800419), (0.44006136382771777, 0.47703337550041125)], [(0.4894556303991542, 0.4994660174884529), (0.4795817590897822, 0.4925054185029567), (0.4491636973149751, 0.46879345445897513), (0.43288460760295555, 0.45511295757832704), (0.46223422271591996, 0.4931226442264398), (0.4627807359704983, 0.4965305884235951)], [(0.4671624478760723, 0.47711207286821455), (0.48510898342062525, 0.49663940582489285), (0.4583580005921286, 0.4747666156405126), (0.4397200151286884, 0.46241908307376844), (0.44373978121925806, 0.4686187245053297), (0.46679513222260915, 0.5002705998510981)], [(0.47665526447267714, 0.48673573129321523), (0.47075049716707706, 0.4832027063213813), (0.44700845751345586, 0.46398252208984037), (0.4690553196911891, 0.487538424680924), (0.4594465035247821, 0.48907918647587456), (0.44598023022398525, 0.4783174478243568)], [(0.46077594203244177, 0.47056543923442584), (0.4548936068071012, 0.467281222869421), (0.449017023757428, 0.4643824757601904), (0.46396229847320347, 0.4812628221924373), (0.48127216207759416, 0.5074564429364421), (0.4761619607363081, 0.5076947192517545)], [(0.4729346779440342, 0.48244318941909414), (0.4659727002050871, 0.4769983345710922), (0.4529533900536622, 0.47289269807618695), (0.45931569204314293, 0.4789078314201749), (0.4611259783034667, 0.4835042618883918), (0.4583647225226586, 0.4892475070108655)]]
    '''
# todo: FIX BELOW
for tempind, currutilavgstore in enumerate(mastUtilavgstore):
    utilerrs = np.array([mastUtilCIstore[tempind][i][1] - currutilavgstore[i] for i in range(len(outlierRemoveList))])
    plt.errorbar(outlierRemoveList, currutilavgstore, yerr=utilerrs, ms=15, mew=1, mec='black', capsize=2)
plt.hlines(0.44825931109091455, xmin=0,xmax=max(outlierRemoveList)*1.05)
plt.ylim([0, 0.6])
#plt.xlim([0, max(outlierRemoveList)*1.05])
plt.xlim([0, 0.011])
plt.title('Utility estimates vs. percent of loss outliers removed\nUsing 20k importance draws, for 10 replications')
plt.xlabel('Percent of loss outliers removed')
plt.ylabel('Utility estimate')
plt.show()
plt.close()


numimportdrawslist = [1000,5000,10000,20000,30000,40000]

mastUtilavgstore_numimpdraws, mastUtilCIstore_numimpdraws = [], []
for rep in range(10): # 10 reps
    print('Rep:', rep)
    utilavgstore, utilCIstore = [], []
    for numimportdraws in numimportdrawslist:
        currutilavg, (currutilCIlo, currutilCIhi) = getImportanceUtilityEstimate2(n, lgdict, paramdict,
                                                        numimportdraws=numimportdraws,
                                                        numdatadrawsforimportance=1000,
                                                        impweightoutlierprop=0.01)
        print('Utility:', currutilavg)
        print('Range:', (currutilCIlo, currutilCIhi))
        utilavgstore.append(currutilavg)
        utilCIstore.append((currutilCIlo, currutilCIhi))
    mastUtilavgstore_numimpdraws.append(utilavgstore)
    mastUtilCIstore_numimpdraws.append(utilCIstore)
'''30-MAR
mastUtilavgstore_numimpdraws = [[0.5374958864043684, 0.4755044530955277, 0.4654855829221094, 0.46383241536562103, 0.4629039790384062, 0.4623620103858608], [0.6504503492370626, 0.4781090907569663, 0.4509756855996736, 0.47350738103691015, 0.46870808214794213, 0.4582021959231857], [0.5432614156979767, 0.49394779034094816, 0.4713299935938986, 0.48097192657760424, 0.4605696385301532, 0.4694113970330962], [0.5687825646663702, 0.5183770236511562, 0.46720935254442697, 0.47847954695028294, 0.45245572850503013, 0.459360686799148], [0.5274215826241604, 0.5008543651192796, 0.48574538013997426, 0.4531371125516226, 0.4594597773355167, 0.4609970960669414], [0.6180145855843482, 0.49218441294978543, 0.4661290845708832, 0.4625430966122379, 0.4507007750177472, 0.4543416043251822], [0.5586949532014067, 0.4370918639012915, 0.45389595325529086, 0.44859193788346774, 0.4654958832279297, 0.4549441190441712], [0.5403133818194372, 0.4853057953674873, 0.4453285924350894, 0.47496973689569444, 0.46091495585185305, 0.46148139039461666], [0.5859733639432232, 0.48389489568171795, 0.4566447407139069, 0.4736711000387559, 0.46265947710392474, 0.45961865797836765], [0.5614494330737685, 0.511287686832131, 0.4917283382550095, 0.44980205094910364, 0.4644064780344115, 0.46633470628727425]]
mastUtilCIstore_numimpdraws = [[(0.5129468501476397, 0.5620449226610971), (0.46333854027013155, 0.48767036592092383), (0.4559311476470054, 0.4750400181972134), (0.45320198038746495, 0.4744628503437771), (0.4544974124320009, 0.47131054564481145), (0.45461642462091456, 0.470107596150807)], [(0.6212796780345387, 0.6796210204395865), (0.46709288018045614, 0.4891253013334764), (0.438142654294996, 0.4638087169043512), (0.46530073638550196, 0.48171402568831834), (0.46039928951448283, 0.4770168747814014), (0.4493517422556925, 0.4670526495906788)], [(0.5141271226759532, 0.5723957087200002), (0.4819438720781921, 0.5059517086037042), (0.4616201375566966, 0.4810398496311006), (0.4730842676550706, 0.48885958550013786), (0.4518051139623007, 0.4693341630980057), (0.46224628078178176, 0.47657651328441064)], [(0.5430704439464225, 0.594494685386318), (0.5049431331689704, 0.531810914133342), (0.4569668810604033, 0.47745182402845066), (0.4703920828215553, 0.48656701107901057), (0.4434789626922466, 0.46143249431781364), (0.4511610986588783, 0.46756027493941765)], [(0.49755719802463005, 0.5572859672236907), (0.4894178229191777, 0.5122909073193815), (0.4767415514388289, 0.4947492088411196), (0.444363261571453, 0.4619109635317922), (0.450684337684101, 0.4682352169869324), (0.452684071318493, 0.4693101208153898)], [(0.587964417498382, 0.6480647536703144), (0.4804436448167091, 0.5039251810828618), (0.45471298322670783, 0.47754518591505857), (0.45295721332165195, 0.4721289799028239), (0.4411098014866055, 0.4602917485488889), (0.44575959427511824, 0.4629236143752462)], [(0.5349653199671045, 0.5824245864357089), (0.4227686731207769, 0.4514150546818061), (0.442276249261111, 0.46551565724947075), (0.43980544038837044, 0.45737843537856504), (0.4572614546237501, 0.47373031183210923), (0.44787294588616655, 0.46201529220217585)], [(0.5108393237230509, 0.5697874399158236), (0.4727107293569399, 0.4979008613780347), (0.4333328773049181, 0.45732430756526066), (0.4653206971407311, 0.4846187766506578), (0.45159428763011933, 0.47023562407358677), (0.45373840569845214, 0.4692243750907812)], [(0.5599576158830395, 0.6119891120034069), (0.47211081895989215, 0.49567897240354375), (0.44711413922503196, 0.4661753422027819), (0.4649726007502135, 0.4823695993272983), (0.4542073213650224, 0.4711116328428271), (0.45066038566047073, 0.46857693029626457)], [(0.5387710221650308, 0.5841278439825062), (0.4983225667392328, 0.5242528069250292), (0.4822360209653822, 0.5012206555446368), (0.44061145019558134, 0.45899265170262593), (0.4551261907396662, 0.47368676532915677), (0.45826269752572335, 0.47440671504882514)]]
'''
for tempind, currutilavgstore in enumerate(mastUtilavgstore_numimpdraws):
    utilerrs = np.array([mastUtilCIstore_numimpdraws[tempind][i][1] - currutilavgstore[i] for i in range(len(numimportdrawslist))])
    plt.errorbar(numimportdrawslist, currutilavgstore, yerr=utilerrs, ms=15, mew=1, mec='black', capsize=2)
plt.hlines(0.46, xmin=0,xmax=max(numimportdrawslist)*1.05, color='black',linestyles='dotted')
plt.ylim([0, 1.1])
plt.xlim([0, max(numimportdrawslist)*1.05])
plt.title('Utility estimates vs. size of MCMC importance set\nOutlier removal of 1%, for 10 replications')
plt.xlabel('Size of MCMC importance set')
plt.ylabel('Utility estimate')
plt.show()
plt.close()


numdatadrawslist = [50,100,200,500,1000]

mastUtilavgstore_numdatadraws, mastUtilCIstore_numdatadraws = [], []
for rep in range(10): # 10 reps
    print('Rep:', rep)
    utilavgstore, utilCIstore = [], []
    for numdatadraws in numdatadrawslist:
        datadraws = paramdict['truthdraws'][choice(np.arange(numtruthdraws), size=numdatadraws, replace=False)]
        paramdict.update({'datadraws': datadraws})
        currutilavg, (currutilCIlo, currutilCIhi) = getImportanceUtilityEstimate2(n, lgdict, paramdict,
                                                        numimportdraws=30000,
                                                        numdatadrawsforimportance=1000,
                                                        impweightoutlierprop=0.005)
        print('Utility:', currutilavg)
        print('Range:', (currutilCIlo, currutilCIhi))
        utilavgstore.append(currutilavg)
        utilCIstore.append((currutilCIlo, currutilCIhi))
    mastUtilavgstore_numdatadraws.append(utilavgstore)
    mastUtilCIstore_numdatadraws.append(utilCIstore)

'''31-MAR
mastUtilavgstore_numdatadraws = [[0.45636308888535604, 0.4540723558280857, 0.44209683014838896, 0.4374356693001289, 0.4577147337291727], [0.459762193732693, 0.4397658914467062, 0.45433763863268695, 0.45512363569965153, 0.45114974781205586], [0.446747651635504, 0.45540003023968234, 0.44476759429565593, 0.44885205342933165, 0.4411466459152473], [0.46646644727859643, 0.4438885293113306, 0.45751158703022154, 0.4489346559135532, 0.4629610411570164], [0.46946928378760244, 0.44713674585383467, 0.46569369471344846, 0.4620539094872096, 0.44072758984469207], [0.44564752992363044, 0.46350137834879224, 0.4588760662604976, 0.4508713275624756, 0.4580450686346942], [0.4300386531901834, 0.4656836897254064, 0.45061542111388064, 0.4639694679301565, 0.46618444556051264], [0.4223850511075975, 0.47691721228884587, 0.44642161939223657, 0.4542935241895072, 0.46036181313698643], [0.45732179161364783, 0.43890006312096963, 0.4691839191185818, 0.45589398355207544, 0.45065779610631385], [0.45657922919706007, 0.45867196272858957, 0.4575675641750774, 0.4640386776111196, 0.4681301456627245]]
mastUtilCIstore_numdatadraws = [[(0.44263744212565737, 0.4700887356450547), (0.4448653693987539, 0.4632793422574175), (0.43507745080257365, 0.4491162094942043), (0.4326797752675642, 0.4421915633326936), (0.45457499322199757, 0.46085447423634784)], [(0.446596186974773, 0.472928200490613), (0.4294886691510138, 0.4500431137423986), (0.4481524031971973, 0.4605228740681766), (0.4515316411792467, 0.45871563022005635), (0.44799958244881033, 0.4542999131753014)], [(0.43157259604010045, 0.46192270723090756), (0.4448304692305536, 0.4659695912488111), (0.4378214597525054, 0.45171372883880645), (0.44506507242816795, 0.45263903443049536), (0.43779347439008154, 0.4444998174404131)], [(0.4509828399726148, 0.48195005458457807), (0.4345407538456443, 0.45323630477701693), (0.45064989083922447, 0.4643732832212186), (0.444351085675053, 0.4535182261520534), (0.4599905930416024, 0.46593148927243044)], [(0.454629084537431, 0.4843094830377739), (0.4379482664415759, 0.4563252252660934), (0.458933729526549, 0.47245365990034793), (0.45783120069517746, 0.4662766182792417), (0.43758967836861107, 0.44386550132077307)], [(0.4325763104255298, 0.4587187494217311), (0.4541112648489185, 0.47289149184866597), (0.4521520623329791, 0.4656000701880161), (0.4465897688469842, 0.455152886277967), (0.4551075362231116, 0.46098260104627675)], [(0.4140567618268314, 0.44602054455353546), (0.45625956938891044, 0.4751078100619024), (0.44363135744239823, 0.45759948478536305), (0.45962544802858263, 0.4683134878317303), (0.4631880556702268, 0.4691808354507985)], [(0.40691266519771707, 0.4378574370174779), (0.466370059871295, 0.48746436470639676), (0.440418450821348, 0.45242478796312513), (0.4498146585883722, 0.45877238979064217), (0.45718253328299063, 0.46354109299098223)], [(0.4449075024084377, 0.469736080818858), (0.4294948137350181, 0.4483053125069212), (0.4622258519313105, 0.47614198630585314), (0.4518691990492165, 0.45991876805493437), (0.447788656261606, 0.45352693595102167)], [(0.4445320094757452, 0.4686264489183749), (0.44913507982232437, 0.46820884563485476), (0.4509999983283728, 0.46413513002178197), (0.4599379536277812, 0.468139401594458), (0.4653383138562792, 0.47092197746916975)]]
'''
for tempind, currutilavgstore in enumerate(mastUtilavgstore_numdatadraws):
    utilerrs = np.array([mastUtilCIstore_numdatadraws[tempind][i][1] - currutilavgstore[i] for i in range(len(numdatadrawslist))])
    plt.errorbar(numdatadrawslist, currutilavgstore, yerr=utilerrs, ms=15, mew=1, mec='black', capsize=2)
plt.hlines(0.46, xmin=0,xmax=max(numdatadrawslist)*1.05, color='black',linestyles='dotted')
plt.ylim([0.3, 0.55])
plt.xlim([0, max(numdatadrawslist)*1.05])
plt.title('Utility estimates vs. number of data draws\nOutlier removal of 1% and 30k importance draws, for 10 replications')
plt.xlabel('Number of data draws')
plt.ylabel('Utility estimate')
plt.show()
plt.close()