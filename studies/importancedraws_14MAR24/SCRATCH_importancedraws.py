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
    dept_df = pd.read_csv('operationalizedsamplingplans/senegal_csv_files/deptfixedcosts.csv', header=0)
    regcost_mat = pd.read_csv('operationalizedsamplingplans/senegal_csv_files/regarcfixedcosts.csv', header=None)
    regNames = ['Dakar', 'Diourbel', 'Fatick', 'Kaffrine', 'Kaolack', 'Kedougou', 'Kolda', 'Louga', 'Matam',
                'Saint-Louis', 'Sedhiou', 'Tambacounda', 'Thies', 'Ziguinchor']
    # Get testing results
    testresults_df = pd.read_csv('operationalizedsamplingplans/senegal_csv_files/dataresults.csv', header=0)
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
lgdict['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}

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
def importance_method_loss_list2(design, numtests, priordatadict, paramdict, numdatadrawsforimportance, numimportdraws):
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
    #Wimport = sampf.build_weights_matrix(importancedraws, paramdict['datadraws'], sampMat, priordatadict)


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
    Vimport_norm = Vimport / np.sum(Vimport)
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
    Uimport_norm = Uimport / np.sum(Uimport)

    VoverU = (Vimport / Uimport)
    VoverU_norm = VoverU * numimportdraws / np.sum(VoverU)

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
        tempwtarray = Wimport[:, j] * VoverU * numimportdraws / np.sum(Wimport[:, j] * VoverU)
        # Remove inds for top 1% of weights
        tempremoveinds = np.where(tempwtarray>np.quantile(tempwtarray, 0.99))
        tempwtarray = np.delete(tempwtarray, tempremoveinds)
        tempwtarray = tempwtarray/np.sum(tempwtarray)
        tempimportancedraws = np.delete(importancedraws, tempremoveinds, axis=0)
        tempRimport = np.delete(Rimport, tempremoveinds, axis=0)
        est = sampf.bayesest_critratio(tempimportancedraws, tempwtarray, q)
        minslist.append(sampf.cand_obj_val(est, tempimportancedraws,
                                           tempwtarray,
                                           #Wimport[:, j]*Vimport_norm / np.sum(Wimport[:, j]*Vimport_norm),
                                           paramdict, tempRimport))
    plt.plot(minslist)
    plt.show()
    plt.close()

    #print(paramdict['baseloss']-np.average(minslist))

    return minslist

def getImportanceUtilityEstimate2(n, lgdict, paramdict, numdatadrawsforimportance, numimportdraws, zlevel=0.95):
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

