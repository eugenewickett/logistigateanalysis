'''Scratch file for dealing with IDENTIFIABLE Ghana data
 Data should be organized and anonymized here prior to moving to 'bayesianexperimentaldesign'
 '''

from logistigate.logistigate import utilities as util # Pull from the submodule "develop" branch
from logistigate.logistigate import methods
from logistigate.logistigate import lg
import numpy as np
from numpy.random import choice
import scipy.special as sps
import scipy.stats as spstat
import matplotlib
import matplotlib.pyplot as plt
import random
import os
import pickle
import pandas as pd

# New prior class, which enables different prior scales at different nodes
class prior_laplace_assort:
    """
    Defines the class instance of an assortment of Laplace priors, with associated mu (mean)
    and scale vectors in the logit-transfomed [0,1] range, and the following methods:
        rand: generate random draws from the distribution
        lpdf: log-likelihood of a given vector
        lpdf_jac: Jacobian of the log-likelihood at the given vector
        lpdf_hess: Hessian of the log-likelihood at the given vector
    beta inputs may be a Numpy array of vectors
    """
    def __init__(self, mu=sps.logit(0.1), scale=np.sqrt(5/2)):
        self.mu = np.array(mu)
        self.scale = np.array(scale)
    def rand(self, n=1):
        retList = [np.random.laplace]
        return np.random.laplace(self.mu, self.scale, n)
    def expitrand(self, n=1): # transformed to [0,1] space
        return sps.expit(np.random.laplace(self.mu, self.scale, n))
    def lpdf(self,beta):
        if beta.ndim == 1: # reshape to 2d
            beta = np.reshape(beta,(1,-1))
        lik = np.log(1/(2*self.scale)) - np.sum(np.abs(beta - self.mu)/self.scale,axis=1)
        return np.squeeze(lik)
    def lpdf_jac(self,beta):
        if beta.ndim == 1: # reshape to 2d
            beta = np.reshape(beta,(1,-1))
        jac = - (1/self.scale)*np.squeeze(1*(beta>=self.mu) - 1*(beta<=self.mu))
        return np.squeeze(jac)
    def lpdf_hess(self,beta):
        if beta.ndim == 1: # reshape to 2d
            beta = np.reshape(beta,(1,-1))
        k,n = len(beta[:,0]),len(beta[0])
        hess = np.tile(np.zeros(shape=(n,n)),(k,1,1))
        return np.squeeze(hess)

def GhanaPriorScratch():
    '''
    RISK CATEGORIES:
    Extremely Low Risk
    Very Low Risk
    Low Risk
    Moderate Risk
    Moderately High Risk
    High Risk
    Very High Risk
    '''

    # Anchor averages
    avgArr1 = np.array([0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25])
    numCat = len(avgArr1)
    var1 = 4.
    # Move to logit space
    avgLogitArr1 = sps.logit(avgArr1)

    probsList = []
    for catInd in range(numCat):
        priorobj = methods.prior_laplace(mu=avgLogitArr1[catInd],scale=np.sqrt(var1/2))
        xrange = np.arange(0.001,0.6,0.001)
        xrangeLogit = sps.logit(xrange)
        probs = np.exp(np.array([ priorobj.lpdf(np.array([xrangeLogit[i]])).tolist() for i in range(xrangeLogit.shape[0])]))
        probsList.append(probs)

    # Now plot density lines
    for currListInd, currList in enumerate(probsList):
        plt.plot(xrange,currList,label=str(avgArr1[currListInd]))
    plt.title('Using Laplace priors; variance of '+str(var1))
    plt.xlabel('SFP rate')
    plt.ylabel('Density')
    plt.show()

    # Do again with normal priors
    avgArr1 = np.array([0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25])
    numCat = len(avgArr1)
    var1 = 4.
    # Move to logit space
    avgLogitArr1 = sps.logit(avgArr1)

    probsList = []
    for catInd in range(numCat):
        priorobj = methods.prior_normal(mu=avgLogitArr1[catInd], var=var1) # normal priors
        xrange = np.arange(0.001, 0.6, 0.001)
        xrangeLogit = sps.logit(xrange)
        probs = np.exp(
            np.array([priorobj.lpdf(np.array([xrangeLogit[i]])).tolist() for i in range(xrangeLogit.shape[0])]))
        probsList.append(probs)

    # Now plot density lines
    for currListInd, currList in enumerate(probsList):
        plt.plot(xrange, currList, label=str(avgArr1[currListInd]))
    plt.title('Using Normal priors; variance of '+str(var1))
    plt.xlabel('SFP rate')
    plt.ylabel('Density')
    plt.show()


    return

def GhanaInference():
    '''Script for analyzing 2022 Ghana data'''

    SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
    filesPath = os.path.join(SCRIPT_DIR, 'uspfiles')
    # DONT use facility for test node
    #GHA_df1 = pd.read_csv(os.path.join(filesPath, 'FACILID_MNFR_MCH.csv'), low_memory=False) # Facilities as test nodes
    #GHAlist_FCLY = GHA_df1.values.tolist()
    #GHA_df2 = pd.read_csv(os.path.join(filesPath, 'CITY_MNFR_MCH.csv'), low_memory=False) # Cities as test nodes
    #GHAlist_CITY = GHA_df2.values.tolist()
    GHA_df3= pd.read_csv(os.path.join(filesPath, 'PROV_MNFR_MCH.csv'), low_memory=False) # Provinces as test nodes
    GHAlist_PROV = GHA_df3.values.tolist()
    GHA_df4 = pd.read_csv(os.path.join(filesPath, 'PROVADJ_MNFR_MCH.csv'), low_memory=False)  # Provinces as test nodes
    GHAlist_PROVADJ = GHA_df4.values.tolist()

    '''
    31 observed CITIES (DISTRICTS)
    4 observed PROVINCES (REGIONS)
    13 observed MANUFACTURERS
    35% SFP rate (62 of 177 tests)
    
    Total Ghana specifications (from Wikipedia, Jan 2023):
    261 DISTRICTS (called metropolitan, municipal and district assemblies, MMDAs)
    16 REGIONS (# MMDAs):
        Ahafo (6)
        Ashanti (43)
        Bono (12)
        Bono East (11)
        Central (22)
        Eastern (33)
        Greater Accra (29)
        Northern (16)
        North East (6)
        Oti (9)
        Savannah (7)
        Upper East (15)
        Upper West (11)
        Volta (18)
        Western (14)
        Western North (9)
        
    The region of Brong-Ahafo was split into Bono, Bono East, and Ahafo in 2018.
    The five regions contained in the data set (Ahafo, Ashanti, Bono, Central, Eastern) are geographically
        adjacent.
    One idea for an additional subset of regions to include, which are close to the original set: 
        Western; Western North; Greater
    '''
    # Change Brong-Ahafo into its new regions
    #len([i for i in GHAlist_PROV if i[0]=='BRONG AHAFO']) # 17 TOTAL
    for i in GHAlist_PROV:
        if i[0] == 'BRONG AHAFO':
            print(i)

    # Set MCMC parameters
    numPostSamps = 1000
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    # Establish a prior
    priorMean = -2.5
    priorVar = 3.5

    # Create a logistigate dictionary and conduct inference
    '''
    # FACILITIES
    lgDict = util.testresultsfiletotable(GHAlist_FCLY, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum())) # Check that everything is in line with expectations
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nGhana - Facility ID/Manufacturer Analysis', '\nGhana - Facility ID/Manufacturer Analysis'])
    # Now print 90% intervals and write intervals to file
    # Supply nodes first
    import csv
    outputFileName = os.path.join(filesPath, 'SNintervals_FCLY.csv')
    f = open(outputFileName, 'w', newline='')
    writer = csv.writer(f)

    writelist = []
    for i in range(lgDict['N'].shape[1]):
        interval = [np.quantile(lgDict['postSamples'][:,i], 0.05),np.quantile(lgDict['postSamples'][:,i], 0.95)]
        name = lgDict['importerNames'][i]
        writelist.append([name, interval[0], interval[1]])
        writer.writerow([name, interval[0], interval[1]])
        print(name + ': ' + str(interval))
    writelist = []
    numSN = lgDict['N'].shape[1]
    for i in range(lgDict['N'].shape[0]):
        interval = [np.quantile(lgDict['postSamples'][:,numSN + i], 0.05), np.quantile(lgDict['postSamples'][:,numSN + i], 0.95)]
        name = lgDict['outletNames'][i]
        writelist.append([name, interval[0], interval[1]])
        writer.writerow([name, interval[0], interval[1]])
        print(name+': '+str(interval))

    f.close()
    '''

    '''
    # CITIES/DISTRICTS
    lgDict = util.testresultsfiletotable(GHAlist_CITY, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))  # Check that everything is in line with expectations
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nGhana - City/Manufacturer Analysis', '\nGhana - City/Manufacturer Analysis'])
    # Now print 90% intervals
    import csv
    outputFileName = os.path.join(filesPath, 'SNintervals_CITY.csv')
    f = open(outputFileName, 'w', newline='')
    writer = csv.writer(f)

    # Supply nodes first
    writelist = []
    for i in range(lgDict['N'].shape[1]):
        interval = [np.quantile(lgDict['postSamples'][:, i], 0.05), np.quantile(lgDict['postSamples'][:, i], 0.95)]
        name = lgDict['importerNames'][i]
        writelist.append([name, interval[0], interval[1]])
        writer.writerow([name, interval[0], interval[1]])
        print(name + ': ' + str(interval))
    # outputFileName = os.path.join(filesPath, 'SNintervals_FCLY.csv')
    writelist = []
    numSN = lgDict['N'].shape[1]
    for i in range(lgDict['N'].shape[0]):
        interval = [np.quantile(lgDict['postSamples'][:, numSN + i], 0.05),
                    np.quantile(lgDict['postSamples'][:, numSN + i], 0.95)]
        name = lgDict['outletNames'][i]
        writelist.append([name, interval[0], interval[1]])
        writer.writerow([name, interval[0], interval[1]])
        print(name + ': ' + str(interval))

    f.close()
    '''

    # PROVINCES
    lgDict = util.testresultsfiletotable(GHAlist_PROV, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))  # Check that everything is in line with expectations
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nGhana - Province/Manufacturer Analysis', '\nGhana - Province/Manufacturer Analysis'])
    # Now print 90% intervals
    import csv
    outputFileName = os.path.join(filesPath, 'SNintervals_PROV.csv')
    f = open(outputFileName, 'w', newline='')
    writer = csv.writer(f)

    # Supply nodes first
    writelist = []
    for i in range(lgDict['N'].shape[1]):
        interval = [np.quantile(lgDict['postSamples'][:, i], 0.05), np.quantile(lgDict['postSamples'][:, i], 0.95)]
        name = lgDict['importerNames'][i]
        writelist.append([name, interval[0], interval[1]])
        writer.writerow([name, interval[0], interval[1]])
        print(name + ': ' + str(interval))
    # outputFileName = os.path.join(filesPath, 'SNintervals_FCLY.csv')
    writelist = []
    numSN = lgDict['N'].shape[1]
    for i in range(lgDict['N'].shape[0]):
        interval = [np.quantile(lgDict['postSamples'][:, numSN + i], 0.05),
                    np.quantile(lgDict['postSamples'][:, numSN + i], 0.95)]
        name = lgDict['outletNames'][i]
        writelist.append([name, interval[0], interval[1]])
        writer.writerow([name, interval[0], interval[1]])
        print(name + ': ' + str(interval))
        ###### WRITE TO CSV
    f.close()

    return

def GhanaSamplingDesignAnalysis():
    '''Script for analyzing 2022 Ghana data and utilities for possible sampling designs'''

    SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
    filesPath = os.path.join(SCRIPT_DIR, 'uspfiles')
    GHA_df1 = pd.read_csv(os.path.join(filesPath, 'FACILID_MNFR.csv'), low_memory=False) # Facilities as test nodes
    GHAlist_FCLY = GHA_df1.values.tolist()
    GHA_df2 = pd.read_csv(os.path.join(filesPath, 'CITY_MNFR.csv'), low_memory=False) # Cities as test nodes
    GHAlist_CITY = GHA_df2.values.tolist()
    GHA_df3= pd.read_csv(os.path.join(filesPath, 'PROV_MNFR.csv'), low_memory=False) # Provinces as test nodes
    GHAlist_PROV = GHA_df3.values.tolist()

    '''
    283 observed FACILITIES
    34 observed CITIES
    5 observed PROVINCES
    41 observed MANUFACTURERS
    16.7% SFP rate (63 of 378 tests)
    '''
    # Set MCMC parameters
    numPostSamps = 10000
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    # Establish a prior
    priorMean = -2.5 #average of 7%
    priorVar = 3.5

    # Create a logistigate dictionary and conduct inference
    # PROVINCES
    lgDict = util.testresultsfiletotable(GHAlist_PROV, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))  # Check that everything is in line with expectations
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nGhana - Province/Manufacturer Analysis', '\nGhana - Province/Manufacturer Analysis'])

    # Now consider some sampling designs
    print(np.sum(lgDict['N'],axis=0)) # Across manufacturers
    print(np.sum(lgDict['N'], axis=1))  # Across provinces
    lgDict.update({'numPostSamples': numPostSamps})
    # Sourcing patterns seem close enough that we will assume each province has the same Q row
    #   Use the sum for each manufacturer and normalize
    Qrow = np.sum(lgDict['N'], axis=0)/np.sum(lgDict['N'])
    tnNum, snNum = lgDict['outletNum'], lgDict['importerNum']
    Q = np.tile(Qrow,(tnNum,1))
    # Now we want to consider some node-sampling designs; we will randomly generate some
    numDesigns = 5
    designMat = np.empty((numDesigns,tnNum))
    np.random.seed(23)
    for desInd in range(numDesigns):
        randVec = np.random.exponential(1,size=tnNum)
        designMat[desInd] = randVec/np.sum(randVec)
    # print(designMat)
    # Now add some simpler designs
    for tnInd in range(tnNum):
        newVec = np.zeros(tnNum)
        newVec[tnInd] = 1.0
        designMat = np.vstack([designMat,newVec])

    # Get design utilities; store utility vectors and times
    import time
    underWt, t = 1., 0.1
    scoredict = {'name': 'AbsDiff', 'underEstWt': underWt}
    riskdict = {'threshold': t}
    marketvec = np.ones(tnNum+snNum)
    lossDict = {'scoreFunc': score_diff, 'scoreDict': scoredict,
                'riskFunc': risk_parabolic, 'riskDict': riskdict,
                'marketVec': marketvec}
    #designNames = ['Design '+str(i+1) for i in range(len(designMat))]
    numtests = 100
    omeganum = 100
    random.seed(35)
    randinds = random.sample(range(len(lgDict['postSamples'])), 100)

    masterTime = []
    masterUtil = []
    for desind, currDesign in enumerate(designMat[1:2]):
        currDesignName = 'Design '+str(desind)
        currStart = time.time()
        currutilvec = getDesignUtility(priordatadict=lgDict.copy(), lossdict=lossDict.copy(),
                                designlist=[currDesign], designnames=[currDesignName],
                                numtests=numtests, omeganum=omeganum,
                                type=['test node', Q], randinds=randinds)
        currTime = time.time() - currStart
        masterTime.append(currTime)
        masterUtil.append(currutilvec)
    print((masterTime[1]))
    print(masterUtil[1])

    '''
    omeganum=100, numtests=100
    underWt, t = 1., 0.1
    scoredict = {'name': 'AbsDiff', 'underEstWt': underWt}
    riskdict = {'threshold': t}
    marketvec = np.ones(tnNum+snNum)
    
    DESIGN 0:   [0.12785163, 0.51550276, 0.25454771, 0.05824892, 0.04384899]
    TIME:       [14514.11165857315]
    UTILVEC:    [[[1.9445849074861141, 1.8981042457450334, 1.9278206725997782, 1.9783444757804893, 1.93367271487467, 2.0121193570246336, 1.9203177267572298, 1.9997250983296744, 1.9176410149979568, 2.018567065379746, 1.9734506666273548, 1.9503532808612751, 1.9072012319850034, 1.9410412313528442, 1.9941349346034243, 1.937098020532463, 1.9156807904066757, 1.9980383972972493, 1.961196424707721, 1.9323393170419811, 1.934203857324445, 2.0117624361232083, 1.9142761673156445, 1.9910879860843307, 1.9907440109419499, 1.9781847987102326, 1.9597140055432414, 0.49526301007617146, 1.975200206967608, 1.9530973218257655, 1.9299568905728086, 2.0093962373363046, 1.8978292253728988, 1.9959969168099236, 2.012235011739977, 1.9907228558141792, 1.916420921794944, 1.9014952868742523, 2.0041728943239048, 2.008404903707848, 1.9830236798473648, 1.926776077073508, 1.9476726644392601, 1.9479617948251606, 1.8715105658752584, 2.015634381914563, 1.874608897833401, 1.9746797209994882, 1.9655174287993533, 1.9917390711321274, 1.9956133465400478, 1.9425656398955216, 1.9710552951934042, 1.844631050131446, 1.9751155035396937, 1.9294182310973316, 1.9529839238559987, 1.9539651454865612, 1.9727017053065723, 1.9425421332606438, 1.99547945737834, 1.987288125163441, 1.9485735118081258, 2.0114952217846827, 2.072736765048787, 1.9120198388658391, 1.9365356812014736, 1.9358396576648234, 1.9495102270047864, 1.9701029637146183, 1.9539608369067747, 1.9788727244437927, 1.929275677105473, 1.9477213465099732, 1.9697192348374708, 1.9656267404110466, 1.9968800633987263, 1.9411978494522515, 2.016591143285792, 1.8661187034350004, 1.9472129083286684, 1.8773238137449222, 1.9690192320533388, 1.9460661571514, 0.0, 1.961774355520726, 2.012455067072379, 2.90527161332299, 1.9296367454516175, 1.912345460462314, 2.001650370564251, 2.067602246639982, 1.9968980181225637, 1.9711203389616017, 2.0680500095146086, 1.9360107883433644, 1.9687419117792087, 1.9586794091381126, 1.949776878640305, 1.9450544536262404]]]
    
    DESIGN 1:   [0.34768811, 0.05486145, 0.1494784 , 0.28871425, 0.15925779]
    TIME:       45991
    UTILVEC:    [[1.9208918378630555, 1.918638950391999, 1.9316003994113962, 1.9766211051897609, 2.006841959974357, 1.9485046708155203, 1.9419170655643965, 2.114580913842551, 2.0121377370807316, 2.0467675792113953, 1.9390330747600477, 2.0097170855877935, 1.9941305197747685, 1.942155425220364, 1.9423408202906143, 1.9297700703795395, 2.046081326072883, 1.8729502618683436, 2.0215359122732557, 1.9631978462151307, 1.9853049803759124, 2.00464233524333, 1.952839466740082, 1.9942946859751784, 2.019278522743033, 1.952170768396543, 1.8381248994130537, 1.9600708809060674, 1.9918253815296008, 1.9563476429083833, 1.9812813702688388, 1.9434777005449275, 2.019864419266891, 1.9674752923576544, 1.9769388748576333, 2.014995556946697, 1.9935761806858958, 2.0058326555528576, 1.9782813554203484, 1.977519053833195, 1.949563062818911, 2.0803637850130388, 4.4305205871688225, 0.0, 1.9296430346931261, 1.9852435105769033, 1.9138177685002555, 1.9894445362516548, 1.9947794779500352, 2.018272480274653, 1.9225345692733764, 1.9474135321586215, 1.878731092839133, 3.9348834787940237, 1.9711187908624215, 2.033852794371972, 1.9428722591994745, 2.0086090207841187, 1.9585438915980986, 2.0213443106435736, 2.0608514273126017, 1.9795401272404527, 1.9125105550323824, 2.041621843427458, 1.9421519336716189, 1.9616218933341079, 1.9518133679019338, 1.996298592541785, 2.013234967456426, 1.9719412781982162, 2.009251506338606, 1.9689683246066783, 2.0697737443157522, 2.088343092538656, 1.9448758370718608, 1.9335161008301345, 1.9574229339131106, 2.0954004860749955, 1.9858672140778781, 1.98287851948484, 1.9575073998640935, 1.9038076447425352, 2.0488263843852472, 1.8918442465566954, 1.9648315121968245, 1.9413555515729917, 2.0881736614617754, 0.0, 1.9996183085727655, 1.9282516803840422, 1.925405023223576, 1.9864271175846855, 1.9193225138297247, 1.9629580191840226, 2.012462598752095, 2.060479054413587, 2.0580094993486924, 2.0111569081543292, 2.069144262477017, 1.891761034307396]]
    
    
    
    '''


    ########################################################
    ########################################################
    ########################################################
    # CITIES
    lgDict = util.testresultsfiletotable(GHAlist_CITY, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))  # Check that everything is in line with expectations
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90',
                         subTitleStr=['\nGhana - City/Manufacturer Analysis', '\nGhana - City/Manufacturer Analysis'])

    return


def balancedesign(N,ntilde):
    '''
    Uses matrix of original batch (N) and next batch (ntilde) to return a balanced design where the target is an even
    number of tests from each (TN,SN) arc for the total tests done
    '''
    n = np.sum(N)
    r,c = N.shape
    D = np.repeat(1/(r*c),r*c)*(n+ntilde)
    D.shape = (r,c)
    D = D - N
    D[D < 0] = 0.
    D = D/np.sum(D)

    return D

def roundDesignLow(D,n):
    '''
    Takes a proposed design, D, and number of new tests, n, to produce an integer tests array by removing tests from
    design traces with the highest number of tests or adding tests to traces with the lowest number of tests.
    '''
    roundMat = np.round(n*D)
    if np.sum(roundMat) > n: # Too many tests; remove from highest represented traces
        roundMat = roundMat.flatten()
        sortinds = np.argsort(-roundMat,axis=None).tolist()
        for removeInd in range(int(np.sum(roundMat)-n)):
            roundMat[sortinds[removeInd]] += -1
        if roundMat.ndim==2:
            roundMat = roundMat.reshape(D.shape[0],D.shape[1])
    elif np.sum(roundMat) < n: # Too few tests; add to lowest represented traces
        roundMat = roundMat.flatten()
        sortinds = np.argsort(roundMat, axis=None).tolist()
        for addind in range(int(n-np.sum(roundMat))):
            roundMat[sortinds[addind]] += 1
        if roundMat.ndim == 2:
            roundMat = roundMat.reshape(D.shape[0],D.shape[1])
    return roundMat

def roundDesignHigh(D,n):
    '''
    Takes a proposed design, D, and number of new tests, n, to produce an integer tests array by removing tests from
    design traces with the lowest number of tests or adding tests to traces with the highest number of tests.
    '''
    roundMat = np.round(n*D)
    if np.sum(roundMat) > n: # Too many tests; remove from lowest represented traces
        roundMat = roundMat.flatten()
        sortinds = np.argsort(roundMat,axis=None).tolist()
        for removeInd in range(int(np.sum(roundMat)-n)):
            roundMat[sortinds[removeInd]] += -1
        if roundMat.ndim == 2:
            roundMat = roundMat.reshape(D.shape[0],D.shape[1])
    elif np.sum(roundMat) < n: # Too few tests; add to highest represented traces
        roundMat = roundMat.flatten()
        sortinds = np.argsort(-roundMat, axis=None).tolist()
        for addind in range(int(n-np.sum(roundMat))):
            roundMat[sortinds[addind]] += 1
        if roundMat.ndim == 2:
            roundMat = roundMat.reshape(D.shape[0],D.shape[1])
    return roundMat

def plotLossVecs(lveclist, lvecnames=[], type='CI', CIalpha = 0.05,legendlabel=[],
                 plottitle='Confidence Intervals for Loss Averages', plotlim=[]):
    '''
    Takes a list of loss vectors and produces either a series of histograms or a single plot marking average confidence
    intervals
    lveclist: list of lists
    type: 'CI' (default) or 'hist'
    CIalpha: alpha for confidence intervals
    '''
    numvecs = len(lveclist)
    # Make dummy names if none entered
    if lvecnames==[]: #empty
        for ind in range(numvecs):
            lvecnames.append('Design '+str(ind+1))
    numDups = 1
    orignamelen = len(lvecnames)
    if orignamelen<len(lveclist): # We have multiple entries per design
        numDups = int(len(lveclist)/len(lvecnames))
        lvecnames = numDups*lvecnames
    # For color palette
    from matplotlib.pyplot import cm

    # Make designated plot type
    if type=='CI':
        lossavgs = []
        lossint_hi = []
        lossint_lo = []
        for lvec in lveclist:
            currN = len(lvec)
            curravg = np.average(lvec)
            lossavgs.append(curravg)
            std = np.std(lvec)
            z = spstat.norm.ppf(1 - (CIalpha / 2))
            intval = z * (std) / np.sqrt(currN)
            lossint_hi.append(curravg + intval)
            lossint_lo.append(curravg - intval)

        # Plot intervals for loss averages
        if lossavgs[0]>0: # We have losses
            xaxislab = 'Loss'
            limmin = 0
            limmax = max(lossint_hi)*1.1
        elif lossavgs[0]<0: # We have utilities
            xaxislab = 'Utility'
            limmin = min(lossint_lo)*1.1
            limmax = 0
        fig, ax = plt.subplots(figsize=(7,7))
        #color = iter(cm.rainbow(np.linspace(0, 1, numvecs/numDups)))
        #for ind in range(numvecs):

        if plotlim==[]:
            plt.xlim([limmin,limmax])
        else:
            plt.xlim(plotlim)
        for ind in range(numvecs):
            if np.mod(ind,orignamelen)==0:
                color = iter(cm.rainbow(np.linspace(0, 1, int(numvecs / numDups))))
            currcolor = next(color)
            if ind<orignamelen:

                plt.plot(lossavgs[ind], lvecnames[ind], 'D', color=currcolor, markersize=6)
            elif ind>=orignamelen and ind<2*orignamelen:
                plt.plot(lossavgs[ind], lvecnames[ind], 'v', color=currcolor, markersize=8)
            elif ind>=2*orignamelen and ind<3*orignamelen:
                plt.plot(lossavgs[ind], lvecnames[ind], 'o', color=currcolor, markersize=6)
            else:
                plt.plot(lossavgs[ind], lvecnames[ind], '^', color=currcolor, markersize=8)
            line = ax.add_line(matplotlib.lines.Line2D(
                 (lossint_hi[ind], lossint_lo[ind]),(lvecnames[ind], lvecnames[ind])))
            line.set(color=currcolor)
            anno_args = {'ha': 'center', 'va': 'center', 'size': 12, 'color': currcolor }
            _ = ax.annotate("|", xy=(lossint_hi[ind], lvecnames[ind]), **anno_args)
            _ = ax.annotate("|", xy=(lossint_lo[ind], lvecnames[ind]), **anno_args)
            #plt.plot((lvecnames[ind], lvecnames[ind]), (lossint_hi[ind], lossint_lo[ind]), '_-',
             #        color=next(color), alpha=0.7, linewidth=3)
        plt.ylabel('Design Name', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
        plt.xlabel(xaxislab, fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontname('Times New Roman')
            label.set_fontsize(12)
        #plt.xticks(rotation=90)
        plt.title(plottitle,fontdict={'fontsize':16,'fontname':'Trebuchet MS'})
        if orignamelen<numvecs: # Add a legend if multiple utilities associated with each design
            import matplotlib.lines as mlines
            diamond = mlines.Line2D([], [], color='black', marker='D', linestyle='None', markersize=8, label=legendlabel[0])
            downtriangle = mlines.Line2D([], [], color='black', marker='v', linestyle='None', markersize=10, label=legendlabel[1])
            if numDups>=3:
                circle = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=8, label=legendlabel[2])
                if numDups>=4:
                    uptriangle = mlines.Line2D([], [], color='black', marker='^', linestyle='None', markersize=10, label=legendlabel[3])
                    plt.legend(handles=[diamond, downtriangle, circle,uptriangle])
                else:
                    plt.legend(handles=[diamond, downtriangle, circle])
            else:
                plt.legend(handles=[diamond, downtriangle],loc='lower right')
        fig.tight_layout()
        plt.show()
        plt.close()
    # HISTOGRAMS
    elif type=='hist':
        maxval = max([max(lveclist[i]) for i in range(numvecs)])
        maxbinnum = max([len(lveclist[i]) for i in range(numvecs)])/5
        bins = np.linspace(0.0, maxval*1.1, 100)
        fig, axs = plt.subplots(numvecs, figsize=(5, 10))
        plt.rcParams["figure.autolayout"] = True
        color = iter(cm.rainbow(np.linspace(0, 1, len(lveclist))))
        for ind in range(numvecs):
            axs[ind].hist(lveclist[ind],bins, alpha=0.5, color=next(color),label=lvecnames[ind])
            axs[ind].set_title(lvecnames[ind])
            axs[ind].set_ylim([0,maxbinnum])
        fig.suptitle(plottitle,fontsize=16)
        fig.tight_layout()
        plt.show()
        plt.close()

    return

def marginalshannoninfo(datadict):
    '''
    Takes a PMS data set and returns a matrix of the Shannon information obtained from a marginal test along each trace;
    for tracked data only
    '''
    tnNum, snNum = datadict['N'].shape
    s, r = datadict['diagSens'], datadict['diagSpec']
    postSamps = datadict['postSamples']

    shannonMat = np.zeros(shape=(tnNum, snNum))

    for samp in postSamps: # Iterate through each posterior sample
        for tnInd in range(tnNum):
            tnRate = samp[tnInd + snNum]
            for snInd in range(snNum):
                snRate = samp[snInd]
                consolRate = tnRate + (1-tnRate)*snRate
                detectRate = s*consolRate+(1-r)*(1-consolRate)
                infobit = detectRate*np.log(detectRate) + (1-detectRate)*np.log(detectRate)
                shannonMat[tnInd,snInd] += infobit

    shannonMat=shannonMat/len(postSamps)
    return shannonMat

def risk_parabolic(SFPratevec, paramDict={'threshold': 0.5}):
    '''Parabolic risk term for vector of SFP rates. Threshold is the top of the parabola. '''
    riskvec = np.empty((len(SFPratevec)))
    for ind in range(len(SFPratevec)):
        currRate = SFPratevec[ind]
        if paramDict['threshold'] <= 0.5:
            currRisk = (currRate+2*(0.5-paramDict['threshold']))*(1-currRate)
        else:
            currRisk = currRate * (1 - currRate - 2*(0.5-paramDict['threshold']))
        riskvec[ind] = currRisk
    return riskvec

def risk_check(SFPratevec, paramDict={'threshold': 0.5, 'slope': 0.5}):
    '''Check risk term, which has minus 'slope' to the right of 'threshold' and (1-'slope') to the left of threshold'''
    riskvec = np.empty((len(SFPratevec)))
    for i in range(len(SFPratevec)):
        riskvec[i] = (1 - SFPratevec[i]*(paramDict['slope']-(1-paramDict['threshold']/SFPratevec[i]
                      if SFPratevec[i]<paramDict['threshold'] else 0)))
    return riskvec

def score_diff(est, targ, paramDict):
    '''
    Returns the difference between vectors est and targ underEstWt, the weight of underestimation error relative to
    overestimation error.
    paramDict requires keys: underEstWt
    '''
    scorevec = np.empty((len(targ)))
    for i in range(len(targ)):
        scorevec[i] = (paramDict['underEstWt']*max(targ[i] - est[i], 0) + max(est[i]-targ[i],0))
    return scorevec

def score_class(est, targ, paramDict):
    '''
    Returns the difference between classification of vectors est and targ using threshold, based on underEstWt,
    the weight of underestimation error relative to overestimation error.
    paramDict requires keys: threshold, underEstWt
    '''
    scorevec = np.empty((len(targ)))
    for i in range(len(targ)):
        estClass = np.array([1 if est[i] >= paramDict['threshold'] else 0 for i in range(len(est))])
        targClass = np.array([1 if targ[i] >= paramDict['threshold'] else 0 for i in range(len(targ))])
        scorevec[i] = (paramDict['underEstWt']*max(targClass[i] - estClass[i], 0) + max(estClass[i]-targClass[i],0))
    return scorevec

def score_check(est, targ, paramDict):
    '''
    Returns a check difference between vectors est and targ using slope, which can be used to weigh underestimation and
    overestimation differently. Slopes less than 0.5 mean underestimation causes a higher loss than overestimation.
    paramDict requires keys: slope
    '''
    scorevec = np.empty((len(targ)))
    for i in range(len(targ)):
        scorevec[i] = (est[i]-targ[i])*(paramDict['slope']- (1 if est[i]<targ[i] else 0))
    return scorevec

def bayesEst(samps, scoredict):
    '''
    Returns the Bayes estimate for a set of SFP rates based on the type of score and parameters used
    scoredict: must have key 'name' and other necessary keys for calculating the associated Bayes estimate
    '''
    scorename = scoredict['name']
    if scorename == 'AbsDiff':
        underEstWt = scoredict['underEstWt']
        est = np.quantile(samps,underEstWt/(1+underEstWt), axis=0)
    elif scorename == 'Check':
        slope = scoredict['slope']
        est = np.quantile(samps,1-slope, axis=0)
    elif scorename == 'Class':
        underEstWt = scoredict['underEstWt']
        critVal = np.quantile(samps, underEstWt / (1 + underEstWt), axis=0)
        classlst = [1 if critVal[i]>=scoredict['threshold'] else 0 for i in range(len(samps[0]))]
        est = np.array(classlst)
    else:
        print('Not a valid score name')

    return est

def loss_pms(est, targ, score, scoreDict, risk, riskDict, market):
    '''
    Loss/utility function tailored for PMS.
    score, risk: score and risk functions with associated parameter dictionaries scoreDict, riskDict,
        that return vectors
    market: vector of market weights
    '''
    currloss = 0. # Initialize the loss/utility
    scorevec = score(est, targ, scoreDict)
    riskvec = risk(targ, riskDict)
    for i in range(len(targ)):
        currloss += scorevec[i] * riskvec[i] * market[i]
    return currloss

def loss_pms2(est, targ, paramDict):
    '''
    Loss/utility function tailored for PMS.
    score, risk: score and risk functions with associated parameter dictionaries scoreDict, riskDict
    market: market weights
    '''
    currloss = 0.
    epsTarg = 0.5 - paramDict['rateTarget']
    if len(paramDict['nodeWtVec'])==0: #
        nodeWtVec = [1. for i in range(len(est))]
    for i in range(len(est)):
        scoreterm = (paramDict['underEstWt']*max(targ[i] - est[i], 0) + max(est[i]-targ[i],0))
        if paramDict['checkloss']==False:
            if epsTarg < 0:
                wtterm = targ[i]*(1-targ[i]-2*epsTarg)
            else:
                wtterm = (targ[i]+2*epsTarg)*(1-targ[i])
        else:
            wtterm = 1 - targ[i]*(paramDict['checkslope']-(1-paramDict['rateTarget']/targ[i] if targ[i]<paramDict['rateTarget'] else 0))
        currloss += scoreterm * wtterm * nodeWtVec[i]
    return currloss

def showRiskVals():
    '''Generate a figure showcasing how the risk changes with different parameter choices'''
    x = np.linspace(0.001,0.999,1000)
    t = 0.3 # Our target
    y1 = (x+2*(0.5-t))*(1-x)
    tauvec = [0.05,0.2,0.4,0.6,0.95]
    fig, ax = plt.subplots(figsize=(8, 7))
    for tau in tauvec:
        newy = [1 - x[i]*(tau-(1-(t/x[i]) if x[i]<t else 0)) for i in range(len(x))]
        plt.plot(x,newy)
    plt.plot(x,y1)
    import matplotlib.ticker as mtick
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.title('Values for Parabolic and selected Check risk terms\n$l=30\%$',fontdict={'fontsize':16,'fontname':'Trebuchet MS'})
    plt.ylabel('Risk term value', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.xlabel('SFP rate', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.text(0.84, 0.97,'Check, $m=0.05$',fontdict={'fontsize': 12, 'fontname': 'Trebuchet MS'})
    plt.text(0.84, 0.84, 'Check, $m=0.2$',fontdict={'fontsize': 12, 'fontname': 'Trebuchet MS'})
    plt.text(0.84, 0.675, 'Check, $m=0.4$', fontdict={'fontsize': 12, 'fontname': 'Trebuchet MS'})
    plt.text(0.84, 0.50, 'Check, $m=0.6$', fontdict={'fontsize': 12, 'fontname': 'Trebuchet MS'})
    plt.text(0.84, 0.21, 'Check, $m=0.95$', fontdict={'fontsize': 12, 'fontname': 'Trebuchet MS'})
    plt.text(0.00, 0.47, 'Parabolic', fontdict={'fontsize': 12, 'fontname': 'Trebuchet MS'})
    fig.tight_layout()
    plt.show()
    plt.close()

    return

def showScoreVals():
    '''Generate a figure showcasing how the score changes with different parameter choices'''
    gEst = np.linspace(0.001,0.999,50) #gamma_hat
    gStar = 0.4 # gamma_star
    tauvec = [0.05,0.25,0.9] # For check score
    vvec = [0.5,1.2] # For absolute difference and classification scores
    tvec = [0.2]
    fig, ax = plt.subplots(figsize=(8, 7))
    for tau in tauvec: # Check scores
        newy = [(gStar-gEst[i])*(tau-(1 if gEst[i]<gStar else 0)) for i in range(len(gEst))]
        plt.plot(gEst,newy,':')
    for v in vvec: # Absolute difference scores
        newy = [-1*(max(gEst[i]-gStar,0)+v*max(gStar-gEst[i],0)) for i in range(len(gEst))]
        plt.plot(gEst,newy,'-')
        for t in tvec:
            classEst = [1 if gEst[i]>=t else 0 for i in range(len(gEst))]
            classStar = 1 if gStar >= t else 0
            newy = [-1*(max(classEst[i]-classStar,0)+v*max(classStar-classEst[i],0)) for i in range(len(gEst))]
            plt.plot(gEst,newy,'<',)
    import matplotlib.ticker as mtick
    plt.ylim([-1.3,0.1])
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.title('Values for selected score terms\n$\gamma^\star=40\%$',fontdict={'fontsize':16,'fontname':'Trebuchet MS'})
    plt.ylabel('Score term value', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.xlabel('SFP rate estimate', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.text(0.16, -0.55,'Class., $l=20\%$, $v=0.5$',fontdict={'fontsize': 10, 'fontname': 'Trebuchet MS'})
    plt.text(0.16, -1.17, 'Class., $l=20\%$, $v=1.2$', fontdict={'fontsize': 10, 'fontname': 'Trebuchet MS'})
    plt.text(0.80, -0.07, 'Check, $m=0.05$', fontdict={'fontsize': 10, 'fontname': 'Trebuchet MS'})
    plt.text(0.8, -0.2, 'Check, $m=0.25$', fontdict={'fontsize': 10, 'fontname': 'Trebuchet MS'})
    plt.text(0.81, -0.36, 'Check, $m=0.9$', fontdict={'fontsize': 10, 'fontname': 'Trebuchet MS'})
    plt.text(0., -0.1, 'Abs. Diff., $v=0.5$', fontdict={'fontsize': 10, 'fontname': 'Trebuchet MS'})
    plt.text(0.13, -0.37, 'Abs. Diff., $v=1.2$', fontdict={'fontsize': 10, 'fontname': 'Trebuchet MS'})
    fig.tight_layout()
    plt.show()
    plt.close()

    return

def utilityOverIncreasingData():
    '''Generate a figure showing the change in utility as n increases'''

    return

def writeObjToPickle(obj, objname='pickleObject'):
    '''HOW TO WRITE PRIOR DRAWS TO A PICKLE OBJECT TO BE LOADED LATER'''
    import pickle
    import os
    outputFilePath = os.getcwd()
    outputFileName = os.path.join(outputFilePath, objname)
    pickle.dump(obj, open(outputFileName, 'wb'))
    return

def getDesignUtility(priordatadict, lossdict, designlist, numtests, omeganum, designnames=[],
                     type=['trace'], priordraws=[], randinds=[], roundAlg=roundDesignLow, method='MCMC'):
    '''
    Produces a list of loss vectors for entered design choices under a given data set and specified loss. Each loss
        vector contains omeganum Monte Carlo integration iterations
    Designed for use with plotLossVecs() to plot Bayesian risk associated with each design.
    priordatadict: dictionary capturing all prior data. should have posterior draws from initial data set already
    included, with keys identical to those provided by logistigate functions
    lossdict: parameter dictionary to pass to lossfunc
    estdecision: list for how to form a decision from the posterior samples; one of ['mean'], ['median'], or
        ['mode', t], where t is the assignment threshold for designating the classification estimate
    designlist: list of sampling probability vectors along all test nodes or traces
    designnames: list of names for the designs
    numtests: how many samples will be obtained under each design
    omeganum: number of prior draws to use for calculating the Bayesian risk
    type: list for the type of sample collection described in each design; one of ['trace'] (collect along SN-TN trace) or
        ['test node', Qest] (collect along test nodes, along with the estimate of the sourcing probability matrix)
    priordraws: set of prior draws to use for synchronized data collection in different designs
    randinds: the indices with which to iterate through priordraws for all omeganum loops
    method: one of 'MCMC' or 'approx'; 'MCMC' completes full MCMC sampling for generating posterior probabilities,
        'approx' approximates the posterior probabilities
    '''
    # Initiate the list to return
    lossveclist = []
    # Retrieve prior draws if empty
    if len(priordraws)==0:
        priordraws = priordatadict['postSamples']
    if len(randinds)==0:
        randinds = [i for i in range(omeganum)]
    if len(designnames)==0:
        for i in range(len(designlist)):
            designnames.append('Design '+str(i))
    # Get key supply-chain elements from priordatadict
    (numTN, numSN) = priordatadict['N'].shape
    Q = priordatadict['transMat'] #May be empty
    s, r = priordatadict['diagSens'], priordatadict['diagSpec']

    # Loop through each design and generate omeganum loss realizations
    for designind, design in enumerate(designlist):
        currlossvec = []
        # Initialize samples to be drawn from traces, per the design
        sampMat = roundAlg(design, numtests)
        for omega in range(omeganum):
            TNsamps = sampMat.copy()
            # Grab a draw from the prior
            currpriordraw = priordraws[randinds[omega]]  # [SN rates, TN rates]
            # Initialize Ntilde and Ytilde
            Ntilde = np.zeros(shape = priordatadict['N'].shape)
            Ytilde = Ntilde.copy()
            while np.sum(TNsamps) > 0.:
                # Go to first non-empty row of TN samps
                i, j = 0, 0
                while np.sum(TNsamps[i])==0:
                    i += 1
                if type[0]=='trace':
                    # Go to first non-empty column of this row
                    while TNsamps[i][j]==0:
                        j += 1
                    TNsamps[i][j] -= 1
                if type[0]=='test node':
                    # Pick a supply node according to Qest
                    j = choice([i for i in range(numSN)], p=np.divide(type[1][i],np.sum(type[1][i])).tolist())
                    TNsamps[i] -= 1
                # Generate test result
                currTNrate = currpriordraw[numSN+i]
                currSNrate = currpriordraw[j]
                currrealrate = currTNrate + (1-currTNrate)*currSNrate # z_star for this sample
                currposrate = s*currrealrate+(1-r)*(1-currrealrate) # z for this sample
                result = np.random.binomial(1, p=currposrate)
                Ntilde[i, j] += 1
                Ytilde[i, j] += result

            # We have a new set of data d_tilde
            Nomega = priordatadict['N'] + Ntilde
            Yomega = priordatadict['Y'] + Ytilde

            postdatadict = priordatadict.copy()
            postdatadict['N'] = Nomega
            postdatadict['Y'] = Yomega

            if method == 'MCMC':
                # Writes over previous MCMC draws
                postdatadict = methods.GeneratePostSamples(postdatadict)
            elif method == 'approx':
                # Before calculating loss, get density weights normalized to len(postSamples)
                postDensWts = []
                for currsamp in postdatadict['postSamples']:
                    if postdatadict['type'] == 'Tracked':
                        #todo: LIKELIHOOD OR POSTERIOR LIKELIHOOD?
                        currWt = np.exp(methods.Tracked_LogLike(currsamp, Ntilde, Ytilde, priordatadict['diagSens'],
                                                         priordatadict['diagSpec']))#, priordatadict['prior']))
                    elif postdatadict['type'] == 'Untracked': #todo: REST OF FUNCTION NEEDS TO BE ADAPTED FOR UNTRACKED, NODE SAMPING SETTING
                        currWt = np.exp(methods.Untracked_LogLike(currsamp, Ntilde, Ytilde,priordatadict['diagSens'],
                                                                  priordatadict['diagSpec']),priordatadict['transMat'])
                    postDensWts.append(currWt)
                postDensWts = np.array(postDensWts)
                postDensWts = postDensWts * (len(postdatadict['postSamples']) / np.sum(postDensWts)) # Normalize


            # Get the Bayes estimate
            currEst = bayesEst(postdatadict['postSamples'],lossdict['scoreDict'])

            # Average loss for all postpost samples
            sumloss = 0
            for currsampind, currsamp in enumerate(postdatadict['postSamples']):
                if method == 'MCMC':
                    currloss = loss_pms(currEst, currsamp, lossdict['scoreFunc'], lossdict['scoreDict'],
                                        lossdict['riskFunc'],lossdict['riskDict'],lossdict['marketVec'])
                    sumloss += currloss
                if method == 'approx': # weigh each sample by p(dTilde|gamma)
                    currWt = postDensWts[currsampind]
                    currloss = currWt * loss_pms(currEst, currsamp, lossdict['scoreFunc'], lossdict['scoreDict'],
                                                 lossdict['riskFunc'],lossdict['riskDict'],lossdict['marketVec'])
                    sumloss += currloss
            avgloss = sumloss/len(postdatadict['postSamples'])

            #Append to utility storage vector
            currlossvec.append(avgloss)
            print(designnames[designind]+', '+'omega '+str(omega) + ' complete')
        lossveclist.append(currlossvec)

    return lossveclist
