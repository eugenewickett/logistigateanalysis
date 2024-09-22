"""
Utility estimates for the 'existing' setting
"""
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
csdict_fam = util.initDataDict(Nfam, Yfam) # Initialize necessary logistigate keys
csdict_fam['TNnames'] = ['MOD_39', 'MOD_17', 'MODHIGH_95', 'MODHIGH_26']
csdict_fam['SNnames'] = ['MNFR ' + str(i+1) for i in range(numSN)]

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
csdict_fam['prior'] = priorObj

# Set up MCMC
csdict_fam['MCMCdict'] = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
# Generate posterior draws
numdraws = 75000
csdict_fam['numPostSamples'] = numdraws
np.random.seed(1000) # To replicate draws later
csdict_fam = methods.GeneratePostSamples(csdict_fam)
# Print inference from initial data
# util.plotPostSamples(csdict_fam, 'int90')

# Loss specification
paramdict = lf.build_diffscore_checkrisk_dict(scoreunderestwt=5., riskthreshold=0.15, riskslope=0.6,
                                              marketvec=np.ones(numTN + numSN))

# Set limits of data collection and intervals for calculation
testmax, testint = 400, 10
testarr = np.arange(testint, testmax + testint, testint)

# Set MCMC draws to use in fast algorithm
numtruthdraws, numdatadraws = 75000, 2000
# Get random subsets for truth and data draws
np.random.seed(444)
truthdraws, datadraws = util.distribute_truthdata_draws(csdict_fam['postSamples'], numtruthdraws, numdatadraws)
paramdict.update({'truthdraws': truthdraws, 'datadraws': datadraws})
# Get base loss
paramdict['baseloss'] = sampf.baseloss(paramdict['truthdraws'], paramdict)

util.print_param_checks(paramdict)  # Check of used parameters

# TEST THAT IMPORTANCE SAMPLING IS WORKING AS EXPECTED
import warnings
import time
warnings.simplefilter(action='ignore', category=FutureWarning)

for reps in range(10):
    n = np.array([100,160,130,10])

    currTime = time.time()
    uEst, uCI = sampf.getImportanceUtilityEstimate(n, csdict_fam, paramdict, numimportdraws=25000,
                                                   numdatadrawsforimportance=1000,
                                                   impweightoutlierprop=0.01, zlevel=0.95)
    print(uCI)
    print("Time: "+str(round(time.time()-currTime, 1)))
'''
n = np.array([0,20,0,0])
basic method: (0.15485069147878416, 0.16180554729293029)
imp method, 5k: (0.15829377402469902, 0.16394741405461755)
imp method, 25k: (0.1640837936618489, 0.16969239341317088)
imp method, 70k: (0.1758604459345512, 0.18075470909683622)

n = np.array([30,90,0,0])
basic method: (0.4987593570451616, 0.5097299074978237)
imp method, 5k: (0.5019468886767082, 0.5090838442627799)
imp method, 25k: (0.5044585260556864, 0.5108398461912593)
imp method, 70k: (0.487326114458986, 0.4931929537382993)
imp method, 5k: (0.5122422019768191, 0.5189348919589116)
                (0.5255941117260712, 0.5319491154816347)
                (0.49904387249604865, 0.5057974983751081)
                (0.5414205810418111, 0.5482604270359901)
                (0.5061389581206663, 0.5133623232472941)
                (0.5209600295270844, 0.5270071370792504)
                (0.5133683186165274, 0.5198186677291663)
                (0.5106304799110062, 0.5174843688397013)
                (0.49869442833495703, 0.505594987666049)
                (0.5021666726116905, 0.5090284350612235)
imp method, 25k:(0.49175313850853475, 0.4978991170025855)
                (0.49056763048572205, 0.496715820793433)
                (0.4994825946708372, 0.5054552122182299)
                (0.49786956178492914, 0.5038921032181591)
                (0.504457159265469, 0.5108036002916723)
                (0.5004352732693327, 0.506756385128255)
                (0.49453361274862173, 0.5008903468077146)
                (0.49197917868521834, 0.4979008017679303)
                (0.5039991702205184, 0.5101557766643947)
                (0.482356877471946, 0.4883969345555932)
imp method, 50k:(0.4979937690759486, 0.5042273433669855)
                (0.4889175566395392, 0.4952324319656132) 
                (0.4969157497779435, 0.5031477601089833)
                (0.4956875372903966, 0.5020303932500181)
                (0.49847694353418226, 0.5047172923867953)
                (0.4882644471929325, 0.4945589030961175)
                (0.493988097471288, 0.4999428389960616)
                (0.49017850585920586, 0.4962521808529643)
                (0.4986508215823773, 0.5051075550246633)
                (0.48635355306417605, 0.4928010474946811)
imp method, 70k:(0.5027218025722038, 0.5087144831865229)
                (0.4950622248634853, 0.5011370980045762)
                (0.48553358299620086, 0.491849077644702)
                (0.49134921729089065, 0.49764985256171235)
                (0.49563187435692124, 0.5018991095329415)
                (0.49979971199129203, 0.5058339298894863)
                (0.4944956536350531, 0.5006523767751168)
                (0.5011506584815444, 0.5070879920969769)
                (0.4911257355688965, 0.49708963942402185)
                (0.4984063234645717, 0.5043349385896652)

x = ["base"] + ["5k"+str(i) for i in range(10)] + ["25k"+str(i) for i in range(10)] + ["50k"+str(i) for i in range(10)] +\ 
        ["70k"+str(i) for i in range(10)] 
y1 =  [0.4987593570451616,   
        0.5122422019768191, 0.5255941117260712, 0.49904387249604865,  0.5414205810418111, 0.5061389581206663,  
        0.5209600295270844, 0.5133683186165274, 0.5106304799110062, 0.49869442833495703, 0.5021666726116905,
        0.49175313850853475, 0.49056763048572205, 0.4994825946708372, 0.49786956178492914, 0.504457159265469,
        0.5004352732693327, 0.49453361274862173, 0.49197917868521834, 0.5039991702205184, 0.482356877471946,
        0.4979937690759486, 0.4889175566395392, 0.4969157497779435, 0.4956875372903966, 0.49847694353418226,
        0.4882644471929325, 0.493988097471288, 0.49017850585920586, 0.4986508215823773, 0.48635355306417605,
        0.5027218025722038, 0.4950622248634853, 0.48553358299620086, 0.49134921729089065, 0.49563187435692124,
        0.49979971199129203, 0.4944956536350531, 0.5011506584815444, 0.4911257355688965, 0.4984063234645717]
y2 = [0.5097299074978237, 
        0.5189348919589116, 0.5319491154816347, 0.5057974983751081, 0.5482604270359901, 0.5133623232472941, 
        0.5270071370792504, 0.5198186677291663, 0.5174843688397013, 0.505594987666049, 0.5090284350612235,
        0.4978991170025855, 0.496715820793433, 0.5054552122182299, 0.5038921032181591, 0.5108036002916723, 
        0.506756385128255, 0.5008903468077146, 0.4979008017679303, 0.5101557766643947, 0.4883969345555932,
        0.5042273433669855, 0.4952324319656132, 0.5031477601089833, 0.5020303932500181, 0.5047172923867953, 
        0.4945589030961175, 0.4999428389960616, 0.4962521808529643, 0.5051075550246633, 0.4928010474946811,
         0.5087144831865229, 0.5011370980045762, 0.491849077644702, 0.49764985256171235, 0.5018991095329415, 
         0.5058339298894863, 0.5006523767751168, 0.5070879920969769, 0.49708963942402185, 0.5043349385896652]
plt.plot(x, y1, 'x')
plt.plot(x, y2, 'x')
plt.ylim([0.4,0.6])
plt.xticks(rotation=90,size=8)
plt.show()

######
25k, 50k, and 70k all appear to be providing consistent estimates
######

n = np.array([100,160,130,10])
basic method: (1.05844248, 1.08450229)
imp method, 5k:     (0.9253012114887964, 0.9322452351882387)
                    (0.9023832148388147, 0.909526388782341)
                    (0.93132875373519, 0.9383196659579629)
                    (0.9164834335523182, 0.9234606764649607)
                    (0.9106526902055565, 0.917616067854202)
                    (0.910940761807298, 0.9177135666844716)
                    (0.9194383690518642, 0.9258166418019158)
                    (0.9291415352710548, 0.9361512151829563)
                    (0.9093926829243972, 0.9171035461283472)
                    (0.91271735323133, 0.9194556405567262)
imp method, 25k:    (0.9105249612539257, 0.916298649275596)
                    (0.8995858105184693, 0.9055040114436017)
                    (0.8996348079531449, 0.9056974150702046)
                    (0.8981722456487302, 0.9042165543030463)
                    (0.8998476271339781, 0.9056177643762788)
                    (0.9079951893322609, 0.9139227540154737)
                    (0.905878184770907, 0.9120793054381031)
                    (0.8948522435309256, 0.9009393867234194)
                    (0.8923878727239616, 0.8985351773598513)
                    (0.8993442600323784, 0.9057793256584943)
imp method, 50k:    (0.901087322056811, 0.9070564036627808)
                    (0.8898447460139489, 0.8959176418969226)
                    (0.8968748880711876, 0.9026403957119462)
                    (0.8924822525594505, 0.898217926404719)
                    (0.8993044601767246, 0.9050790046820625)
                    (0.9087572547262657, 0.9149117006819896)
                    (0.8962660094580917, 0.9022109860883296)
                    (0.8994962657862196, 0.9053616524352874)
                    (0.8764893704572316, 0.8824850547931973)
                    (0.9026563297919503, 0.9085403435417503)
imp method, 70k:    (0.8983593534562251, 0.9038108443381712)
                    (0.8762051336083789, 0.8823053508585839)
                    (0.8969175761755273, 0.9028357799954729)
                    (0.902630736465087, 0.9087533632038687)
                    (0.8958296766591782, 0.9016779137292181)
                    (0.89745877429779, 0.9032658085404062)
                    (0.9016528323977218, 0.9073876071110525)
                    (0.9055098912469126, 0.9114888587665018)
                    (0.9050263853739637, 0.911021908246755)
                    (0.896510727425204, 0.9023684507874241)
                    
x = ["base"] + ["5k"+str(i) for i in range(10)] + ["25k"+str(i) for i in range(10)] + ["50k"+str(i) for i in range(10)] +\ 
        ["70k"+str(i) for i in range(10)] 
y1 =    [1.05844248,   
        0.9253012114887964, 0.9023832148388147, 0.93132875373519, 0.9164834335523182, 0.9106526902055565,
        0.910940761807298, 0.9194383690518642, 0.9291415352710548, 0.9093926829243972, 0.91271735323133,
        0.9105249612539257, 0.8995858105184693, 0.8996348079531449, 0.8981722456487302, 0.8998476271339781, 
        0.9079951893322609, 0.905878184770907, 0.8948522435309256, 0.8923878727239616, 0.8993442600323784,
        0.901087322056811, 0.8898447460139489, 0.8968748880711876, 0.8924822525594505, 0.8993044601767246, 
        0.9087572547262657, 0.8962660094580917, 0.8994962657862196, 0.8764893704572316, 0.9026563297919503,
        0.8983593534562251, 0.8762051336083789, 0.8969175761755273, 0.902630736465087, 0.8958296766591782, 
        0.89745877429779, 0.9016528323977218, 0.9055098912469126, 0.9050263853739637, 0.896510727425204]
y2 =    [1.08450229, 
        0.9322452351882387, 0.909526388782341, 0.9383196659579629, 0.9234606764649607, 0.917616067854202, 
        0.9177135666844716, 0.9258166418019158, 0.9361512151829563, 0.9171035461283472, 0.9194556405567262,
        0.916298649275596, 0.9055040114436017, 0.9056974150702046, 0.9042165543030463, 0.9056177643762788, 
        0.9139227540154737, 0.9120793054381031, 0.9009393867234194, 0.8985351773598513, 0.9057793256584943,
        0.9070564036627808, 0.8959176418969226, 0.9026403957119462, 0.898217926404719, 0.9050790046820625, 
        0.9149117006819896, 0.9022109860883296, 0.9053616524352874, 0.8824850547931973, 0.9085403435417503,
        0.9038108443381712, 0.8823053508585839, 0.9028357799954729, 0.9087533632038687, 0.9016779137292181, 
        0.9032658085404062, 0.9073876071110525, 0.9114888587665018, 0.911021908246755, 0.9023684507874241]
plt.plot(x, y1, 'x')
plt.plot(x, y2, 'x')
plt.ylim([0.0,1.1])
plt.xticks(rotation=90,size=8)
plt.show()                   
                    
'''

###############
# UPDATED HEURISTIC
###############
alloc, util_avg, util_hi, util_lo = sampf.get_greedy_allocation(csdict_fam, testmax, testint, paramdict,
                                                                numimpdraws=60000, numdatadrawsforimp=5000,
                                                                impwtoutlierprop=0.005,
                                                                printupdate=True, plotupdate=True,
                                                                plottitlestr='Familiar Setting')
''' 21-JUN
[1, 1, 1, 1, 1, 0, 1, 1, 0,
 0, 1, 1, 2, 2, 0, 2, 1, 2,
 0, 2, 2, 1, 1, 1, 2, 3, 0,
 2, 0, 0, 2, 0, 2, 1, 2, 1,
 2, 1, 0, 2]
[(0.08709501401732922, 0.09248817606288728), (0.15485069147878416, 0.16180554729293029), (0.21132869080310357, 0.21931614005076927),
(0.25203372890146003, 0.26063269080331164), (0.29508233713522936, 0.3040828160807423), (0.3287205863959288, 0.33822854576624817), 
(0.36679105798445644, 0.3762582028535526), (0.3971444263505597, 0.4070388117759731), (0.42485199517149574, 0.4353124807349904),
(0.4486951832489643, 0.45897840819784275), (0.4761109827432908, 0.48670866415972003), (0.4987593570451616, 0.5097299074978237),
(0.5263266477028608, 0.5378674990051908), (0.5478847373930049, 0.5603035172340813), (0.5694258037490463, 0.5816813331255415),
(0.5935104072398629, 0.6068050714362494), (0.6142821720902099, 0.6267292115039553), (0.6375264642099632, 0.651162904882391),
(0.6569186350254319, 0.6712814975687573), (0.6751661618775924, 0.6901828740981897), (0.6991582605312183, 0.7140608585450436),
(0.7218481897152977, 0.7377503914991284), (0.7391189140792602, 0.7550833236590053), (0.7595669154703588, 0.7763418662828772),
(0.7728264632929887, 0.7898690825001178), (0.7930497369742868, 0.8103019605083919), (0.8096944981072554, 0.8272548106037041),
(0.8314307088795085, 0.8494696827012405), (0.8509469900480071, 0.8700952115080858), (0.8698554241240988, 0.8897037201438907),
(0.8914266265511126, 0.9116777365803721), (0.9113010414418516, 0.9332423013141198), (0.9227597014567777, 0.9449340291518391),
(0.9476763861351147, 0.9699704646742606), (0.9638385560225826, 0.986696220828474), (0.9821130962293003, 1.0061686183597554),
(1.0042917413295647, 1.0287685913425824), (1.0217862509874143, 1.0462029388490077), (1.038252436160777, 1.0634683423858708),
(1.058442480561017, 1.0845022908644895)]
'''
np.save(os.path.join('casestudyoutputs', 'familiar', 'fam_alloc'), alloc)
np.save(os.path.join('casestudyoutputs', 'familiar', 'fam_util_avg'), util_avg)
np.save(os.path.join('casestudyoutputs', 'familiar', 'fam_util_hi'), util_hi)
np.save(os.path.join('casestudyoutputs', 'familiar', 'fam_util_lo'), util_lo)

# Evaluate utility for uniform and rudimentary
util_avg_unif, util_hi_unif, util_lo_unif = np.zeros((int(testmax / testint) + 1)), \
                                            np.zeros((int(testmax / testint) + 1)), \
                                            np.zeros((int(testmax / testint) + 1))
util_avg_rudi, util_hi_rudi, util_lo_rudi = np.zeros((int(testmax / testint) + 1)), \
                                            np.zeros((int(testmax / testint) + 1)), \
                                            np.zeros((int(testmax / testint) + 1))
plotupdate = True
for testind in range(testarr.shape[0]):
    # Uniform utility
    des_unif = util.round_design_low(np.ones(numTN) / numTN, testarr[testind]) / testarr[testind]
    currlosslist = sampf.sampling_plan_loss_list(des_unif, testarr[testind], csdict_fam, paramdict)
    avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
    util_avg_unif[testind+1] = paramdict['baseloss'] - avg_loss
    util_lo_unif[testind+1] = paramdict['baseloss'] - avg_loss_CI[1]
    util_hi_unif[testind+1] = paramdict['baseloss'] - avg_loss_CI[0]
    print(des_unif)
    print('Utility at ' + str(testarr[testind]) + ' tests, Uniform: ' + str(util_avg_unif[testind+1]))
    # Rudimentary utility
    des_rudi = util.round_design_low(np.divide(np.sum(Nfam, axis=1), np.sum(Nfam)), testarr[testind]) / testarr[testind]
    currlosslist = sampf.sampling_plan_loss_list(des_rudi, testarr[testind], csdict_fam, paramdict)
    avg_loss, avg_loss_CI = sampf.process_loss_list(currlosslist, zlevel=0.95)
    util_avg_rudi[testind+1] = paramdict['baseloss'] - avg_loss
    util_lo_rudi[testind+1] = paramdict['baseloss'] - avg_loss_CI[1]
    util_hi_rudi[testind+1] = paramdict['baseloss'] - avg_loss_CI[0]
    print(des_rudi)
    print('Utility at ' + str(testarr[testind]) + ' tests, Rudimentary: ' + str(util_avg_rudi[testind+1]))
    if plotupdate:
        util_avg_arr = np.vstack((util_avg_unif, util_avg_rudi))
        util_hi_arr = np.vstack((util_hi_unif, util_hi_rudi))
        util_lo_arr = np.vstack((util_lo_unif, util_lo_rudi))
        # Plot
        util.plot_marg_util_CI(util_avg_arr, util_hi_arr, util_lo_arr, testmax=testmax, testint=testint,
                               titlestr='Familiar Setting, comparison with other approaches')

# Store matrices
np.save(os.path.join('casestudyoutputs', 'familiar', 'util_avg_arr_fam'), util_avg_arr)
np.save(os.path.join('casestudyoutputs', 'familiar', 'util_hi_arr_fam'), util_hi_arr)
np.save(os.path.join('casestudyoutputs', 'familiar', 'util_lo_arr_fam'), util_lo_arr)

targind = 10 # where do we want to gauge budget savings?
targval = util_avg[targind]

# Uniform
kInd = next(x for x, val in enumerate(util_avg_arr[0].tolist()) if val > targval)
unif_saved = round((targval - util_avg_arr[0][kInd - 1]) / (util_avg_arr[0][kInd] - util_avg_arr[0][kInd - 1]) *\
                      testint) + (kInd - 1) * testint - targind*testint
print(unif_saved)  # 33
# Rudimentary
kInd = next(x for x, val in enumerate(util_avg_arr[1].tolist()) if val > targval)
rudi_saved = round((targval - util_avg_arr[1][kInd - 1]) / (util_avg_arr[1][kInd] - util_avg_arr[1][kInd - 1]) *\
                      testint) + (kInd - 1) * testint - targind*testint
print(rudi_saved)  # 57
