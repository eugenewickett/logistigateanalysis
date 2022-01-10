# -*- coding: utf-8 -*-
'''
Script that generates and analyzes a synthetic set of PMS data using 3 geographic levels. Each data point has

'''

from logistigate.logistigate import utilities as util # Pull from the submodule "develop" branch
from logistigate.logistigate import methods
from logistigate.logistigate import lg

def generateSyntheticData():
    '''
    Form a synthetic data set with 3 geographic tiers of information associated with test nodes. True SFP manifestation
    happens in the 2nd tier, as well as with a few select supply nodes.

    '''
    import scipy.special as sps
    import numpy as np
    import os
    SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
    filesPath = os.path.join(SCRIPT_DIR, 'MQDfiles')
    outputFileName = os.path.join(filesPath, 'pickleOutput')
    # sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, 'logistigate', 'exmples','data')))


    import pickle
    openFile = open(outputFileName, 'rb')  # Read the file
    dataDict = pickle.load(openFile)
    sen_df = dataDict['df_SEN']
    sen_df_2010 = sen_df[(sen_df['Date_Received'] == '7/12/2010') & (sen_df['Manufacturer_GROUPED'] != 'Unknown') & (
                sen_df['Facility_Location_GROUPED'] != 'Missing')].copy()
    tbl_sen1 = sen_df_2010[['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    tbl_sen1 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in tbl_sen1]
    tbl_sen2 = sen_df_2010[['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    tbl_sen2 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in tbl_sen2]
    tbl_sen3 = sen_df_2010[['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    tbl_sen3 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in tbl_sen3]

    '''Replace Senegal identities with generic names'''


    import random

    # Replace Manufacturers
    orig_MANUF_lst = ['Ajanta Pharma Limited', 'Aurobindo Pharmaceuticals Ltd', 'Bliss Gvis Pharma Ltd', 'Cipla Ltd',
                      'Cupin', 'EGR pharm Ltd', 'El Nasr', 'Emcure Pharmaceuticals Ltd', 'Expharm',
                      'F.Hoffmann-La Roche Ltd', 'Gracure Pharma Ltd', 'Hetdero Drugs Limited', 'Imex Health',
                      'Innothera Chouzy', 'Ipca Laboratories', 'Lupin Limited', 'Macleods Pharmaceuticals Ltd',
                      'Matrix Laboratories Limited', 'Medico Remedies Pvt Ltd', 'Mepha Ltd', 'Novartis', 'Odypharm Ltd',
                      'Pfizer', 'Sanofi Aventis', 'Sanofi Synthelabo']
    shuf_MANUF_lst = orig_MANUF_lst.copy()
    random.seed(333)
    random.shuffle(shuf_MANUF_lst)
    # print(shuf_MANUF_lst)
    for i in range(len(shuf_MANUF_lst)):
        currName = shuf_MANUF_lst[i]
        newName = 'Mnfr ' + str(i)
        for ind, item in enumerate(tbl_sen1):
            if item[1] == currName:
                tbl_sen1[ind][1] = newName
        for ind, item in enumerate(tbl_sen2):
            if item[1] == currName:
                tbl_sen2[ind][1] = newName
        for ind, item in enumerate(tbl_sen3):
            if item[1] == currName:
                tbl_sen3[ind][1] = newName
    # Replace Province
    orig_PROV_lst = ['Dakar', 'Kaffrine', 'Kaolack', 'Kedougou', 'Kolda', 'Matam', 'Saint Louis']
    shuf_PROV_lst = orig_PROV_lst.copy()
    random.seed(333)
    random.shuffle(shuf_PROV_lst)
    # print(shuf_PROV_lst)
    for i in range(len(shuf_PROV_lst)):
        currName = shuf_PROV_lst[i]
        newName = 'Province ' + str(i)
        for ind, item in enumerate(tbl_sen1):
            if item[0] == currName:
                tbl_sen1[ind][0] = newName
    # Replace Facility Location
    orig_LOCAT_lst = ['Dioum', 'Diourbel', 'Fann- Dakar', 'Guediawaye', 'Hann', 'Kaffrine (City)', 'Kanel',
                      'Kaolack (City)', 'Kebemer', 'Kedougou (City)', 'Kolda (City)', 'Koumpantoum', 'Matam (City)',
                      'Mbour-Thies', 'Medina', 'Ouro-Sogui', 'Richard Toll', 'Rufisque-Dakar', 'Saint Louis (City)',
                      'Tambacounda', 'Thies', 'Tivaoune', 'Velingara']
    shuf_LOCAT_lst = orig_LOCAT_lst.copy()
    random.seed(333)
    random.shuffle(shuf_LOCAT_lst)
    # print(shuf_LOCAT_lst)
    for i in range(len(shuf_LOCAT_lst)):
        currName = shuf_LOCAT_lst[i]
        newName = 'Facil. Location ' + str(i)
        for ind, item in enumerate(tbl_sen2):
            if item[0] == currName:
                tbl_sen2[ind][0] = newName
    # Replace Facility Name
    orig_NAME_lst = ['CHR', 'CTA-Fann', 'Centre Hospitalier Regional de Thies', 'Centre de Sante Diourbel',
            'Centre de Sante Mbacke', 'Centre de Sante Ousmane Ngom', 'Centre de Sante Roi Baudouin',
            'Centre de Sante de Dioum', 'Centre de Sante de Kanel', 'Centre de Sante de Kedougou',
            'Centre de Sante de Kolda', 'Centre de Sante de Koumpantoum', 'Centre de Sante de Matam',
            'Centre de Sante de Richard Toll', 'Centre de Sante de Tambacounda', 'Centre de Sante de Velingara',
            'Centre de Traitement de la Tuberculose de Touba', 'District Sanitaire Touba', 'District Sanitaire de Mbour',
            'District Sanitaire de Rufisque', 'District Sanitaire de Tivaoune', 'District Sud', 'Hopital Diourbel',
            'Hopital Regional de Saint Louis', 'Hopital Regionale de Ouro-Sogui', 'Hopital Touba', 'Hopital de Dioum',
            'Hopitale Regionale de Koda', 'Hopitale Regionale de Tambacounda', 'PNA', 'PRA', 'PRA Diourbel', 'PRA Thies',
            'Pharmacie', 'Pharmacie Awa Barry', 'Pharmacie Babacar Sy', 'Pharmacie Boubakh',
            'Pharmacie Ceikh Ousmane Mbacke', 'Pharmacie Centrale Dr A.C.', "Pharmacie Chateau d'Eau",
            'Pharmacie Cheikh Tidiane', 'Pharmacie El Hadj Omar Tall', 'Pharmacie Fouladou', 'Pharmacie Kancisse',
            'Pharmacie Keneya', 'Pharmacie Kolda', 'Pharmacie Koldoise', 'Pharmacie Mame Diarra Bousso Dr Y.D.D.',
            'Pharmacie Mame Fatou Diop Yoro', 'Pharmacie Mame Ibrahima Ndour Dr A.N.', 'Pharmacie Mame Madia',
            'Pharmacie Ndamatou Dr O.N.', 'Pharmacie Oriantale', 'Pharmacie Oumou Khairy Ndiaye', 'Pharmacie Ousmane',
            "Pharmacie Regionale d' Approvisionnement de Saint Louis", 'Pharmacie Saloum', 'Pharmacie Sogui',
            'Pharmacie Teddungal', 'Pharmacie Thiala', 'Pharmacie Thierno Mouhamadou Seydou Ba',
            'Pharmacie Touba Mosque Dr A.M.K.', 'Pharmacie Ya Salam', 'Pharmacie du Baool Dr El-B.C.',
            'Pharmacie du Fleuve', 'Pharmacie du Marche']
    shuf_NAME_lst = orig_NAME_lst.copy()
    random.seed(333)
    random.shuffle(shuf_NAME_lst)
    # print(shuf_NAME_lst)
    for i in range(len(shuf_NAME_lst)):
        currName = shuf_NAME_lst[i]
        newName = 'Facil. Name ' + str(i)
        for ind, item in enumerate(tbl_sen3):
            if item[0] == currName:
                tbl_sen3[ind][0] = newName









    '''
    Use estimated Q to generate 400 samples under different seeds until finding a data set that works
    '''
    import numpy as np
    import random

    Qrow = np.array([.01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01,
                     .02, .02, .02, .03, .03, .05, .05, .07, .07, .07, .10, .15, .20])
    random.seed(3)
    random.shuffle(Qrow)
    # Qrow: [0.01, 0.03, 0.1 , 0.02, 0.01, 0.01, 0.07, 0.01, 0.01, 0.02, 0.2, 0.02,
    #        0.01, 0.01, 0.07, 0.15, 0.01, 0.01, 0.03, 0.07, 0.01, 0.01, 0.05, 0.05, 0.01])


    # Overall SFP rate: ???
    # SN rates: 1% baseline; 20% node: 25%, 5% node: ~25/30%, 7% node: 10%, 2% node: 40%
    # TN rates: 1% baseline; 1 major node: 25%, 1 minor node: 30%; 3 minor nodes: 10%; 1 minor minor node: 50%

    numTN, numSN = 25, 25
    numSamples = 500
    s, r = 1.0, 1.0


    dataTblDict = {}

    SNnames = ['Manufacturer ' + str(i + 1) for i in range(numSN)]
    TNnames = ['District ' + str(i + 1) for i in range(numTN)]

    trueRates = np.zeros(numSN + numTN)  # importers first, outlets second

    SNtrueRates = [.02 for i in range(numSN)]
    SN1ind = 3 # 40% SFP rate
    SN2ind = 10 # 25% SFP rate, major node
    SN3ind = 14 # 10% SFP rate, minor node
    SN4ind = 22 # 20% SFP rate, minor node
    SNtrueRates[SN1ind], SNtrueRates[SN2ind] = 0.4, 0.25
    SNtrueRates[SN3ind], SNtrueRates[SN4ind] = 0.1, 0.2

    trueRates[:numSN] = SNtrueRates # SN SFP rates

    TN1ind = 5 # 20% sourced node, 25% SFP rate
    TN2inds = [2, 11, 14, 22] # 10% sourced
    TN3inds = [3, 6, 8, 10, 16, 17, 24] # 3% sourced
    TN4inds = [0, 1, 9, 12, 18, 23] # 2% sourced
    TNsampProbs = [.01 for i in range(numTN)] # Update sampling probs
    TNsampProbs[TN1ind] = 0.2
    for j in TN2inds:
        TNsampProbs[j] = 0.10
    for j in TN3inds:
        TNsampProbs[j] = 0.03
    for j in TN4inds:
        TNsampProbs[j] = 0.02
    print(np.sum(TNsampProbs))

    TNtrueRates = [.01 for i in range(numTN)] # Update SFP rates for TNs
    TNtrueRates[TN1ind] = 0.25
    TNtrueRates[TN2inds[1]] = 0.1
    TNtrueRates[TN2inds[2]] = 0.1
    TNtrueRates[TN3inds[1]] = 0.4
    trueRates[numSN:] = TNtrueRates # Put TN rates in main vector

    rseed = 47
    random.seed(rseed)
    np.random.seed(rseed+1)
    testingDataList = []
    for currSamp in range(numSamples):
        currTN = random.choices(TNnames, weights=TNsampProbs, k=1)[0]
        currSN = random.choices(SNnames, weights=Qrow, k=1)[0] #[TNnames.index(currTN)] to index Q
        currTNrate = trueRates[numSN + TNnames.index(currTN)]
        currSNrate = trueRates[SNnames.index(currSN)]
        realRate = currTNrate + currSNrate - currTNrate * currSNrate
        realResult = np.random.binomial(1, p=realRate)
        if realResult == 1:
            result = np.random.binomial(1, p = s)
        if realResult == 0:
            result = np.random.binomial(1, p=1. - r)
        testingDataList.append([currTN, currSN, result])

    # Inspect testing data; check: (1) overall SFP rate, (2) plots, (3) N, Y matrices align more or less with
    # statements from case-study section
    priorMean, priorVar = -2.5, 3.5
    numPostSamps = 1000
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}

    lgDict = util.testresultsfiletotable(testingDataList, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])




    #np.savetxt("N_matrix.csv", lgDict['N'], delimiter=",")
    #np.savetxt("Y_matrix.csv", lgDict['Y'], delimiter=",")

    # NEED TO DO:
    #   GROUP DISTRICTS INTO PROVINCES, RE-RUN
    #   SEPARATE DISTRICTS INTO FACILITIES, RE-RUN
    #


    '''
    SNinds = lgDict['importerNames'].index('Mnfr. 5')
    print('Manufacturer 5: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    SNinds = lgDict['importerNames'].index('Mnfr. 8')
    print('Manufacturer 8: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    SNinds = lgDict['importerNames'].index('Mnfr. 10')
    print('Manufacturer 10: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    TNinds = lgDict['outletNames'].index('Province 2')
    print('Province 2: (' + str(np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.05))[
                            :5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.95))[:5] + ')')

    lgDict = util.testresultsfiletotable(tbl_SEN_G2_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
    TNinds = lgDict['outletNames'].index('Facil. Location 7')
    print('Facility Location 7: (' + str(
        np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.05))[
                                     :5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.95))[:5] + ')')
    TNinds = lgDict['outletNames'].index('Facil. Location 8')
    print('Facility Location 8: (' + str(
        np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.05))[
                                     :5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.95))[:5] + ')')

    lgDict = util.testresultsfiletotable(tbl_SEN_G3_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Name', '\nSenegal - Facility Name'])
    '''

    return
