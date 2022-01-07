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
    priorMean, priorVar = -2.5, 3.5
    numPostSamps = 1000
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}

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


    # Number of SFPs: 18%
    # SN rates: 1% baseline; 5% node: ~25/30%, 3% node: ~5%,
    # TN rates: 1% baseline; 1 major node: 17%, 1 minor node: 10%; 3 minor nodes: 5%


    numTN, numSN = 25, 25
    rseed = 44

    dataTblDict = {}

    SNnames = ['Manufacturer ' + str(i + 1) for i in range(numSN)]
    outNames = ['District ' + str(i + 1) for i in range(numTN)]

    trueRates = np.zeros(numSN + numTN)  # importers first, outlets second
    trueRates[:numSN] =

    random.seed(randSeed)

    trueRates[:numImp] = [random.betavariate(1, 9) for i in range(numImp)]
    trueRates[numImp:] = [random.betavariate(1, 9) for i in range(numOut)]

    # Generate random transition matrix
    transMat = np.zeros(shape=(numOut, numImp))
    if randSeed >= 0:
        random.seed(randSeed + 1)
    for outInd in range(numOut):
        rowRands = [random.paretovariate(transMatLambda) for i in range(numImp)]
        if numImp > 10:  # Only keep 10 randomly chosen importers, if numImp > 10
            rowRands[10:] = [0.0 for i in range(numImp - 10)]
            random.shuffle(rowRands)

        normalizedRands = [rowRands[i] / sum(rowRands) for i in range(numImp)]
        # only keep transition probabilities above 2%
        # normalizedRands = [normalizedRands[i] if normalizedRands[i]>0.02 else 0.0 for i in range(numImp)]

        # normalizedRands = [normalizedRands[i] / sum(normalizedRands) for i in range(numImp)]
        transMat[outInd, :] = normalizedRands

    # np.linalg.det(transMat.T @ transMat) / numOut
    # 1.297 for n=50

    # Generate testing data
    testingDataList = []
    if dataType == 'Tracked':
        if randSeed >= 0:
            random.seed(randSeed + 2)
            np.random.seed(randSeed)
        for currSamp in range(numSamples):
            currOutlet = random.sample(outNames, 1)[0]
            currImporter = random.choices(impNames, weights=transMat[outNames.index(currOutlet)], k=1)[0]
            currOutRate = trueRates[numImp + outNames.index(currOutlet)]
            currImpRate = trueRates[impNames.index(currImporter)]
            realRate = currOutRate + currImpRate - currOutRate * currImpRate
            realResult = np.random.binomial(1, p=realRate)
            if realResult == 1:
                result = np.random.binomial(1, p=diagSens)
            if realResult == 0:
                result = np.random.binomial(1, p=1 - diagSpec)
            testingDataList.append([currOutlet, currImporter, result])
    elif dataType == 'Untracked':
        if randSeed >= 0:
            random.seed(randSeed + 3)
            np.random.seed(randSeed)
        for currSamp in range(numSamples):
            currOutlet = random.sample(outNames, 1)[0]
            currImporter = random.choices(impNames, weights=transMat[outNames.index(currOutlet)], k=1)[0]
            currOutRate = trueRates[numImp + outNames.index(currOutlet)]
            currImpRate = trueRates[impNames.index(currImporter)]
            realRate = currOutRate + currImpRate - currOutRate * currImpRate
            realResult = np.random.binomial(1, p=realRate)
            if realResult == 1:
                result = np.random.binomial(1, p=diagSens)
            if realResult == 0:
                result = np.random.binomial(1, p=1 - diagSpec)
            testingDataList.append([currOutlet, result])

    dataTblDict.update({'outletNames': outNames, 'importerNames': impNames,
                        'diagSens': 0.90, 'diagSpec': 0.99, 'type': dataType,
                        'dataTbl': testingDataList, 'transMat': transMat,
                        'trueRates': trueRates})







    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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


return
