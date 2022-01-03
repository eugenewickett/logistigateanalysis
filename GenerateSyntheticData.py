# -*- coding: utf-8 -*-
'''
Script that generates and analyzes a synthetic set of PMS data using 3 geographic levels. Each data point has

'''

from logistigate.logistigate import utilities as util # Pull from the submodule "develop" branch
from logistigate.logistigate import methods
from logistigate.logistigate import lg

def generateSyntheticData():
    ''' Form a synthetic data set'''
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
    priorMean, priorVar = -1.338762078, 0.209397261 * 5
    numPostSamps = 1000
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}

    # NAME KEYS
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
    # print(orig_MANUF_lst)
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





    ['Dakar', 'Kaffrine', 'Kaolack', 'Kedougou', 'Kolda', 'Matam', 'Saint Louis']

    lgDict = util.testresultsfiletotable(tbl_sen1, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))




    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})


    lgDict = lg.runlogistigate(lgDict)






    numSN, numTN = 46, 7





    return
