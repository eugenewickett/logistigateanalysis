# -*- coding: utf-8 -*-
'''
Script that analyzes a more complete version of the MQDB data, where more supply-chain features are associated with
each PMS data point.

'''

from logistigate.logistigate import utilities as util # Pull from the submodule "develop" branch
from logistigate.logistigate import methods
from logistigate.logistigate import lg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import scipy.stats as sps
import scipy.special as spsp
import random


def assignlabels(df, coltogroup, categorylist=[], thresh=90):
    '''
    Function that takes a pandas dataframe, a column name, an (optional) list of categories, and a [optional] score
    threshold, and return a pandas object with an added column of 'columnName_GROUPED' where each corresponding element
    of column name is set to the closest matching category, if a category list is provided and the match value exceeds
    the threshold, or to a representative element if a group of elements have a matching fuzzywuzzy score above the
    score threshold.
    '''
    import pandas as pd
    from fuzzywuzzy import process
    from fuzzywuzzy import fuzz

    if not isinstance(df, pd.DataFrame): # Do we have a DataFrame?
        print('Please enter a pandas DataFrame object.')
        return
    if not coltogroup in df.columns: # Does the column exist?
        print('Column ' + coltogroup + ' is not in the DataFrame.')
        return

    newcol_lst = []  # Initialize list to be turned into new column
    newcol_name = coltogroup + '_GROUPED'
    if not categorylist == []:
        # Group according to categoryList entries
        for currEntry in df[coltogroup].astype('string'):
            if pd.isnull(currEntry):
                bestmatch = currEntry
            else:
                currmatch = process.extractOne(currEntry, categorylist)
                if currmatch[1] < thresh: # Insufficient match; use filler entry
                    bestmatch = 'MANUALLY_MODIFY'
                else:
                    bestmatch = currmatch[0]
            newcol_lst.append(bestmatch)

    else:
        # Loop through each item and check if any preceding item matches more than the threshold value
        listtogroup = df[coltogroup].astype('string').fillna('NA VALUE').tolist()
        candmatch = [listtogroup[0],0.0] # Needed initialization
        for entryInd, currEntry in enumerate(listtogroup):
            #if pd.isnull(currEntry):
            #    newcol_lst.append(currEntry)
                #df.iloc[entryInd][newcol_name] = df.iloc[entryInd][coltogroup]
            #else:
            if not pd.isnull(currEntry):
                if entryInd > 0:
                    candmatch = process.extractOne(currEntry, listtogroup[:entryInd]) # Check previous entries
                if candmatch[1] > thresh:
                    bestInd = listtogroup.index(candmatch[0], 0, entryInd)
                    newcol_lst.append(newcol_lst[bestInd]) # Use whatever value the best match had
                    #df.iloc[entryInd][newcol_name] = candmatch[0]
                else:
                    newcol_lst.append(currEntry)
                    #df.iloc[entryInd][newcol_name] = df.iloc[entryInd][coltogroup]
            else:
                newcol_lst.append(currEntry)

    df[newcol_name] = newcol_lst

    return df

'''
Some checks of the function's performance

listtogroup[482]
MQD_df_CAM = assignlabels(MQD_df_CAM, 'Facility_Name')

# How many did we change?
counter = 0
for i in range(len(listtogroup)):
    if not MQD_df_CAM.iloc[i]['Facility_Name'] == MQD_df_CAM.iloc[i]['Facility_Name_GROUPED']:
        counter += 1
print(counter)
'''

'''
    Now check how the assignlabels() function did as compared with the done-by-hand process

    correct = 0
    wrong = 0
    nulls = 0
    for i in range(len(MQD_df_CAM['Facility_Location'])):
        if not pd.isnull(MQD_df_CAM.iloc[i]['Facility_Location_EDIT']):
            if MQD_df_CAM.iloc[i]['Facility_Location_EDIT'] ==  MQD_df_CAM.iloc[i]['Facility_Location_GROUPED']:
                correct += 1
            else:
                wrong += 1
                print(MQD_df_CAM.iloc[i]['Facility_Location_EDIT'], MQD_df_CAM.iloc[i]['Facility_Location_GROUPED'])
        else:
            nulls += 1

    (WRONG)-(CORRECT)-(NULL)
    CAMBODIA - FACILITY LOCATION: 158-880-1953 
    CAMBODIA - FACILITY NAME
    

    '''

def cleanMQD():
    '''
    Script that cleans up raw Medicines Quality Database data for use in logistigate.
    It reads in the CSV file 'MQDB_Master_Expanded.csv', with necessary columns 'Country_Name,' 'Province,'
    'Therapeutic Indication', 'Manufacturer,' 'Facility Type', 'Date Sample Collected', 'Final Test Result,' and
    'Type of Test', and returns a dictionary of objects to be formatted for use with logistigate.
    '''
    # Read in the raw database file
    import os
    import pandas as pd
    SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
    filesPath = os.path.join(SCRIPT_DIR, 'MQDfiles')
    MQD_df = pd.read_csv(os.path.join(filesPath,'MQDB_Master_Expanded2.csv'),low_memory=False) # Main raw database file

    #Change 'pass' to 'Pass'
    MQD_df.loc[MQD_df.Final_Test_Conclusion=='pass','Final_Test_Conclusion'] = 'Pass'

    #Drop 'Province', which has NULL values for some reason; keep 'Province_Name'
    MQD_df = MQD_df.drop('Province', axis=1)

    # Remove 'Guatemala', 'Bolivia', 'Colombia', 'Ecuador', 'Guyana', and 'Yunnan China' due to low sample size
    #dropCountries = ['Bolivia', 'Colombia', 'Ecuador', 'Guatemala', 'Guyana', 'Yunnan China']
    #MQD_df = MQD_df[~MQD_df['Country_Name'].isin(dropCountries)]

    # By collection year; need to add a year column
    MQD_df['Year_Sample_Collected'] = pd.DatetimeIndex(MQD_df['Date_Sample_Collected']).year
    MQD_df.pivot_table(index=['Country_Name','Year_Sample_Collected'], columns=['Final_Test_Conclusion'], aggfunc='size', fill_value=0)

    # Retitle 'Manufacturer_Name' to 'Manufacturer' to synthesize with old code
    MQD_df= MQD_df.rename(columns={'Manufacturer_Name':'Manufacturer'})

    # Put therapeutic indications into preset categories
    indicationsList = [
        'Analgesic',
        'Antiasthmatic',
        'Antibiotic',
        'Antidiabetic',
        'Antifungus',
        'Antihistamine',
        'Antiinflammatory',
        'Antimalarial',
        'Antipyretic/Analgesic',
        'Antiretroviral',
        'Antituberculosis',
        'Antiviral ',
        'Diarrhea',
        'Diuretic',
        'NA VALUE',
        'Preeclampsia/Eclampsia'
    ]
    MQD_df = assignlabels(MQD_df, 'Indication', indicationsList, thresh=95)
    # Manual adjustments
    MQD_df.loc[(MQD_df.Indication == '1') | (MQD_df.Indication == 'Missing') | (MQD_df.Indication == 'NA VALUE')
                | (MQD_df.Indication == 'Anthelmintic') | (MQD_df.Indication == 'Bronchodialator'),
                'Indication_GROUPED'] = 'NA VALUE'
    MQD_df.loc[(MQD_df.Indication == 'Postpartum hemorrhage') | (MQD_df.Indication == 'Preeclampsia/Eclampsia')
                | (MQD_df.Indication == 'Preeclampsia') | (MQD_df.Indication == 'Eclampsia'),
               'Indication_GROUPED'] = 'Preeclampsia/Eclampsia'
    MQD_df.loc[(MQD_df.Indication == 'ARV') | (MQD_df.Indication == 'Antiretroviral'),
               'Indication_GROUPED'] = 'Antiretroviral'
    MQD_df.loc[(MQD_df.Indication == 'Antihistamine') | (MQD_df.Indication == 'antihistamine (corticosteriod)'),
               'Indication_GROUPED'] = 'Antihistamine'
    MQD_df.loc[(MQD_df.Indication == 'Antipyretic') | (MQD_df.Indication == 'Antipyretic/Analgesic'),
               'Indication_GROUPED'] = 'Antipyretic/Analgesic'
    MQD_df.loc[(MQD_df.Indication == 'Analgesic & anti-inflammatory') | (MQD_df.Indication == 'Analgesic'),
               'Indication_GROUPED'] = 'Analgesic'
    MQD_df.loc[(MQD_df.Indication == 'Antimalarial') | (MQD_df.Indication == 'Antimalarials'),
               'Indication_GROUPED'] = 'Antimalarial'

    # Get data particular to each country of interest
    MQD_df_CAM = MQD_df[MQD_df['Country_Name'] == 'Cambodia'].copy()
    MQD_df_ETH = MQD_df[MQD_df['Country_Name'] == 'Ethiopia'].copy()
    MQD_df_GHA = MQD_df[MQD_df['Country_Name'] == 'Ghana'].copy()
    MQD_df_KEN = MQD_df[MQD_df['Country_Name'] == 'Kenya'].copy()
    MQD_df_LAO = MQD_df[MQD_df['Country_Name'] == 'Lao PDR'].copy()
    MQD_df_MOZ = MQD_df[MQD_df['Country_Name'] == 'Mozambique'].copy()
    MQD_df_PER = MQD_df[MQD_df['Country_Name'] == 'Peru'].copy()
    MQD_df_PHI = MQD_df[MQD_df['Country_Name'] == 'Philippines'].copy()
    MQD_df_SEN = MQD_df[MQD_df['Country_Name'] == 'Senegal'].copy()
    MQD_df_THA = MQD_df[MQD_df['Country_Name'] == 'Thailand'].copy()
    MQD_df_VIE = MQD_df[MQD_df['Country_Name'] == 'Viet Nam'].copy()
    #MQD_df_VIE.count()

    # COMMANDS USEFUL FOR SLICING THE DATAFRAME
    # MQD_df.keys()
    # pd.unique(MQD_df['Country_Name'])
    # MQD_df.pivot_table(index=['Country_Name'],columns=['Final_Test_Conclusion'],aggfunc='size',fill_value=0)

    # Consolidate typos or seemingly identical entries in significant categories

    # CAMBODIA PROCESSING
    # Province_Name
    templist = MQD_df_CAM['Province_Name'].tolist()
    MQD_df_CAM['Province_Name_GROUPED'] = templist
    MQD_df_CAM.loc[
        (MQD_df_CAM.Province_Name == 'Ratanakiri')
        | (MQD_df_CAM.Province_Name == 'Rattanakiri'), 'Province_Name_GROUPED'] = 'Ratanakiri'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Province_Name == 'Steung Treng')
        | (MQD_df_CAM.Province_Name == 'Stung Treng'), 'Province_Name_GROUPED'] = 'Stung Treng'

    templist = MQD_df_CAM['Manufacturer'].tolist()
    MQD_df_CAM['Manufacturer_GROUPED'] = templist
    # Manufacturer
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'AMN Life Science') | (MQD_df_CAM.Manufacturer == 'AMN Life Science Pvt Ltd'),
        'Manufacturer_GROUPED'] = 'AMN Life Science'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Acdhon Co., Ltd') | (MQD_df_CAM.Manufacturer == 'Acdhon Company Ltd'),
        'Manufacturer_GROUPED'] = 'Acdhon Co., Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Alembic Limited') | (MQD_df_CAM.Manufacturer == 'Alembic Pharmaceuticals Ltd'),
        'Manufacturer_GROUPED'] = 'Alembic Limited'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'ALICE PHARMA PVT LTD') | (MQD_df_CAM.Manufacturer == 'Alice Pharma Pvt.Ltd')
        | (MQD_df_CAM.Manufacturer == 'Alice Pharmaceuticals'), 'Manufacturer_GROUPED'] = 'Alice Pharmaceuticals'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Atoz Pharmaceutical Pvt.Ltd') | (MQD_df_CAM.Manufacturer == 'Atoz Pharmaceuticals Ltd'),
        'Manufacturer_GROUPED'] = 'Atoz Pharmaceuticals Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Aurobindo Pharma LTD') | (MQD_df_CAM.Manufacturer == 'Aurobindo Pharma Ltd.')
        | (MQD_df_CAM.Manufacturer == 'Aurobindo Pharmaceuticals Ltd'), 'Manufacturer_GROUPED'] = 'Aurobindo'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Aventis') | (MQD_df_CAM.Manufacturer == 'Aventis Pharma Specialite'),
        'Manufacturer_GROUPED'] = 'Aventis'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Bailly Creat') | (MQD_df_CAM.Manufacturer == 'Laboratoire BAILLY- CREAT'),
        'Manufacturer_GROUPED'] = 'Bailly Creat'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Bright Future Laboratories') | (MQD_df_CAM.Manufacturer == 'Bright Future Pharma'),
        'Manufacturer_GROUPED'] = 'Bright Future Laboratories'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Burapha') | (MQD_df_CAM.Manufacturer == 'Burapha Dispensary Co, Ltd'),
        'Manufacturer_GROUPED'] = 'Burapha'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'CHANKIT') | (MQD_df_CAM.Manufacturer == 'Chankit Trading Ltd')
        | (MQD_df_CAM.Manufacturer == 'Chankit trading Ltd, Part'),
        'Manufacturer_GROUPED'] = 'Chankit Trading Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Chea Chamnan Laboratoire Co., LTD') | (MQD_df_CAM.Manufacturer == 'Chea Chamnan Laboratories Co., Ltd')
        | (MQD_df_CAM.Manufacturer == 'Chea Chamnan Laboratory Company Ltd'),
        'Manufacturer_GROUPED'] = 'Chea Chamnan Laboratory Company Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Cipla Ltd.') | (MQD_df_CAM.Manufacturer == 'Cipla Ltd'),
        'Manufacturer_GROUPED'] = 'Cipla Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'DOMESCO MEDICAL IMP EXP JOINT STOCK CORP')
        | (MQD_df_CAM.Manufacturer == 'DOMESCO MEDICAL IMP EXP JOINT_stock corp')
        | (MQD_df_CAM.Manufacturer == 'DOMESCO MEDICAL IMPORT EXPORT JOINT STOCK CORP')
        | (MQD_df_CAM.Manufacturer == 'Domesco'),
        'Manufacturer_GROUPED'] = 'Domesco'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Emcure Pharmaceutical') | (MQD_df_CAM.Manufacturer == 'Emcure'),
        'Manufacturer_GROUPED'] = 'Emcure'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Eurolife Healthcare Pvt Ltd') | (MQD_df_CAM.Manufacturer == 'Eurolife'),
        'Manufacturer_GROUPED'] = 'Eurolife'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Fabrique par Modo Plan')
        | (MQD_df_CAM.Manufacturer == 'Fabrique'),
        'Manufacturer_GROUPED'] = 'Fabrique'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Flamingo Pharmaceutical Limited') | (MQD_df_CAM.Manufacturer == 'Flamingo Pharmaceuticals Ltd'),
        'Manufacturer_GROUPED'] = 'Flamingo Pharmaceuticals Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Glenmark Pharmaceuticals Ltd') |
        (MQD_df_CAM.Manufacturer == 'Glenmark Pharmaceuticals Ltd.'),
        'Manufacturer_GROUPED'] = 'Glenmark Pharmaceuticals Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Global Pharma Health care PVT-LTD')
        | (MQD_df_CAM.Manufacturer == 'GlobalPharma Healthcare Pvt-Ltd')
        | (MQD_df_CAM.Manufacturer == 'Global Pharma'),
        'Manufacturer_GROUPED'] = 'Global Pharma'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Gracure Pharmaceuticals Ltd.') | (MQD_df_CAM.Manufacturer == 'Gracure Pharmaceuticals'),
        'Manufacturer_GROUPED'] = 'Gracure Pharmaceuticals'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Il Dong Pharmaceutical Company Ltd') | (MQD_df_CAM.Manufacturer == 'Il Dong Pharmaceuticals Ltd'),
        'Manufacturer_GROUPED'] = 'Il Dong Pharmaceuticals Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Khandelwal Laboratories Ltd')
        | (MQD_df_CAM.Manufacturer == 'Khandewal Lab')
        | (MQD_df_CAM.Manufacturer == 'Khandelwal'),
        'Manufacturer_GROUPED'] = 'Khandelwal'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Laboratories EPHAC Co., Ltd')
        | (MQD_df_CAM.Manufacturer == 'EPHAC Laboratories Ltd'),
        'Manufacturer_GROUPED'] = 'Laboratories EPHAC Co., Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Lyka Laboratories Ltd')
        | (MQD_df_CAM.Manufacturer == 'Lyka Labs Limited.')
        | (MQD_df_CAM.Manufacturer == 'Lyka Labs'),
        'Manufacturer_GROUPED'] = 'Lyka Labs'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Marksans Pharmaceuticals Ltd') | (MQD_df_CAM.Manufacturer == 'Marksans Pharma Ltd.')
        | (MQD_df_CAM.Manufacturer == 'Marksans Pharma Ltd.,'),
        'Manufacturer_GROUPED'] = 'Marksans Pharma Ltd.'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'MASALAB') | (MQD_df_CAM.Manufacturer == 'Masa Lab Co., Ltd'),
        'Manufacturer_GROUPED'] = 'Masa Lab Co., Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Medical Supply Pharmaceutical Enterprise')
        | (MQD_df_CAM.Manufacturer == 'Medical Supply Pharmaceutical Enteprise'),
        'Manufacturer_GROUPED'] = 'Medical Supply Pharmaceutical Enterprise'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Medley Pharmaceutical') |
        (MQD_df_CAM.Manufacturer == 'Medley Pharmaceuticals') |
        (MQD_df_CAM.Manufacturer == 'Medley Pharmaceuticals Ltd'),
        'Manufacturer_GROUPED'] = 'Medley Pharmaceuticals'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Medopharm Pvt. Ltd.')
        | (MQD_df_CAM.Manufacturer == 'Medopharm'),
        'Manufacturer_GROUPED'] = 'Medopharm'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Micro Laboratories Ltd') | (MQD_df_CAM.Manufacturer == 'MICRO LAB LIMITED')
        | (MQD_df_CAM.Manufacturer == 'Micro Labs Ltd') | (MQD_df_CAM.Manufacturer == 'Microlabs Limited'),
        'Manufacturer_GROUPED'] = 'Microlabs'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Millimed Co., Ltd Thailand')
        | (MQD_df_CAM.Manufacturer == 'Millimed'),
        'Manufacturer_GROUPED'] = 'Millimed'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Orchid Health Care') | (MQD_df_CAM.Manufacturer == 'Orchid Health'),
        'Manufacturer_GROUPED'] = 'Orchid Health'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Osoth Inter Laboratory Co., LTD') | (MQD_df_CAM.Manufacturer == 'Osoth Inter Laboratories'),
        'Manufacturer_GROUPED'] = 'Osoth Inter Laboratories'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'PHARMASANT LABORATORIES Co.,LTD') | (MQD_df_CAM.Manufacturer == 'Pharmasant Laboratories Co., Ltd'),
        'Manufacturer_GROUPED'] = 'Pharmasant Laboratories Co., Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Plethico Pharmaceuticals, Ltd')
        | (MQD_df_CAM.Manufacturer == 'Plethico Pharmaceuticals Ltd')
        | (MQD_df_CAM.Manufacturer == 'Plethico Pharmaceutical Ltd')
        | (MQD_df_CAM.Manufacturer == 'Plethico'),
        'Manufacturer_GROUPED'] = 'Plethico'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'PPM Laboratory') | (MQD_df_CAM.Manufacturer == 'PPM')
        | (MQD_df_CAM.Manufacturer == 'Pharma Product Manufacturing'),
        'Manufacturer_GROUPED'] = 'PPM'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Laboratories Pvt. Ltd') |
        (MQD_df_CAM.Manufacturer == 'PVT Laboratories Ltd'),
        'Manufacturer_GROUPED'] = 'PVT Laboratories Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Ranbaxy Laboratories Limited.')
        | (MQD_df_CAM.Manufacturer == 'Ranbaxy Pharmaceuticals'),
        'Manufacturer_GROUPED'] = 'Ranbaxy Pharmaceuticals'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Shijiazhuang Pharma Group Zhongnuo Pharmaceutical [Shijiazhuang] Co.,LTD')
        | (MQD_df_CAM.Manufacturer == 'Shijiazhuang Pharmaceutical Group Ltd'),
        'Manufacturer_GROUPED'] = 'Shijiazhuang Pharmaceutical Group Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Sanofi-Aventis Vietnam') | (MQD_df_CAM.Manufacturer == 'Sanofi Aventis'),
        'Manufacturer_GROUPED'] = 'Sanofi Aventis'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Stada Vietnam Joint Venture Co., Ltd.') | (MQD_df_CAM.Manufacturer == 'Stada Vietnam Joint Venture'),
        'Manufacturer_GROUPED'] = 'Stada Vietnam Joint Venture'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Shandong Reyoung Pharmaceutical Co., Ltd') | (
                    MQD_df_CAM.Manufacturer == 'Shandong Reyoung Pharmaceuticals Ltd'),
        'Manufacturer_GROUPED'] = 'Shandong Reyoung Pharmaceuticals Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'T Man Pharma Ltd. Part.')
        | (MQD_df_CAM.Manufacturer == 'T-MAN Pharma Ltd., Part')
        | (MQD_df_CAM.Manufacturer == 'T-Man Pharmaceuticals Ltd'),
        'Manufacturer_GROUPED'] = 'T-Man Pharmaceuticals Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Umedica Laboratories PVT. LTD.')
        | (MQD_df_CAM.Manufacturer == 'Umedica Laboratories PVT. Ltd')
        | (MQD_df_CAM.Manufacturer == 'Umedica Laboratories Pvt Ltd')
        | (MQD_df_CAM.Manufacturer == 'Umedica'),
        'Manufacturer_GROUPED'] = 'Umedica'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Utopian Co,.LTD') | (MQD_df_CAM.Manufacturer == 'Utopian Co., Ltd')
        | (MQD_df_CAM.Manufacturer == 'Utopian Company Ltd'),
        'Manufacturer_GROUPED'] = 'Utopian Company Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Vesco Pharmaceutical Ltd.,Part')
        | (MQD_df_CAM.Manufacturer == 'Vesco Pharmaceutical Ltd Part'),
        'Manufacturer_GROUPED'] = 'Vesco Pharmaceutical Ltd Part'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Yanzhou Xier Kangtai Pharmaceutical Co., Ltd')
        | (MQD_df_CAM.Manufacturer == 'Yanzhou Xier Kangtai Pharm'),
        'Manufacturer_GROUPED'] = 'Yanzhou Xier Kangtai Pharm'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Zhangjiakou DongFang pharmaceutical Co., Ltd')
        | (MQD_df_CAM.Manufacturer == 'Zhangjiakou Dongfang Phamaceutical'),
        'Manufacturer_GROUPED'] = 'Zhangjiakou Dongfang Phamaceutical'

    # Facility_Location
    # Designated Cambodia districts for clustering:
    distNames_CAM = ['Anlong Veng District',
    'Bakan District',
    'Banlung District',
    'Battambang City',
    'Borkeo District',
    'Cham Knan District',
    'Chamroeun District',
    'Chbamon District',
    'Chheb District',
    'Chom Ksan District',
    'Choran Ksan District',
    'Dongtong Market',
    'Kampong Bay District',
    'Kampong Cham District',
    'Kampong Siam District',
    'Kampong Thmor Market',
    'Kampong Thom Capital',
    'Kampong Trach District',
    'Keo Seima District',
    'Koh Kong District',
    'Kolen District',
    'Krakor District',
    'Kratie District',
    'Maung Russei District',
    'Memot District',
    'Missing',
    'O Tavao District',
    'Oyadav District',
    'Pailin City',
    'Peamror District',
    'Pearing District',
    'Phnom Kravanh District',
    'Phnom Preal District',
    'Ponhea Krek District',
    'Posat City',
    'Preah Vihear Town',
    'Prey Chhor District',
    'Prey Veng District',
    'Pursat City',
    'Roveahek District',
    'Rovieng District',
    'Sala Krau District',
    'Sampov Meas District',
    'Samraong District',
    'Sangkum Thmei District',
    'Senmonorom City',
    'Smach Mean Chey District',
    'Sre Ambel District',
    'Srey Santhor District',
    'Suong City',
    'Svay Antor District',
    'Svay Chrom District',
    'Svay Rieng District',
    'Takeo Capital',
    'Trapeang Prasat District',
    'Van Sai District']
    MQD_df_CAM = assignlabels(MQD_df_CAM, 'Facility_Location', categorylist=distNames_CAM)

    # Manually changed entries
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Anlong Veng District')
        | (MQD_df_CAM.Facility_Location == 'O Chugnean Village, Anglong Veng Commune, Anglong Veng District. Phone: 012 297 224')
        | (MQD_df_CAM.Facility_Location == 'O Chugnean Village, Anlong Veng Commune, Anlong Veng District. Tel. 012 429 643')
        | (MQD_df_CAM.Facility_Location == 'O Chungchean Village, Anlong Veng Commune, Anlong Veng District')
        | (MQD_df_CAM.Facility_Location == "O'Chungchean Village, Anlong Veng Commune, Anlong Veng District")
        | (MQD_df_CAM.Facility_Location == "O'Chungchean Village, Anlong Veng Commune, Anlong Veng District")
        | (MQD_df_CAM.Facility_Location == "O'Chungchean Village, Anlong Veng Commune, Anlong Veng District")
        | (MQD_df_CAM.Facility_Location == "O'Chungchean Village, Anlong Veng Commune, Anlong Veng District"),
        'Facility_Location_GROUPED'] = 'Anlong Veng District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Bakan District')
        | (MQD_df_CAM.Facility_Location == ''),
        'Facility_Location_GROUPED'] = 'Bakan District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Banlung District')
        | (MQD_df_CAM.Facility_Location == 'Banlung')
        | (MQD_df_CAM.Facility_Location == 'Banllung District')
        | (MQD_df_CAM.Facility_Location == 'Banlung City')
        | (MQD_df_CAM.Facility_Location == 'Banlung Market')
        | (MQD_df_CAM.Facility_Location == 'Bor Keo, Banlung District')
        | (MQD_df_CAM.Facility_Location == 'Andong Meas, Banlung')
        | (MQD_df_CAM.Facility_Location == "O'yadav, Banlung")
        | (MQD_df_CAM.Facility_Location == "O'yadav, Banlung District")
        | (MQD_df_CAM.Facility_Location == 'Street #78, Banlung City')
        | (MQD_df_CAM.Facility_Location == 'near Banlung Market')
        | (MQD_df_CAM.Facility_Location == 'Srok Ban lung'),
        'Facility_Location_GROUPED'] = 'Banlung District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Battambang City')
        | (MQD_df_CAM.Facility_Location == 'Battambang city')
        | (MQD_df_CAM.Facility_Location == 'Maung Reussy Dist. Battambang province'),
        'Facility_Location_GROUPED'] = 'Battambang City'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Borkeo District')
        | (MQD_df_CAM.Facility_Location == 'Borkeo district')
        | (MQD_df_CAM.Facility_Location == 'Cabinet-Keo Akara, near Borkeo Market')
        | (MQD_df_CAM.Facility_Location == 'Midwife- Saren, near Borkeo Market'),
        'Facility_Location_GROUPED'] = 'Borkeo District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Cham Knan District')
        | (MQD_df_CAM.Facility_Location == ''),
        'Facility_Location_GROUPED'] = 'Cham Knan District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Chamroeun District')
        | (MQD_df_CAM.Facility_Location == 'Cham Roeun District'),
        'Facility_Location_GROUPED'] = 'Chamroeun District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'National Road No.4, Chbamon district')
        | (MQD_df_CAM.Facility_Location == 'Chbamon District')
        | (MQD_df_CAM.Facility_Location == 'No. 3B, Peanich Kam village, Roka Thom commune, Chbamon district')
        | (MQD_df_CAM.Facility_Location == 'Peanichkam village, Roka Thom commune, Chbamon District')
        | (MQD_df_CAM.Facility_Location == 'Roka Thom Commune, Chbamon District')
        | (MQD_df_CAM.Facility_Location == '#01D, Psar Kampong Speu, Chbamon district'),
        'Facility_Location_GROUPED'] = 'Chbamon District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Chheb District')
        | (MQD_df_CAM.Facility_Location == ''),
        'Facility_Location_GROUPED'] = 'Chheb District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Chom Ksan District')
        | (MQD_df_CAM.Facility_Location == 'O Chhounh Village, Chom Ksan district')
        | (MQD_df_CAM.Facility_Location == 'Sra Em village, Chom Ksan District'),
        'Facility_Location_GROUPED'] = 'Chom Ksan District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Choran Ksan District')
        | (MQD_df_CAM.Facility_Location == ''),
        'Facility_Location_GROUPED'] = 'Choran Ksan District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Dongtong District')
        | (MQD_df_CAM.Facility_Location == 'Dongtong Market')
        | (MQD_df_CAM.Facility_Location == 'No. 50, South of Dongtong Market'),
        'Facility_Location_GROUPED'] = 'Dongtong District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Kampong Bay District')
        | (MQD_df_CAM.Facility_Location == 'No. 93, St. 3, Kampong Bay district')
        | (MQD_df_CAM.Facility_Location == 'No. 5, St. 3, Kampong Bay district')
        | (MQD_df_CAM.Facility_Location == 'Kampong Bay district')
        | (MQD_df_CAM.Facility_Location == '#79, Kampong Bay district')
        | (MQD_df_CAM.Facility_Location == '#16, St. 7 Makara, Kandal village, Kampong Bay district'),
        'Facility_Location_GROUPED'] = 'Kampong Bay District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Kampong Cham City')
        | (MQD_df_CAM.Facility_Location == 'Kampong Cham District')
        | (MQD_df_CAM.Facility_Location == 'Memot Market, Kampong Cham')
        | (MQD_df_CAM.Facility_Location == 'Steung Market, Kampong Cham')
        | (MQD_df_CAM.Facility_Location == 'Street Preah Bath Ang Duong (East Phsar Thom), Kampong Cham')
        | (MQD_df_CAM.Facility_Location == 'Street Preah Bath Ang Duong (Near Kosona Bridge), Kampong Cham'),
        'Facility_Location_GROUPED'] = 'Kampong Cham District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Kampong Siam District')
        | (MQD_df_CAM.Facility_Location == 'Kampong Siam district'),
        'Facility_Location_GROUPED'] = 'Kampong Siam District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Kampong Thmor Market')
        | (MQD_df_CAM.Facility_Location == '#66, Tral village, Kompong Thmor market')
        | (MQD_df_CAM.Facility_Location == '66, Tral village, Kompong Thmor market')
        | (MQD_df_CAM.Facility_Location == 'No. 3, Rd 6A, Kampong Thmor'),
        'Facility_Location_GROUPED'] = 'Kampong Thmor Market'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Kampong Thom Capital')
        | (MQD_df_CAM.Facility_Location == 'Kampong Thom Market, Kampong Thom capital')
        | (MQD_df_CAM.Facility_Location == 'No. 15Eo, Kampong Thom market, Kampong Thom capital')
        | (MQD_df_CAM.Facility_Location == 'No. 43, Rd No. 6, Kampong Thom capital')
        | (MQD_df_CAM.Facility_Location == 'No. 9 Eo, Kampong Thom Market, Kampong Thom capital')
        | (MQD_df_CAM.Facility_Location == 'No.45, Rd No. 6, Kampong Thom capital'),
        'Facility_Location_GROUPED'] = 'Kampong Thom Capital'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Kampong Trach District')
        | (MQD_df_CAM.Facility_Location == 'Kampong Trach Village, Kampong Trach district'),
        'Facility_Location_GROUPED'] = 'Kampong Trach District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Keo Seima District')
        | (MQD_df_CAM.Facility_Location == 'Keoseima District')
        | (MQD_df_CAM.Facility_Location == 'Keoseima district')
        | (MQD_df_CAM.Facility_Location == 'Keosema District')
        | (MQD_df_CAM.Facility_Location == "Khum Sre Kh'tob, Keo Seima district"),
        'Facility_Location_GROUPED'] = 'Keo Seima District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Koh Kong District')
        | (MQD_df_CAM.Facility_Location == 'Koh Kong Province')
        | (MQD_df_CAM.Facility_Location == 'Kohk Kong Capital')
        | (MQD_df_CAM.Facility_Location == "Pum trorpeagh , Sre'ambel  ,  koh kong  province."),
        'Facility_Location_GROUPED'] = 'Koh Kong District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Kolen District')
        | (MQD_df_CAM.Facility_Location == 'Sro Yang Village, Sro Yang Commune, Kolen District'),
        'Facility_Location_GROUPED'] = 'Kolen District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Krakor District')
        | (MQD_df_CAM.Facility_Location == 'Chheutom Commune, Krakor District'),
        'Facility_Location_GROUPED'] = 'Krakor District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Kratie District')
        | (MQD_df_CAM.Facility_Location == 'Kratie commune, Kratie Distrist, Kratie'),
        'Facility_Location_GROUPED'] = 'Kratie District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Maung Russei District')
        | (MQD_df_CAM.Facility_Location == ''),
        'Facility_Location_GROUPED'] = 'Maung Russei District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Memot District')
        | (MQD_df_CAM.Facility_Location == 'Khum Dar, Memot District')
        | (MQD_df_CAM.Facility_Location == 'OD Memut'),
        'Facility_Location_GROUPED'] = 'Memot District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'O Tavao District')
        | (MQD_df_CAM.Facility_Location == "Krachab, O'Tavao District")
        | (MQD_df_CAM.Facility_Location == "Krachab, O'Tavao, Pailin"),
        'Facility_Location_GROUPED'] = 'O Tavao District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Oyadav District')
        | (MQD_df_CAM.Facility_Location == ''),
        'Facility_Location_GROUPED'] = 'Oyadav District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Pailin City')
        | (MQD_df_CAM.Facility_Location == 'O Ta Puk Leu')
        | (MQD_df_CAM.Facility_Location == 'Pailin City')
        | (MQD_df_CAM.Facility_Location == 'Pailin Ville')
        | (MQD_df_CAM.Facility_Location == 'Pang Rolim, O Ta Vao, Pailin City')
        | (MQD_df_CAM.Facility_Location == 'O Tapok Leu, Palin City')
        | (MQD_df_CAM.Facility_Location == 'O Tapokleu, Pahy Market')
        | (MQD_df_CAM.Facility_Location == 'O Tapuk Leu, Pailen City')
        | (MQD_df_CAM.Facility_Location == 'O Tapuk Leu, Pailin City')
        | (MQD_df_CAM.Facility_Location == 'O. Tapok Leu, Pahy Market')
        | (MQD_df_CAM.Facility_Location == 'Opeut Village, Pailin. Tel. 017 492909')
        | (MQD_df_CAM.Facility_Location == 'Pahee market, Pailin, Tel: 089 579829')
        | (MQD_df_CAM.Facility_Location == 'Phsar Pahi, Pailin City')
        | (MQD_df_CAM.Facility_Location == 'Phsar Pahi, Pailin City'),
        'Facility_Location_GROUPED'] = 'Pailin City'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Peamror District')
        | (MQD_df_CAM.Facility_Location == '#309, Prek Khsay commune, Peamror district')
        | (MQD_df_CAM.Facility_Location == 'National Road N0.1, Prek Khsay commune, Peamror district'),
        'Facility_Location_GROUPED'] = 'Peamror District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Pearing District')
        | (MQD_df_CAM.Facility_Location == 'St. 8A. Roka commune, Pearing district'),
        'Facility_Location_GROUPED'] = 'Pearing District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Phnom Kravanh District')
        | (MQD_df_CAM.Facility_Location == 'Leach village, Phnom Kravanh district')
        | (MQD_df_CAM.Facility_Location == 'Phnom Kravanh District')
        | (MQD_df_CAM.Facility_Location == 'Phnom Krovanh District'),
        'Facility_Location_GROUPED'] = 'Phnom Kravanh District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Phnom Preal District')
        | (MQD_df_CAM.Facility_Location == 'Kondamrey - Phnom Preal')
        | (MQD_df_CAM.Facility_Location == 'Koun Domrey, Phnom Preal District')
        | (MQD_df_CAM.Facility_Location == 'O dontaleu Phnom Preal District')
        | (MQD_df_CAM.Facility_Location == 'Phnom Preal')
        | (MQD_df_CAM.Facility_Location == 'Phnom Preal District'),
        'Facility_Location_GROUPED'] = 'Phnom Preal District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Ponhea Krek District')
        | (MQD_df_CAM.Facility_Location == 'Ponhea Krek District'),
        'Facility_Location_GROUPED'] = 'Ponhea Krek District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Posat City')
        | (MQD_df_CAM.Facility_Location == 'Peal Nhek 2, Posat City'),
        'Facility_Location_GROUPED'] = 'Posat City'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Preah Vihear Town')
        | (MQD_df_CAM.Facility_Location == 'Preah Vihear Province')
        | (MQD_df_CAM.Facility_Location == 'Preah Vihear Town')
        | (MQD_df_CAM.Facility_Location == 'Preah Vihear Town')
        | (MQD_df_CAM.Facility_Location == 'Preah Vihear Town'),
        'Facility_Location_GROUPED'] = 'Preah Vihear Town'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Prey Chhor District')
        | (MQD_df_CAM.Facility_Location == 'OD Prey Chhor')
        | (MQD_df_CAM.Facility_Location == 'Phsar Prey Toteng, Prey Chhor District'),
        'Facility_Location_GROUPED'] = 'Prey Chhor District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Prey Veng District')
        | (MQD_df_CAM.Facility_Location == '#26A, St.15, Kampong Leav commune, Prey Veng district')
        | (MQD_df_CAM.Facility_Location == '#36, St. 15, Kampong Leav commune, Prey Veng district'),
        'Facility_Location_GROUPED'] = 'Prey Veng District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Pursat City')
        | (MQD_df_CAM.Facility_Location == 'Peal Nhek 2, Pursat City')
        | (MQD_df_CAM.Facility_Location == 'Phum Piel Nhek, Pursat')
        | (MQD_df_CAM.Facility_Location == 'Village Peal Nhek 2, Pursat City'),
        'Facility_Location_GROUPED'] = 'Pursat City'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Roveahek District')
        | (MQD_df_CAM.Facility_Location == 'Angkor Prosre commune, Roveahek district')
        | (MQD_df_CAM.Facility_Location == 'Kampong Trach commune, Roveahek district'),
        'Facility_Location_GROUPED'] = 'Roveahek District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Rovieng District')
        | (MQD_df_CAM.Facility_Location == 'Roveing District')
        | (MQD_df_CAM.Facility_Location == 'Ro Vieng District, Tel.: 012 24 82 65')
        | (MQD_df_CAM.Facility_Location == 'Ro Vieng District')
        | (MQD_df_CAM.Facility_Location == 'Ro Veing District')
        | (MQD_df_CAM.Facility_Location == 'Rovieng District')
        | (MQD_df_CAM.Facility_Location == 'Rovieng District'),
        'Facility_Location_GROUPED'] = 'Rovieng District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Salakrav District')
        | (MQD_df_CAM.Facility_Location == 'Salakrav, Pailin tel.:097 555653')
        | (MQD_df_CAM.Facility_Location == 'Salakrao District, Pailin,')
        | (MQD_df_CAM.Facility_Location == 'Stoeung Kach,Sala Krao District')
        | (MQD_df_CAM.Facility_Location == 'Steng Trong, Sala Krao District')
        | (MQD_df_CAM.Facility_Location == 'Sala krao District')
        | (MQD_df_CAM.Facility_Location == 'Sala Krao District')
        | (MQD_df_CAM.Facility_Location == 'Sala Krau District')
        | (MQD_df_CAM.Facility_Location == 'Prom market, Salakrav, Pailin. Tel.: 055 6900987')
        | (MQD_df_CAM.Facility_Location == 'Prom market, Salakrav, Pailin Tel. 097 678936')
        | (MQD_df_CAM.Facility_Location == 'Phsar Prom stoeung Kach, Salakrao District')
        | (MQD_df_CAM.Facility_Location == 'Phsar  Prum, Stoeung Kach,Sala Krao District')
        | (MQD_df_CAM.Facility_Location == 'Phnom Preal, Sala Krao District')
        | (MQD_df_CAM.Facility_Location == 'Phnom Preal village, Salakrao District')
        | (MQD_df_CAM.Facility_Location == 'Phnom Preal Village, Salakrao District,')
        | (MQD_df_CAM.Facility_Location == 'Phnom Kuy, Salakrav, Pailin, Tel 011 911 293')
        | (MQD_df_CAM.Facility_Location == 'Phnom Koy,Sala krao District')
        | (MQD_df_CAM.Facility_Location == 'Phnom Koy Village, Salakrao')
        | (MQD_df_CAM.Facility_Location == 'O donta krom, Khom Steung Trong, Salakrav, Pailin. Tel.: 012 27 44 65')
        | (MQD_df_CAM.Facility_Location == 'Kondamrey Village, Phnom Preal, Salakrao District')
        | (MQD_df_CAM.Facility_Location == 'Kondamrey Village, Phnom Preal, Salakrao District')
        | (MQD_df_CAM.Facility_Location == 'Kondamrey Commune, Phnom Preal Dsitrict')
        | (MQD_df_CAM.Facility_Location == 'Phnom Koy Village, Salakrao')
        | (MQD_df_CAM.Facility_Location == 'Phnom Preal Village, Salakrao District,')
        | (MQD_df_CAM.Facility_Location == 'Phnom Preal village, Salakrao District'),
        'Facility_Location_GROUPED'] = 'Sala Krau District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Sampov Meas District')
        | (MQD_df_CAM.Facility_Location == ''),
        'Facility_Location_GROUPED'] = 'Sampov Meas District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Chhouk Village, Samrong Commune, Samrong City')
        | (MQD_df_CAM.Facility_Location == 'Chhouk Village, Samrong Commune, Samrong City')
        | (MQD_df_CAM.Facility_Location == 'Chhouk Village, Samrong Commune, Samrong City Tel. 012 983334')
        | (MQD_df_CAM.Facility_Location == 'Chhouk Village, Samrong Commune, Samrong City Tel. 012 983335')
        | (MQD_df_CAM.Facility_Location == 'Chhouk Village, Samrong Commune, Samrong City Tel. 012 983336')
        | (MQD_df_CAM.Facility_Location == 'Chhouk Village, Samrong Commune, Samrong City Tel. 012 983337')
        | (MQD_df_CAM.Facility_Location == 'Chhouk Village, Samrong Commune, Samrong City Tel. 012 983338')
        | (MQD_df_CAM.Facility_Location == 'National Road No. 2, Samrong district, Khum Lom Chang')
        | (MQD_df_CAM.Facility_Location == 'O smach Village, O Smach Commune, Samrong District')
        | (MQD_df_CAM.Facility_Location == 'Samrong City')
        | (MQD_df_CAM.Facility_Location == 'Samrong District')
        | (MQD_df_CAM.Facility_Location == 'Samrong Village, Samrong Commune, Samrong City')
        | (MQD_df_CAM.Facility_Location == 'Samraong City')
        | (MQD_df_CAM.Facility_Location == 'Samrong Village, Samrong Commune, Samrong City')
        | (MQD_df_CAM.Facility_Location == 'Samrong Village, Samrong Commune, Samrong City')
        | (MQD_df_CAM.Facility_Location == 'Samrong Village, Samrong Commune, samrong City')
        | (MQD_df_CAM.Facility_Location == 'Phum Thmey, Roveang, Sam Rong district'),
        'Facility_Location_GROUPED'] = 'Samraong District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Sangkum Thmei District')
        | (MQD_df_CAM.Facility_Location == 'Sangkomthmey District')
        | (MQD_df_CAM.Facility_Location == 'Sangkum Thmei District')
        | (MQD_df_CAM.Facility_Location == 'Sangkomthmey District, Tel: 011 56 99 26')
        | (MQD_df_CAM.Facility_Location == 'Sangkom Thmei District, Tel.: 011 56 99 26'),
        'Facility_Location_GROUPED'] = 'Sangkum Thmei District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Senmonorom City')
        | (MQD_df_CAM.Facility_Location == 'Sangkat Speanmeanchey, Senmonorom City')
        | (MQD_df_CAM.Facility_Location == 'Senmonorom District')
        | (MQD_df_CAM.Facility_Location == 'Senmonorom district'),
        'Facility_Location_GROUPED'] = 'Senmonorom City'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Smach Mean Chey District')
        | (MQD_df_CAM.Facility_Location == ''),
        'Facility_Location_GROUPED'] = 'Smach Mean Chey District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Sre Ambel District')
        | (MQD_df_CAM.Facility_Location == 'Sre Ambel'),
        'Facility_Location_GROUPED'] = 'Sre Ambel District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Village Prek Por Krom, Prek Por Commune, Srey Santhor District')
        | (MQD_df_CAM.Facility_Location == 'Srey Santhor District, Kampong Cham')
        | (MQD_df_CAM.Facility_Location == 'Srey Santhor District')
        | (MQD_df_CAM.Facility_Location == 'Srey Santhor District')
        | (MQD_df_CAM.Facility_Location == 'Prek Por Commune, Srey Santhor District, ')
        | (MQD_df_CAM.Facility_Location == 'Rokar Village, Prek Por Commune, Srey Santhor District'),
        'Facility_Location_GROUPED'] = 'Srey Santhor District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Steung Treng Downtown, Tel: 092 251125')
        | (MQD_df_CAM.Facility_Location == 'Steung Treng Downtown, Tel: 092 958707')
        | (MQD_df_CAM.Facility_Location == 'Steung Treng Downtown, Tel: 097 9822096')
        | (MQD_df_CAM.Facility_Location == 'Steung Treng Downtown, Tel:011252525')
        | (MQD_df_CAM.Facility_Location == 'Steung Treng down town')
        | (MQD_df_CAM.Facility_Location == 'Steung Treng downtown, Tel: 017 808287')
        | (MQD_df_CAM.Facility_Location == 'Steung Treng downtown, tel.: 099906174')
        | (MQD_df_CAM.Facility_Location == 'Steung Treng downtown,Tel: 097 90 43 071'),
        'Facility_Location_GROUPED'] = 'Steung Treng Downtown'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Suong City')
        | (MQD_df_CAM.Facility_Location == '#65, National Road 7, Soung Commune, Suong City'),
        'Facility_Location_GROUPED'] = 'Suong City'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Svay Antor District')
        | (MQD_df_CAM.Facility_Location == 'Pich Chenda commune,Svay Antor district')
        | (MQD_df_CAM.Facility_Location == 'Svay Antor commune, Svay Antor district'),
        'Facility_Location_GROUPED'] = 'Svay Antor District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Svay Chrom District')
        | (MQD_df_CAM.Facility_Location == 'National Road No. 1, Crol Kor commune, Svay Chrom district'),
        'Facility_Location_GROUPED'] = 'Svay Chrom District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Svay Rieng District')
        | (MQD_df_CAM.Facility_Location == '# 111, Svay Rieng capital')
        | (MQD_df_CAM.Facility_Location == '#1, Rd. 6, Svay Rieng Commune, Svay Rieng district')
        | (MQD_df_CAM.Facility_Location == '#5, Veal Yun Market, Svay Rieng capital')
        | (MQD_df_CAM.Facility_Location == 'Svay Rieng Province')
        | (MQD_df_CAM.Facility_Location == 'Veal Yun market, Svay Rieng capital')
        | (MQD_df_CAM.Facility_Location == 'Svay Rieng District'),
        'Facility_Location_GROUPED'] = 'Svay Rieng District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Takeo Capital')
        | (MQD_df_CAM.Facility_Location == 'No. 01, Khum Rokaknong, St. 8, Takeo capital')
        | (MQD_df_CAM.Facility_Location == 'No. 02, Sangkat Rokaknong, St. 8, Takeo capital')
        | (MQD_df_CAM.Facility_Location == 'No. 146, St. 2, Takeo capital')
        | (MQD_df_CAM.Facility_Location == 'No. 2, St. 23, Takeo Capital market')
        | (MQD_df_CAM.Facility_Location == 'No. 215, St. 28, Corneer of market of Takeo capital')
        | (MQD_df_CAM.Facility_Location == 'No. 215, St. 28, Corner of market of Takeo capital')
        | (MQD_df_CAM.Facility_Location == 'No. 5, St. 2, Khum Rokaknong, Takeo capital')
        | (MQD_df_CAM.Facility_Location == 'Takeo capital market'),
        'Facility_Location_GROUPED'] = 'Takeo Capital'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Trapaing Prasat District')
        | (MQD_df_CAM.Facility_Location == 'Trapaing Prasat Village, Trapaing Prasate Commune, Trapaing Prasate District')
        | (MQD_df_CAM.Facility_Location == 'Trapaing Prasate District')
        | (MQD_df_CAM.Facility_Location == 'Trapeang Prasat District')
        | (MQD_df_CAM.Facility_Location == 'Trapaing Prasate Village, Trapaing Prasate Commune, Trapaing Prasate District')
        | (MQD_df_CAM.Facility_Location == 'Trapaing Prasate Village, Trapaing Prasate Commune, Trapaing Prasate District')
        | (MQD_df_CAM.Facility_Location == 'Trapaing Prasate Village, Trapaing Prasate Commune, Trapaing Prasate District')
        | (MQD_df_CAM.Facility_Location == 'Tumnub Dach Village, Tumnub Dach Commune, Trapaing Prasate Distict')
        | (MQD_df_CAM.Facility_Location == 'Tumnub Dach Village, Tumnub Dach Commune, Trapaing prasate District')
        | (MQD_df_CAM.Facility_Location == 'Tumnubdach Village, Tumnubdach Commune, Trapaing Prasate District')
        | (MQD_df_CAM.Facility_Location == 'Trapaing Prasate Village, Trapaing Prasate Commune, Trapaing Prasate District'),
        'Facility_Location_GROUPED'] = 'Trapeang Prasat District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Van Sai District')
        | (MQD_df_CAM.Facility_Location == 'Srok Vern sai'),
        'Facility_Location_GROUPED'] = 'Van Sai District'
    MQD_df_CAM.loc[ (MQD_df_CAM.Facility_Location_GROUPED == 'Missing')
        | (MQD_df_CAM.Facility_Location_GROUPED == 'MANUALLY_MODIFY')
        | (MQD_df_CAM.Facility_Location_GROUPED == 'NA VALUE')    ,
        'Facility_Location_GROUPED'] = 'NA VALUE'
    MQD_df_CAM.loc[MQD_df_CAM['Facility_Location_GROUPED'].isnull(), 'Facility_Location_GROUPED'] = 'NA VALUE'

    #piv = MQD_df_CAM.pivot_table(index=['Facility_Location'], columns=['Final_Test_Conclusion'], aggfunc='size',
     #                            fill_value=0)
    #piv.sort_values(['Pass'],ascending=False)

    # Facility_Name
    '''
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Name == '')
        | (MQD_df_CAM.Facility_Name == ''),
        'Facility_Name'] = ''
    '''
    MQD_df_CAM = assignlabels(MQD_df_CAM, 'Facility_Name')
    #todo: MANUAL ADJUSTMENTS TO FACILITY_NAME FOR CAMBODIA LIKELY NEEDED



    '''
    USEFUL FUNCTIONS
    
    MQD_df_CAM.keys()
    MQD_df_CAM[MQD_df_CAM.Facility_Location == "Salakrao District, Pailin,"].count()

    a = MQD_df_CAM['Facility_Name'].astype('str').unique()
    print(len(a))
    for item in sorted(a):
        print(item)
    piv = MQD_df_CAM.pivot_table(index=['Facility_Location'], columns=['Final_Test_Conclusion'], aggfunc='size', fill_value=0)

    '''

    # ETHIOPIA PROCESSING
    # Province
    templist = MQD_df_ETH['Province_Name'].tolist()
    MQD_df_ETH['Province_Name_GROUPED'] = templist
    MQD_df_ETH.loc[
        (MQD_df_ETH.Province_Name == 'Alamata') | (MQD_df_ETH.Province_Name == 'Alamata-Site9')
        | (MQD_df_ETH.Province_Name == 'Alamata-site9')
        | (MQD_df_ETH.Province_Name == 'Alamata-Site7') | (MQD_df_ETH.Province_Name == 'Alamata-Site8')
        | (MQD_df_ETH.Province_Name == 'Alamata-Site10') | (MQD_df_ETH.Province_Name == 'Alamata-Site11')
        | (MQD_df_ETH.Province_Name == 'Alamata-site7') | (MQD_df_ETH.Province_Name == 'Alamata-site8')
        | (MQD_df_ETH.Province_Name == 'Alamata-site10') | (MQD_df_ETH.Province_Name == 'Alamata-site11')
        | (MQD_df_ETH.Province_Name == 'Alamata-Site12') | (MQD_df_ETH.Province_Name == 'Alamata-Site13')
        | (MQD_df_ETH.Province_Name == 'Alamata-Site14') | (MQD_df_ETH.Province_Name == 'Alamata-Site15')
        | (MQD_df_ETH.Province_Name == 'Alamata-Site16') | (MQD_df_ETH.Province_Name == 'Alamata-Site17')
        | (MQD_df_ETH.Province_Name == 'Alamata-Site18') | (MQD_df_ETH.Province_Name == 'Alamata-Site19')
        | (MQD_df_ETH.Province_Name == 'Alamata-Site20') | (MQD_df_ETH.Province_Name == 'Alamata-Site21')
        | (MQD_df_ETH.Province_Name == 'Alamata-Site22') | (MQD_df_ETH.Province_Name == 'Alamata-Site23')
        | (MQD_df_ETH.Province_Name == 'Alamata-Site24') | (MQD_df_ETH.Province_Name == 'Alamata-Site25')
        | (MQD_df_ETH.Province_Name == 'Alamata-Site26') | (MQD_df_ETH.Province_Name == 'Alamata-Site27')
        | (MQD_df_ETH.Province_Name == 'Alamata-Site28') | (MQD_df_ETH.Province_Name == 'Alamata-Site29')
        | (MQD_df_ETH.Province_Name == 'Alamata-Site30') | (MQD_df_ETH.Province_Name == 'Alamata-Site31')
        | (MQD_df_ETH.Province_Name == 'Alamata-Site32') | (MQD_df_ETH.Province_Name == 'Alamata-Site33')
        | (MQD_df_ETH.Province_Name == 'Alamata-Site34') | (MQD_df_ETH.Province_Name == 'Alamata-Site35')
        | (MQD_df_ETH.Province_Name == 'Alamata-Site36') | (MQD_df_ETH.Province_Name == 'Alamata-Site37')
        | (MQD_df_ETH.Province_Name == 'Alamata-Site38') | (MQD_df_ETH.Province_Name == 'Alamata-Site39')
        | (MQD_df_ETH.Province_Name == 'Alamata-Site40') | (MQD_df_ETH.Province_Name == 'Alamata-Site41')
        | (MQD_df_ETH.Province_Name == 'Alamata-Site42') | (MQD_df_ETH.Province_Name == 'Alamata-Site43')
        | (MQD_df_ETH.Province_Name == 'Alamata-Site44') | (MQD_df_ETH.Province_Name == 'Alamata-Site45')
        | (MQD_df_ETH.Province_Name == 'Alamata-Site46'),
        'Province_Name_GROUPED'] = 'Alamata'

    # Facility_Location
    templist = MQD_df_ETH['Facility_Location'].tolist()
    MQD_df_ETH['Facility_Location_GROUPED'] = templist
    '''
    A/K S/C
    Abobo
    Adama City Admin
    Agaro
    Ambo City Admin
    Arbaminchi
    Arsi
    Asayita
    Asela Woreda
    Assosa
    Awash 7 Kilo
    Babile
    Bahir Dar
    Borena
    Central
    Chefra
    Chinakson
    Chiro
    Dessie
    Diredawa
    East Ethiopia
    East Shoa
    Gambela
    Gedio
    Gende Weha
    Gimbi
    Goji Zone
    Gondar
    Gulele
    Haramaya
    Harar
    Hartishek
    Hawasa
    Jigjiga
    Jimma
    Kebribaya
    Kirkos Sc
    Larie
    Lideta Sc
    Logiya
    Mekele
    Metehara
    Metema
    Metu
    Nedjo
    Nekemte
    North Showa
    Northeast Ethiopia
    Northwest Ethiopia
    Semera
    Shashamane
    Shele
    South Ethiopia
    Southwest Ethiopia
    Southwest Showa
    Togochale
    Walayta Sodo
    Weldiya
    West Ethiopia
    West Showa
    Wolayita
    Wolisso Town Admin
    '''
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Location == 'Arbamichi') | (MQD_df_ETH.Facility_Location == 'Arbaminchi'),
        'Facility_Location_GROUPED'] = 'Arbaminchi'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Location == 'Bahirdar') | (MQD_df_ETH.Facility_Location == 'Bahir Dar'),
        'Facility_Location_GROUPED'] = 'Bahir Dar'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Location == 'Borena') | (MQD_df_ETH.Facility_Location == 'Borena Zone'),
        'Facility_Location_GROUPED'] = 'Borena'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Location == 'Centeral') | (MQD_df_ETH.Facility_Location == 'Central'),
        'Facility_Location_GROUPED'] = 'Central'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Location == 'Diredawa') | (MQD_df_ETH.Facility_Location == 'Dirredawa'),
        'Facility_Location_GROUPED'] = 'Diredawa'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Location == 'East') | (MQD_df_ETH.Facility_Location == 'East Ethiopia'),
        'Facility_Location_GROUPED'] = 'East Ethiopia'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Location == 'Gambela') | (MQD_df_ETH.Facility_Location == 'Gambela Tawn')
        | (MQD_df_ETH.Facility_Location == 'Gambela Town') | (MQD_df_ETH.Facility_Location == 'Gambella'),
        'Facility_Location_GROUPED'] = 'Gambela'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Location == 'Gedio') | (MQD_df_ETH.Facility_Location == 'Gedio Zone'),
        'Facility_Location_GROUPED'] = 'Gedio'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Location == 'Gulele') | (MQD_df_ETH.Facility_Location == 'Guilele Sc'),
        'Facility_Location_GROUPED'] = 'Gulele'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Location == 'Harar') | (MQD_df_ETH.Facility_Location == 'Harer'),
        'Facility_Location_GROUPED'] = 'Harar'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Location == 'Hartishek') | (MQD_df_ETH.Facility_Location == 'Hatshek'),
        'Facility_Location_GROUPED'] = 'Hartishek'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Location == 'Jimma') | (MQD_df_ETH.Facility_Location == 'Jimma  Tawn'),
        'Facility_Location_GROUPED'] = 'Jimma'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Location == 'Metu') | (MQD_df_ETH.Facility_Location == 'Metu Tawn')
        | (MQD_df_ETH.Facility_Location == 'Metu Town'),
        'Facility_Location_GROUPED'] = 'Metu'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Location == 'Nekemete') | (MQD_df_ETH.Facility_Location == 'Nkemete')
        | (MQD_df_ETH.Facility_Location == 'Nekemte'),
        'Facility_Location_GROUPED'] = 'Nekemte'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Location == 'North Shoa') | (MQD_df_ETH.Facility_Location == 'North Showa'),
        'Facility_Location_GROUPED'] = 'North Showa'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Location == 'North West Ethiopia')
        | (MQD_df_ETH.Facility_Location == 'North west Ethiopia')
        | (MQD_df_ETH.Facility_Location == 'Nothr west Ethiopia')
        | (MQD_df_ETH.Facility_Location == 'North westEthiopia'),
        'Facility_Location_GROUPED'] = 'Northwest Ethiopia'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Location == 'North east Ethiopia'),
        'Facility_Location_GROUPED'] = 'Northeast Ethiopia'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Location == 'South') | (MQD_df_ETH.Facility_Location == 'South Ethiopia'),
        'Facility_Location_GROUPED'] = 'South Ethiopia'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Location == 'South West Shoa'),
        'Facility_Location_GROUPED'] = 'Southwest Showa'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Location == 'South west Ethiopia'),
        'Facility_Location_GROUPED'] = 'Southwest Ethiopia'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Location == 'Walayta Sodo') | (MQD_df_ETH.Facility_Location == 'Walaytasodo'),
        'Facility_Location_GROUPED'] = 'Walayta Sodo'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Location == 'West'),
        'Facility_Location_GROUPED'] = 'West Ethiopia'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Location == 'West Shoa') | (MQD_df_ETH.Facility_Location == 'West Showa'),
        'Facility_Location_GROUPED'] = 'West Showa'
    MQD_df_ETH.loc[MQD_df_ETH['Facility_Location_GROUPED'].isnull(), 'Facility_Location_GROUPED'] = 'NA VALUE'

    # Facility_Name
    MQD_df_ETH = assignlabels(MQD_df_ETH, 'Facility_Name')

    '''
    for i in range(len(MQD_df_ETH['Facility_Name'])):
        if not MQD_df_ETH.iloc[i]['Facility_Name'] == MQD_df_ETH.iloc[i]['Facility_Name_GROUPED']:
            print('|| '+ str(MQD_df_ETH.iloc[i]['Facility_Name']) +' || ' + str(MQD_df_ETH.iloc[i]['Facility_Name_GROUPED']) + ' ||')
    '''
    # Manual adjustments
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Name_GROUPED == 'Adare Gen. Hos') |
        (MQD_df_ETH.Facility_Name_GROUPED == 'Adare General Hospital'),
        'Facility_Name_GROUPED'] = 'Adare General Hospital'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Name_GROUPED == 'Amanuel') |
        (MQD_df_ETH.Facility_Name_GROUPED == 'Amanuel Drug Store'),
        'Facility_Name_GROUPED'] = 'Amanuel Drug Store'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Name_GROUPED == 'Arbamincji Gen.Hospital') |
        (MQD_df_ETH.Facility_Name_GROUPED == 'Arbaminch Gen Hos') |
        (MQD_df_ETH.Facility_Name_GROUPED == 'Arbaminch General Hospital'),
        'Facility_Name_GROUPED'] = 'Arbaminch General Hospital'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Name_GROUPED == 'Assossa Hospital') |
        (MQD_df_ETH.Facility_Name_GROUPED == 'Assosa General Hospital'),
        'Facility_Name_GROUPED'] = 'Assosa General Hospital'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Name_GROUPED == 'Eskinder') |
        (MQD_df_ETH.Facility_Name_GROUPED == 'Eskinder Pharmacy'),
        'Facility_Name_GROUPED'] = 'Eskinder Pharmacy'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Name_GROUPED == 'Jhon Abisinya') |
        (MQD_df_ETH.Facility_Name_GROUPED == 'Jhone Abysinia'),
        'Facility_Name_GROUPED'] = 'Jhone Abysinia'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Name_GROUPED == 'Karamara Hospital  Pharmacy') |
        (MQD_df_ETH.Facility_Name_GROUPED == 'Karamara hospital'),
        'Facility_Name_GROUPED'] = 'Karamara hospital'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Name_GROUPED == 'Logiya') |
        (MQD_df_ETH.Facility_Name_GROUPED == 'Logia  Health Center'),
        'Facility_Name_GROUPED'] = 'Logia  Health Center'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Name_GROUPED == 'Metema District Hospital') |
        (MQD_df_ETH.Facility_Name_GROUPED == 'Metema hospital'),
        'Facility_Name_GROUPED'] = 'Metema hospital'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Name_GROUPED == 'Ongazl Pharmacy') |
        (MQD_df_ETH.Facility_Name_GROUPED == 'Ougazi Pharmacy'),
        'Facility_Name_GROUPED'] = 'Ougazi Pharmacy'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Name_GROUPED == 'Rehobot Clinic') |
        (MQD_df_ETH.Facility_Name_GROUPED == 'Rohobot') |
        (MQD_df_ETH.Facility_Name_GROUPED == 'Rohbot clinic'),
        'Facility_Name_GROUPED'] = 'Rohbot clinic'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Name_GROUPED == 'Sahel Pharma') |
        (MQD_df_ETH.Facility_Name_GROUPED == 'Sahal Pharmacy'),
        'Facility_Name_GROUPED'] = 'Sahal Pharmacy'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Name_GROUPED == 'Tesfa') |
        (MQD_df_ETH.Facility_Name_GROUPED == 'Tesfa Drug Store'),
        'Facility_Name_GROUPED'] = 'Tesfa Drug Store'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Facility_Name_GROUPED == 'Walayta Sodo Uni Teaching &Reveral Hos') |
        (MQD_df_ETH.Facility_Name_GROUPED == 'Walayta Universcity Teaching And Reveral Hospital') |
        (MQD_df_ETH.Facility_Name_GROUPED == 'Walaytasodo Univercity Teaching Referal Hospital'),
        'Facility_Name_GROUPED'] = 'Walayta Universcity Teaching And Reveral Hospital'

    # Manufacturer
    MQD_df_ETH = assignlabels(MQD_df_ETH, 'Manufacturer',thresh=90)

    # Manual adjustments
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'Addis Pharmaceutical Factory')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Addis Pharmaceutical Plc')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Addis Pharmaceuticals'),
        'Manufacturer_GROUPED'] = 'Addis Pharmaceuticals'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'Astra Life Care Pvt.Ltd')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Astralife Care'),
        'Manufacturer_GROUPED'] = 'Astralife Care'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'Cadila')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Cadila Healthcare')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Cadila Pharm.Plc')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Cadila Pharmaceuticals')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Cadila,Health.Ltd'),
        'Manufacturer_GROUPED'] = 'Cadila'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'Cipla')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Cipla Ltd')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Cipla,Ltd.Plot10')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Cipla,Midlkumba')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Cipla,Pithmpur'),
        'Manufacturer_GROUPED'] = 'Cipla'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'East African Pharmaceuticals')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'East Africa Plc')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'East African Pharmaceutical Factory'),
        'Manufacturer_GROUPED'] = 'East African Pharmaceuticals'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'Emcure')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Emcure Pharmaceuticals')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Emcurepharmaceuticals,Pune')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Emucure Pharmaceutical Ltd'),
        'Manufacturer_GROUPED'] = 'Emcure'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'Ethiopian Pharmaceutical Manufacturing')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Ethiopian Pharmaceuticals'),
        'Manufacturer_GROUPED'] = 'Ethiopian Pharmaceuticals'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'Fawes Pharmaceuticals')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Fewes Pharmaceuticals Factory'),
        'Manufacturer_GROUPED'] = 'Fawes Pharmaceuticals'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'Flamingo')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Flamingo Pharmaceuticaks.Ltd Taloga,4102208')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Flamingo Pharmaceuticals'),
        'Manufacturer_GROUPED'] = 'Flamingo'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'Gulin Pharma')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Guilin Pharmaceuticals.Co.Ltd')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Gulin Pharmaceuticals'),
        'Manufacturer_GROUPED'] = 'Gulin Pharma'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'Houns Co., Ltd')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Huons Co.Ltd'),
        'Manufacturer_GROUPED'] = 'Houns Co., Ltd'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'IPCA')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Ipca Laboratories Ltd'),
        'Manufacturer_GROUPED'] = 'IPCA'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'Jeil  Pharm. Co.Ltd')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Jiel Pharma Co.Ltd'),
        'Manufacturer_GROUPED'] = 'Jiel Pharma Co.Ltd'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'Julphar')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Julphar  Pharma Plc')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Julphar Pharmaceutical Plc')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Julphar Pharmaceuticals'),
        'Manufacturer_GROUPED'] = 'Julphar'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'Lab,Renaudin')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Laboratorie Renaudin'),
        'Manufacturer_GROUPED'] = 'Laboratorie Renaudin'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'Leben Laboratories')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Leben  Laboratories.Pvt.Ltd')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Leben Laboratories,Trinity Street')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Leben Laboratory Pvt.Ltd')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Lebonlaboratories Pvt.Ltd.Mumbi'),
        'Manufacturer_GROUPED'] = 'Leben Laboratories'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'Laboratorie Pharmaceutical Rodael')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Lp Rodael'),
        'Manufacturer_GROUPED'] = 'Laboratorie Pharmaceutical Rodael'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'Macleodes Harmaceuticals')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Macleods'),
        'Manufacturer_GROUPED'] = 'Macleods'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'Medicamen Biotech Laboratories')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Medicamen Biotech'),
        'Manufacturer_GROUPED'] = 'Medicamen Biotech'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'Mepha Ltd')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Mepha'),
        'Manufacturer_GROUPED'] = 'Mepha'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'NA VALUE')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'No information'),
        'Manufacturer_GROUPED'] = 'NA VALUE'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'Novartis, Basel')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Novartis Saqlik')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Novartis Pharmaag')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Novartis'),
        'Manufacturer_GROUPED'] = 'Novartis'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'Nutriset Laboratories Rode(France)')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Nutiset Sas Laboratories'),
        'Manufacturer_GROUPED'] = 'Nutiset Sas Laboratories'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'Remedica Ltd')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'REMEDICA'),
        'Manufacturer_GROUPED'] = 'REMEDICA'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'Sandoz GmbH')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Sandoz'),
        'Manufacturer_GROUPED'] = 'Sandoz'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'Sk+F Eskayef Bangladish Ltd')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Sktf,Eskayef Ban Gladeshltd.')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Skf,Eskaef'),
        'Manufacturer_GROUPED'] = 'Skf,Eskaef'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'Uniquepharmaceutical Laboratories')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Unique Pharmaceuticals')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Unsque Pharmaceutical Lab'),
        'Manufacturer_GROUPED'] = 'Unique Pharmaceuticals'
    MQD_df_ETH.loc[
        (MQD_df_ETH.Manufacturer_GROUPED == 'Universal Corporation Ltd')
        | (MQD_df_ETH.Manufacturer_GROUPED == 'Universal Co.Ltd'),
        'Manufacturer_GROUPED'] = 'Universal Co.Ltd'


    # GHANA
    # Province_Name
    MQD_df_GHA = assignlabels(MQD_df_GHA, 'Province_Name', thresh=90)
    # Manual adjustments
    MQD_df_GHA.loc[
        (MQD_df_GHA.Province_Name_GROUPED == 'Northern Region')
        | (MQD_df_GHA.Province_Name_GROUPED == 'Northern'),
        'Province_Name_GROUPED'] = 'Northern'

    # Facility_Location
    MQD_df_GHA = assignlabels(MQD_df_GHA, 'Facility_Location', thresh=90)
    # Manual adjustments
    MQD_df_GHA.loc[
        (MQD_df_GHA.Facility_Location_GROUPED == 'Missing')
        | (MQD_df_GHA.Facility_Location_GROUPED == 'NA VALUE'),
        'Facility_Location_GROUPED'] = 'NA VALUE'
    MQD_df_GHA.loc[
        (MQD_df_GHA.Facility_Location_GROUPED == 'Tamale')
        | (MQD_df_GHA.Facility_Location_GROUPED == 'Kukuo, Tamale'),
        'Facility_Location_GROUPED'] = 'Tamale'

    # Facility_Name
    MQD_df_GHA = assignlabels(MQD_df_GHA, 'Facility_Name', thresh=90)
    # Manual adjustments
    # todo: MANUAL ADJUSTMENTS REQUIRED

    # Manufacturer
    MQD_df_GHA = assignlabels(MQD_df_GHA, 'Manufacturer', thresh=90)
    # Manual adjustments
    MQD_df_GHA.loc[
        (MQD_df_GHA.Manufacturer_GROUPED == 'Bliss GVS Pharma Ltd')
        | (MQD_df_GHA.Manufacturer_GROUPED == 'Bliss GVS Pharmaceuticals Ltd.'),
        'Manufacturer_GROUPED'] = 'Bliss GVS Pharma Ltd'
    MQD_df_GHA.loc[
        (MQD_df_GHA.Manufacturer_GROUPED == 'Cipla Ltd. India')
        | (MQD_df_GHA.Manufacturer_GROUPED == 'Cipla Ltd'),
        'Manufacturer_GROUPED'] = 'Cipla Ltd'
    MQD_df_GHA.loc[
        (MQD_df_GHA.Manufacturer_GROUPED == 'Guilin Pharmaceutical Co. Ltd')
        | (MQD_df_GHA.Manufacturer_GROUPED == 'Guilin Pharmaceutical Company Ltd.'),
        'Manufacturer_GROUPED'] = 'Guilin Pharmaceutical Co. Ltd'
    MQD_df_GHA.loc[
        (MQD_df_GHA.Manufacturer_GROUPED == 'Kinapharma Limited')
        | (MQD_df_GHA.Manufacturer_GROUPED == 'Kinapharma Ltd'),
        'Manufacturer_GROUPED'] = 'Kinapharma Ltd'
    MQD_df_GHA.loc[
        (MQD_df_GHA.Manufacturer_GROUPED == 'Maphar')
        | (MQD_df_GHA.Manufacturer_GROUPED == 'Maphar Laboratories'),
        'Manufacturer_GROUPED'] = 'Maphar'
    MQD_df_GHA.loc[
        (MQD_df_GHA.Manufacturer_GROUPED == 'Pharmanova Limited')
        | (MQD_df_GHA.Manufacturer_GROUPED == 'Pharmanova Ltd'),
        'Manufacturer_GROUPED'] = 'Pharmanova Ltd'


    # KENYA
    # Province_Name
    MQD_df_KEN = assignlabels(MQD_df_KEN, 'Province_Name', thresh=90)

    # Facility_Location ('County' for Kenya)
    facilityLocationList = [
        'AHERO',
        'AMAGORO',
        'ATHI RIVER',
        'BUNGOMA',
        'BUSIA TOWN',
        'CHEPKOILEL JUNCTION',
        'DAGORETI',
        'ELDORET',
        'HURLINGHAM',
        'HURUMA',
        'ISINYA',
        'KAJIADO',
        'KAKAMEGA',
        'KAREN',
        'KENYATTA MARKET',
        'KHAYEGA',
        'KILIFI',
        'KIKONO',
        'KINANGO',
        'KISII',
        'KISUMU',
        'KITALE',
        'KITENGELA',
        'KWALE',
        'LIKONI',
        'MALABA',
        'MARAGOLI',
        'MASENO',
        'MBALE TOWN',
        'MIKIDANI',
        'MLOLONGO',
        'MOMBASA',
        'MSABWENI',
        'MTWAPA',
        'MUMIAS',
        'NA VALUE',
        'NAMBALE',
        'NAIROBI',
        'NGONG',
        'PRESTIGE',
        'RABOUR',
        'RUIRU',
        'SIAYA',
        'SONDU',
        'THIKA',
        'UASIN GISHU',
        'UKUNDA',
        'WEBUYE',
        'WESTLAND'
    ]
    MQD_df_KEN = assignlabels(MQD_df_KEN, 'Facility_Location', facilityLocationList, thresh=90)
    # Manual adjustments
    MQD_df_KEN.loc[
        (MQD_df_KEN.Facility_Location == 'NAIROBI')
        | (MQD_df_KEN.Facility_Location == 'ALONG FITINA ROAD, OFF')
        | (MQD_df_KEN.Facility_Location == 'ALONG KENYATTA AVENUE,')
        | (MQD_df_KEN.Facility_Location == 'ALONG KENYATTA AVENUE, OPP.')
        | (MQD_df_KEN.Facility_Location == 'CITY CENTRE')
        | (MQD_df_KEN.Facility_Location == 'GITHURAI KIMBO ROAD')
        | (MQD_df_KEN.Facility_Location == 'JUJA, NEAR KENYATTA ROAD')
        | (MQD_df_KEN.Facility_Location == 'LENANA ROAD')
        | (MQD_df_KEN.Facility_Location == 'LIFESTYLE NAKUMATT')
        | (MQD_df_KEN.Facility_Location == 'MBAGATHI ROAD')
        | (MQD_df_KEN.Facility_Location == 'MBAGATHI WAY')
        | (MQD_df_KEN.Facility_Location == 'MBAGATHI WAY FLATS')
        | (MQD_df_KEN.Facility_Location == 'OFF. OTIENDE ROAD')
        | (MQD_df_KEN.Facility_Location == 'P.O Box 1022, 00606 Sarit Centre')
        | (MQD_df_KEN.Facility_Location == 'P.O. BOX 1022, NORTH AIRPORT ROAD')
        | (MQD_df_KEN.Facility_Location == 'Petro Station, Uganda Road')
        | (MQD_df_KEN.Facility_Location == 'RONALD NGARA STREET')
        | (MQD_df_KEN.Facility_Location == 'Uganda Road, P O Box 40 Turbo'),
        'Facility_Location_GROUPED'] = 'NAIROBI'
    MQD_df_KEN.loc[
        (MQD_df_KEN.Facility_Location == 'ATHI RIVER')
        | (MQD_df_KEN.Facility_Location == 'EPZ ROAD'),
        'Facility_Location_GROUPED'] = 'ATHI RIVER'
    MQD_df_KEN.loc[
        (MQD_df_KEN.Facility_Location == 'BUNGOMA')
        | (MQD_df_KEN.Facility_Location == 'CANON AWORI STREET'),
        'Facility_Location_GROUPED'] = 'BUNGOMA'
    MQD_df_KEN.loc[
        (MQD_df_KEN.Facility_Location == 'BUSIA TOWN')
        | (MQD_df_KEN.Facility_Location == 'BUSIA')
        | (MQD_df_KEN.Facility_Location == 'HOSPITAL ROAD, BUSIA')
        | (MQD_df_KEN.Facility_Location == 'P O BOX 420 BUSIA (K)')
        | (MQD_df_KEN.Facility_Location == 'P O BOX 485 BUSIA (K)')
        | (MQD_df_KEN.Facility_Location == 'P O BOX 87 BUSIA (K)'),
        'Facility_Location_GROUPED'] = 'BUSIA'
    MQD_df_KEN.loc[
        (MQD_df_KEN.Facility_Location == 'KISII')
        | (MQD_df_KEN.Facility_Location == 'HOSPITAL ROAD, OPP. SHABANA HARDWARE, KISII'),
        'Facility_Location_GROUPED'] = 'KISII'
    MQD_df_KEN.loc[
        (MQD_df_KEN.Facility_Location == 'KISUMU')
        | (MQD_df_KEN.Facility_Location == 'KISUMU USENGE RD, NEXT TO BONDO TOWNSHIP PRI. SCH.')
        | (MQD_df_KEN.Facility_Location == 'OPPOSITE MAGHARIBI PETROL'),
        'Facility_Location_GROUPED'] = 'KISUMU'
    MQD_df_KEN.loc[
        (MQD_df_KEN.Facility_Location == 'MOMBASA')
        | (MQD_df_KEN.Facility_Location == 'LIKONO'),
        'Facility_Location_GROUPED'] = 'MOMBASA'
    MQD_df_KEN.loc[
        (MQD_df_KEN.Facility_Location == 'NA VALUE')
        | (MQD_df_KEN.Facility_Location == 'Missing')
        | (MQD_df_KEN.Facility_Location == 'Relax In Building'),
        'Facility_Location_GROUPED'] = 'NA VALUE'
    MQD_df_KEN.loc[
        (MQD_df_KEN.Facility_Location == 'UASIN GISHU')
        | (MQD_df_KEN.Facility_Location == 'Turbo Town Centre'),
        'Facility_Location_GROUPED'] = 'UASIN GISHU'

    # Facility_Name
    MQD_df_KEN = assignlabels(MQD_df_KEN, 'Facility_Name', thresh=90)
    # todo: Manual adjustments required

    # Manufacturer
    templist = MQD_df_KEN['Manufacturer'].tolist()
    MQD_df_KEN['Manufacturer_GROUPED'] = templist

    ### RESUME HERE ###
    MQD_df_KEN.loc[
        (MQD_df_KEN.Manufacturer == 'Ajanta Pharma Limited')
        | (MQD_df_KEN.Manufacturer == 'Ajanta'),
        'Manufacturer_GROUPED'] = 'Ajanta'
    MQD_df_KEN.loc[
        (MQD_df_KEN.Manufacturer == 'Biodeal Laboratories Ltd')
        | (MQD_df_KEN.Manufacturer == 'Biodeal'),
        'Manufacturer_GROUPED'] = 'Biodeal'
    MQD_df_KEN.loc[
        (MQD_df_KEN.Manufacturer == 'Bliss Gvis Pharma Ltd')
        | (MQD_df_KEN.Manufacturer == 'Bliss Gvs'),
        'Manufacturer_GROUPED'] = 'Bliss Gvs'
    MQD_df_KEN.loc[
        (MQD_df_KEN.Manufacturer == 'Cipla Ltd')
        | (MQD_df_KEN.Manufacturer == 'Cipla'),
        'Manufacturer_GROUPED'] = 'Cipla'
    MQD_df_KEN.loc[
        (MQD_df_KEN.Manufacturer == 'Cosmos Ltd')
        | (MQD_df_KEN.Manufacturer == 'Cosmos'),
        'Manufacturer_GROUPED'] = 'Cosmos'
    MQD_df_KEN.loc[
        (MQD_df_KEN.Manufacturer == 'Dawa Ltd')
        | (MQD_df_KEN.Manufacturer == 'Dawa'),
        'Manufacturer_GROUPED'] = 'Dawa'
    MQD_df_KEN.loc[
        (MQD_df_KEN.Manufacturer == 'Elys Chemical Industrial Ltd')
        | (MQD_df_KEN.Manufacturer == 'Elys Chemical Industries'),
        'Manufacturer_GROUPED'] = 'Elys Chemical Industries'
    MQD_df_KEN.loc[
        (MQD_df_KEN.Manufacturer == 'FARMACEUTICUS LAKECITY S.A DF')
        | (MQD_df_KEN.Manufacturer == 'Farmaceuticos L. Sadecv'),
        'Manufacturer_GROUPED'] = 'Farmaceuticos L. Sadecv'
    MQD_df_KEN.loc[
        (MQD_df_KEN.Manufacturer == 'Flamingo Pharmaceuticals Ltd')
        | (MQD_df_KEN.Manufacturer == 'Flamingo'),
        'Manufacturer_GROUPED'] = 'Flamingo'
    MQD_df_KEN.loc[
        (MQD_df_KEN.Manufacturer == 'Guilin Pharmaceutical Co., Ltd.')
        | (MQD_df_KEN.Manufacturer == 'Guilin'),
        'Manufacturer_GROUPED'] = 'Guilin'
    MQD_df_KEN.loc[
        (MQD_df_KEN.Manufacturer == 'Indus Pharma (Pvt) Ltd')
        | (MQD_df_KEN.Manufacturer == 'Indus'),
        'Manufacturer_GROUPED'] = 'Indus'
    MQD_df_KEN.loc[
        (MQD_df_KEN.Manufacturer == 'IPCA')
        | (MQD_df_KEN.Manufacturer == 'Ipca Laboratories'),
        'Manufacturer_GROUPED'] = 'Ipca Laboratories'
    MQD_df_KEN.loc[
        (MQD_df_KEN.Manufacturer == 'Laboratory and Allied')
        | (MQD_df_KEN.Manufacturer == 'Laboratory & Allied Ltd'),
        'Manufacturer_GROUPED'] = 'Laboratory & Allied Ltd'
    MQD_df_KEN.loc[
        (MQD_df_KEN.Manufacturer == 'Roche Products (PTY) Ltd')
        | (MQD_df_KEN.Manufacturer == 'Roche Products')
        | (MQD_df_KEN.Manufacturer == 'Roche'),
        'Manufacturer_GROUPED'] = 'Roche'
    MQD_df_KEN.loc[
        (MQD_df_KEN.Manufacturer == 'Shanghai pharmaceutical industrial corporation')
        | (MQD_df_KEN.Manufacturer == 'Shanghai Pharmateq Ltd'),
        'Manufacturer_GROUPED'] = 'Shanghai Pharmateq Ltd'
    MQD_df_KEN.loc[
        (MQD_df_KEN.Manufacturer == 'Troikaa Pharmaceuticals Ltd')
        | (MQD_df_KEN.Manufacturer == 'Troikaa'),
        'Manufacturer_GROUPED'] = 'Troikaa'
    MQD_df_KEN.loc[
        (MQD_df_KEN.Manufacturer == 'Umedica Laboratories PVT. Ltd')
        | (MQD_df_KEN.Manufacturer == 'Umedica Laboratories Pvt Ltd'),
        'Manufacturer_GROUPED'] = 'Umedica Laboratories Pvt Ltd'
    MQD_df_KEN.loc[
        (MQD_df_KEN.Manufacturer == 'Universal Corporation Ltd')
        | (MQD_df_KEN.Manufacturer == 'Universal'),
        'Manufacturer_GROUPED'] = 'Universal'


    # LAOS
    # Province_Name
    templist = MQD_df_LAO['Province_Name'].tolist()
    MQD_df_LAO['Province_Name_GROUPED'] = templist
    #Manual adjustments
    MQD_df_LAO.loc[
        (MQD_df_LAO.Province_Name == 'Saiyabouly')
        | (MQD_df_LAO.Province_Name == 'Sayabuly'),
        'Province_Name_GROUPED'] = 'Sayabuly'
    MQD_df_LAO.loc[
        (MQD_df_LAO.Province_Name == 'Xiengkhouang')
        | (MQD_df_LAO.Province_Name == 'Xiengkhuang'),
        'Province_Name_GROUPED'] = 'Xiengkhuang'

    # Facility_Location: 'Districts' in Laos
    MQD_df_LAO = assignlabels(MQD_df_LAO, 'Facility_Location', thresh=90)
    # Remove extra spaces
    tempList = []
    for elem in MQD_df_LAO['Facility_Location_GROUPED']:
        tempList.append(" ".join(elem.split()))
    MQD_df_LAO['Facility_Location_GROUPED'] = tempList

    # Manual adjustments
    MQD_df_LAO.loc[
        (MQD_df_LAO.Facility_Location_GROUPED == 'Luangprabang District')
        | (MQD_df_LAO.Facility_Location_GROUPED == 'Luaprabang District'),
        'Facility_Location_GROUPED'] = 'Luangprabang District'
    MQD_df_LAO.loc[
        (MQD_df_LAO.Facility_Location_GROUPED == 'Pakse District')
        | (MQD_df_LAO.Facility_Location_GROUPED == 'Parkse District'),
        'Facility_Location_GROUPED'] = 'Pakse District'
    MQD_df_LAO.loc[
        (MQD_df_LAO.Facility_Location_GROUPED == 'Pek District')
        | (MQD_df_LAO.Facility_Location_GROUPED == 'Perk District'),
        'Facility_Location_GROUPED'] = 'Pek District'
    MQD_df_LAO.loc[
        (MQD_df_LAO.Facility_Location_GROUPED == 'Pheing District')
        | (MQD_df_LAO.Facility_Location_GROUPED == 'Phieng District'),
        'Facility_Location_GROUPED'] = 'Phieng District'
    MQD_df_LAO.loc[
        (MQD_df_LAO.Facility_Location_GROUPED == 'Viengxai District')
        | (MQD_df_LAO.Facility_Location_GROUPED == 'Viengsay District'),
        'Facility_Location_GROUPED'] = 'Viengxai District'

    # Facility_Name
    # Remove all the weird spacing
    tempList = []
    for elem in MQD_df_LAO['Facility_Name']:
        tempList.append(" ".join(elem.split()))
    MQD_df_LAO['Facility_Name'] = tempList

    MQD_df_LAO = assignlabels(MQD_df_LAO, 'Facility_Name', thresh=90)
    #todo: Manually adjust

    # Manufacturer
    MQD_df_LAO = assignlabels(MQD_df_LAO, 'Manufacturer', thresh=90)
    # Manual adjustments
    MQD_df_LAO.loc[
        (MQD_df_LAO.Manufacturer_GROUPED == 'Asian Union Laboratories Co., Ltd')
        | (MQD_df_LAO.Manufacturer_GROUPED == 'Asian Union'),
        'Manufacturer_GROUPED'] = 'Asian Union'
    MQD_df_LAO.loc[
        (MQD_df_LAO.Manufacturer_GROUPED == 'Bangkok')
        | (MQD_df_LAO.Manufacturer_GROUPED == 'Bangkok Lab')
        | (MQD_df_LAO.Manufacturer_GROUPED == 'Bankok Lab & Cosmetic Co. LTD')
        | (MQD_df_LAO.Manufacturer_GROUPED == 'Bangkok Drug Co.'),
        'Manufacturer_GROUPED'] = 'Bangkok Drug Co.'
    MQD_df_LAO.loc[
        (MQD_df_LAO.Manufacturer_GROUPED == 'Central Pharmaceutical Technical Development Company')
        | (MQD_df_LAO.Manufacturer_GROUPED == 'Central'),
        'Manufacturer_GROUPED'] = 'Central Pharmaceutical Technical Development Company'
    MQD_df_LAO.loc[
        (MQD_df_LAO.Manufacturer_GROUPED == 'Conety dophaduoc phamtrung')
        | (MQD_df_LAO.Manufacturer_GROUPED == 'Congty'),
        'Manufacturer_GROUPED'] = 'Conety dophaduoc phamtrung'
    MQD_df_LAO.loc[
        (MQD_df_LAO.Manufacturer_GROUPED == 'Mekophar Chemical Pharmaceutical Joint-Stock Co.')
        | (MQD_df_LAO.Manufacturer_GROUPED == 'Mekophar'),
        'Manufacturer_GROUPED'] = 'Mekophar'
    MQD_df_LAO.loc[
        (MQD_df_LAO.Manufacturer_GROUPED == 'Phammin Den')
        | (MQD_df_LAO.Manufacturer_GROUPED == 'Pham Minh Dan'),
        'Manufacturer_GROUPED'] = 'Pham Minh Dan'
    MQD_df_LAO.loc[
        (MQD_df_LAO.Manufacturer_GROUPED == 'Thailand')
        | (MQD_df_LAO.Manufacturer_GROUPED == 'Thai'),
        'Manufacturer_GROUPED'] = 'Thailand'

    # MOZAMBIQUE
    # Province_Name
    templist = MQD_df_MOZ['Province_Name'].tolist()
    MQD_df_MOZ['Province_Name_GROUPED'] = templist
    # Manual adjustments
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Province_Name_GROUPED == 'Maputo'),
        'Province_Name_GROUPED'] = 'Maputo City'

    # Facility_Location; 'Districts' in Mozambique
    facilityLocationList = [
        'Ancuabe',
        'Balama',
        'Chiúre',
        'Ibo',
        'Macomia' ,
        'Mecúfi' ,
        'Meluco'  ,
        'Metuge',
        'Mocímboa da Praia',
        'Montepuez',
        'Mueda' ,
        'Muidumbe' ,
        'Namuno' ,
        'Nangade' ,
        'Palma',
        'Pemba',
        'Quissanga',
        'Bilene',
        'Chibuto',
        'Chicualacuala',
        'Chigubo' ,
        'Chokwè' ,
        'Chongoene',
        'Guija',
        'Limpopo',
        'Mabalane',
        'Mandlakaze',
        'Mapai',
        'Massangena',
        'Massingir',
        'Xai-Xai',
        'Funhalouro',
        'Govuro' ,
        'Homoine',
        'Inhambane' ,
        'Inharrime',
        'Inhassoro',
        'Jangamo' ,
        'Mabote',
        'Massinga',
        'Maxixe City',
        'Morrumbene',
        'Panda',
        'Vilankulo',
        'Zavala',
        'Barue',
        'Chimoio',
        'Gondola',
        'Guro',
        'Macate',
        'Machaze',
        'Macossa',
        'Manica',
        'Mossurize',
        'Sussundenga',
        'Tambara',
        'Vanduzi',
        'Boane',
        'Magude' ,
        'Manhiça',
        'Marracuene',
        'Matola City',
        'Matutuíne',
        'Moamba',
        'Namaacha',
        'KaMavota',
        'KaMaxakeni',
        'KaMphumu',
        'KaMubukwana',
        'KaNyaka',
        'KaTembe',
        'Nlhamankulu',
        'Angoche',
        'Erati',
        'Ilha de Mocambique',
        'Lalaua',
        'Larde',
        'Liupo',
        'Malema',
        'Meconta',
        'Mecubúri',
        'Memba',
        'Mogincual',
        'Mogovolas',
        'Moma',
        'Monapo',
        'Mossuril',
        'Muecate',
        'Murrupula',
        'Nacala',
        'Nacala-a-Velha',
        'Nacaroa',
        'Nampula City',
        'Rapale',
        'Ribaue',
        'Chimbonila',
        'Cuamba',
        'Lago',
        'Lichinga',
        'Majune',
        'Mandimba',
        'Marrupa',
        'Maua',
        'Mavago',
        'Mecanhelas',
        'Mecula',
        'Metarica',
        'Muembe',
        'Ngauma',
        'Sanga',
        'Beira',
        'Buzi',
        'Caia',
        'Chemba',
        'Cheringoma',
        'Chibabava',
        'Dondo',
        'Gorongosa',
        'Machanga',
        'Maringue',
        'Marromeu',
        'Muanza',
        'Nhamatanda',
        'Angonia',
        'Cahora-Bassa',
        'Changara',
        'Chifunde',
        'Chiuta',
        'Doa',
        'Macanga',
        'Magoe',
        'Marara',
        'Maravia',
        'Moatize',
        'Mutarara',
        'Tete',
        'Tsangano',
        'Zumbo',
        'Alto Molocue',
        'Chinde',
        'Derre',
        'Gile',
        'Gurue',
        'Ile',
        'Inhassunge',
        'Luabo',
        'Lugela',
        'Maganja da Costa',
        'Milange',
        'Mocuba',
        'Mocubela',
        'Molumbo',
        'Mopeia',
        'Morrumbala',
        'Mulevala',
        'Namacurra',
        'Namarroi',
        'Nicoadala',
        'Pebane',
        'Quelimane',
        'Bairro Cimento',
        'Bairro Ingomane',
        'Bairro Natite',
    ]
    MQD_df_MOZ = assignlabels(MQD_df_MOZ, 'Facility_Location', facilityLocationList, thresh=90)
    # Manual adjustments
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Facility_Location == 'Gorongoza')
        | (MQD_df_MOZ.Facility_Location == 'Gorongosa'),
        'Facility_Location_GROUPED'] = 'Gorongosa'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Facility_Location == 'Machava')
        | (MQD_df_MOZ.Facility_Location == 'Machava II'),
        'Facility_Location_GROUPED'] = 'Machava'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Facility_Location == 'Matola')
        | (MQD_df_MOZ.Facility_Location == 'Matola I'),
        'Facility_Location_GROUPED'] = 'Matola'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Facility_Location == 'Maputo (province)')
        | (MQD_df_MOZ.Facility_Location == 'Maputp provincia')
        | (MQD_df_MOZ.Facility_Location == 'Mamputo Provincia'),
        'Facility_Location_GROUPED'] = 'Maputo (province)'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Facility_Location == 'Bairro Cimento'),
        'Facility_Location_GROUPED'] = 'Bairro Cimento'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Facility_Location == 'Bairro Ingomane'),
        'Facility_Location_GROUPED'] = 'Bairro Ingomane'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Facility_Location == 'Bairro Natite'),
        'Facility_Location_GROUPED'] = 'Bairro Natite'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Facility_Location == 'Liberdade'),
        'Facility_Location_GROUPED'] = 'Liberdade'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Facility_Location == 'Maputo (city)'),
        'Facility_Location_GROUPED'] = 'Maputo (city)'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Facility_Location == 'Songo')
        | (MQD_df_MOZ.Facility_Location == 'Cahora-Bassa'),
        'Facility_Location_GROUPED'] = 'Cahora-Bassa'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Facility_Location == 'Ndlavela'),
        'Facility_Location_GROUPED'] = 'Ndlavela'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Facility_Location == 'T3'),
        'Facility_Location_GROUPED'] = 'T3'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Facility_Location == 'Unidade A'),
        'Facility_Location_GROUPED'] = 'Unidade A'

    # Facility_Name
    MQD_df_MOZ = assignlabels(MQD_df_MOZ, 'Facility_Name', thresh=90)
    #todo: Manual checks needed

    # Manufacturer
    MQD_df_MOZ = assignlabels(MQD_df_MOZ, 'Manufacturer', thresh=90)
    # Manual adjustments
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Alicom')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Alicon'),
        'Manufacturer_GROUPED'] = 'Alicon'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Apex')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Apex Drug House'),
        'Manufacturer_GROUPED'] = 'Apex'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Aurobindo')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Aurobindo-Pharma'),
        'Manufacturer_GROUPED'] = 'Aurobindo'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'BDH Industries')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'BDh'),
        'Manufacturer_GROUPED'] = 'BDH Industries'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Bal Pharma')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Bal Pharmaceuticals'),
        'Manufacturer_GROUPED'] = 'Bal Pharmaceuticals'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Bharat')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Bharat - Gujarate')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Bharat Paenterals'),
        'Manufacturer_GROUPED'] = 'Bharat'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'CSPC')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'CSPC Oui Pharmaceutical Ltd')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'CSPC Ouyi')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'CSPC Zhongnuo')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'CSPS'),
        'Manufacturer_GROUPED'] = 'CSPC'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Cadila')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Cadila Pharmaceuticals'),
        'Manufacturer_GROUPED'] = 'Cadila'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Cipla')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Cipla Ltd'),
        'Manufacturer_GROUPED'] = 'Cipla'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Fourrts')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Fourrts Lab. PVT LTD')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Fourtes')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Fourtts'),
        'Manufacturer_GROUPED'] = 'Fourrts'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Fredun')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Fredupharma'),
        'Manufacturer_GROUPED'] = 'Fredun'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Fosun Pharm')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Fuson Pharma'),
        'Manufacturer_GROUPED'] = 'Fosun Pharm'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Global Pharma')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Global Pharma Healthcare'),
        'Manufacturer_GROUPED'] = 'Global Pharma'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Gracura')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Gracure'),
        'Manufacturer_GROUPED'] = 'Gracura'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Guilin Pharmaceutical')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Guilinn'),
        'Manufacturer_GROUPED'] = 'Guilin Pharmaceutical'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Hebei Jiheng (group)')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Hebei jiheng')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Hebeijinh')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Hebel Jihang'),
        'Manufacturer_GROUPED'] = 'Hebei jiheng'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Hetero Drugs')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Hetero Labs'),
        'Manufacturer_GROUPED'] = 'Hetero Drugs'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'IPCA')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Ipca Laboratories'),
        'Manufacturer_GROUPED'] = 'IPCA'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Jiangsu')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Jiangsu Pengyao'),
        'Manufacturer_GROUPED'] = 'Jiangsu Pengyao'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Jiangxi')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Jiaxing Xien')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Jiungxl'),
        'Manufacturer_GROUPED'] = 'Jiangxi'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Kern')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Kern Pharma'),
        'Manufacturer_GROUPED'] = 'Kern'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Kaprin')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Kopran'),
        'Manufacturer_GROUPED'] = 'Kopran'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Lupin')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Kupin')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Lupin Ltd'),
        'Manufacturer_GROUPED'] = 'Lupin'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Lupin')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Kupin')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Lupin Ltd')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Luporn'),
        'Manufacturer_GROUPED'] = 'Lupin'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Macleodos')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Macleods Pharmaceutical'),
        'Manufacturer_GROUPED'] = 'Macleods Pharmaceutical'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Made by Generics, Ltd')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Made for Generics, Ltd')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Generics, Ltd'),
        'Manufacturer_GROUPED'] = 'Generics, Ltd'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Maxhael')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Maxheal'),
        'Manufacturer_GROUPED'] = 'Maxheal'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Medicamen')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Medicamen Biotech'),
        'Manufacturer_GROUPED'] = 'Medicamen'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Mercury')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Mercury Labs'),
        'Manufacturer_GROUPED'] = 'Mercury'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Milan')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Milan Laboratories')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Mylan'),
        'Manufacturer_GROUPED'] = 'Milan'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Missing')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Missing on blister')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'NA VALUE')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Not Available'),
        'Manufacturer_GROUPED'] = 'NA VALUE'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Naging')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Nanjing')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Nanjing Baijingyu')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Nanjingsaijingyn')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Nanying')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Nenjing'),
        'Manufacturer_GROUPED'] = 'Nanjing'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Neomed')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Neomedic'),
        'Manufacturer_GROUPED'] = 'Neomedic'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Novartis')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Novartis P.C.- USA for Novartis Pharma AG-Switzerland')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Novartis USA'),
        'Manufacturer_GROUPED'] = 'Novartis'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Rambaxy')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Ranbaxy')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Ranbaxy Laboratories'),
        'Manufacturer_GROUPED'] = 'Ranbaxy'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Reyound')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Reyoung')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Reyoung Pharmeceutical'),
        'Manufacturer_GROUPED'] = 'Reyoung'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Riemser')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Riemser Arzneimittel')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Riemser Pharma'),
        'Manufacturer_GROUPED'] = 'Riemser'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Sinochem')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Sinochem Ningbo'),
        'Manufacturer_GROUPED'] = 'Sinochem'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Stallion')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Stallion Laboratories Pvt')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Stallion Labs. PVT'),
        'Manufacturer_GROUPED'] = 'Stallion'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Strides')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Strides Arcolab'),
        'Manufacturer_GROUPED'] = 'Strides'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Svizera')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Svizera Labs')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Svuzera'),
        'Manufacturer_GROUPED'] = 'Svizera'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Torrent')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Torrent Pharmaceuticals'),
        'Manufacturer_GROUPED'] = 'Torrent'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Umedica')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Umedica lab. PVT'),
        'Manufacturer_GROUPED'] = 'Umedica'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Unique')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Unique Pharmaceutical'),
        'Manufacturer_GROUPED'] = 'Unique'
    MQD_df_MOZ.loc[
        (MQD_df_MOZ.Manufacturer_GROUPED == 'Yanzhon')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Yanzhon xier')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Yanzhou')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Yanzhouxier')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Yanzhow')
        | (MQD_df_MOZ.Manufacturer_GROUPED == 'Yanzhowxierkangtal'),
        'Manufacturer_GROUPED'] = 'Yanzhouxier'

    # PERU
    # Province_Name; 'Regions' in Peru
    templist = MQD_df_PER['Province_Name'].tolist()
    MQD_df_PER['Province_Name_GROUPED'] = templist

    # Facility_Location; 'Provinces' in Peru are the next political sub-division, but the data seem to
    #   reflect instead the streets where the outlets are located
    MQD_df_PER = assignlabels(MQD_df_PER, 'Facility_Location', thresh=90)
    #todo: Manual adjustments

    # Facility_Name
    MQD_df_PER = assignlabels(MQD_df_PER, 'Facility_Name', thresh=90)

    # Manufacturer
    MQD_df_PER = assignlabels(MQD_df_PER, 'Manufacturer', thresh=90)
    # Manual adjustments
    MQD_df_PER.loc[
        (MQD_df_PER.Manufacturer_GROUPED == 'Hersil S.A.')
        | (MQD_df_PER.Manufacturer_GROUPED == 'Hersil S.A. Laboratorios Industriales Farmaceuticos'),
        'Manufacturer_GROUPED'] = 'Hersil S.A.'
    MQD_df_PER.loc[
        (MQD_df_PER.Manufacturer_GROUPED == 'Genfar S.A.')
        | (MQD_df_PER.Manufacturer_GROUPED == 'Genfar Peru S.A.')
        | (MQD_df_PER.Manufacturer_GROUPED == 'Imported for: Gen Far Peru S.A.'),
        'Manufacturer_GROUPED'] = 'Genfar S.A.'
    MQD_df_PER.loc[
        (MQD_df_PER.Manufacturer_GROUPED == 'Cipa S.A.')
        | (MQD_df_PER.Manufacturer_GROUPED == 'Manufactured for: Cipa S.A.'),
        'Manufacturer_GROUPED'] = 'Cipa S.A.'
    MQD_df_PER.loc[
        (MQD_df_PER.Manufacturer_GROUPED == 'Corporacion Infarmasa S.A.')
        | (MQD_df_PER.Manufacturer_GROUPED == 'Manufactured for: Corporacion Infarmasa S.A.'),
        'Manufacturer_GROUPED'] = 'Corporacion Infarmasa S.A.'

    # PHILIPPINES
    # Province_Name; 'Provinces' in the Philippines
    templist = MQD_df_PHI['Province_Name'].tolist()
    MQD_df_PHI['Province_Name_GROUPED'] = templist
    MQD_df_PHI.loc[(MQD_df_PHI.Province_Name_GROUPED == 'CALABARZON'),
                   'Province_Name_GROUPED'] = 'Calabarzon'
    MQD_df_PHI.loc[(MQD_df_PHI.Province_Name_GROUPED == 'region 1 '),
                   'Province_Name_GROUPED'] = 'Region 1'
    MQD_df_PHI.loc[(MQD_df_PHI.Province_Name_GROUPED == 'region7'),
                   'Province_Name_GROUPED'] = 'Region 7'
    MQD_df_PHI.loc[(MQD_df_PHI.Province_Name_GROUPED == 'region9'),
                   'Province_Name_GROUPED'] = 'Region 9'

    # Facility_Location; these appear to mostly be street addresses for the Philippines
    MQD_df_PHI = assignlabels(MQD_df_PHI, 'Facility_Location', thresh=90)
    #todo: Manual adjustments

    # Facility_Name
    MQD_df_PHI = assignlabels(MQD_df_PHI, 'Facility_Name', thresh=90)
    #todo: Manual adjustments

    # Manufacturer
    MQD_df_PHI = assignlabels(MQD_df_PHI, 'Manufacturer', thresh=90)
    # Manual adjustments
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer_GROUPED == 'AM-Europharma')
                   | (MQD_df_PHI.Manufacturer_GROUPED == 'Am-Euro Pharma Corporation'),
                   'Manufacturer_GROUPED'] = 'AM-Europharma'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer_GROUPED == 'Biotech Research Lab Inc.')
                   | (MQD_df_PHI.Manufacturer_GROUPED == 'BRLI'),
                   'Manufacturer_GROUPED'] = 'BRLI'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer_GROUPED == 'Compact Pharmaceutical Corp.')
                   | (MQD_df_PHI.Manufacturer_GROUPED == 'Compact Pharmaceutical Corporation'),
                   'Manufacturer_GROUPED'] = 'Compact'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer_GROUPED == 'Drugmakers Biotech Research Laboratories, Inc.')
                   | (MQD_df_PHI.Manufacturer_GROUPED == 'Drugmakers Laboratories, Inc.'),
                   'Manufacturer_GROUPED'] = 'Drugmakers Laboratories, Inc.'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer_GROUPED == 'J.M. Tolman Laboratories, Inc.')
                   | (MQD_df_PHI.Manufacturer_GROUPED == 'J.M. Tolmann Lab. Inc.')
                   | (MQD_df_PHI.Manufacturer_GROUPED == 'Tolmann'),
                   'Manufacturer_GROUPED'] = 'J.M. Tolmann Lab. Inc.'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer_GROUPED == 'Lumar Pharmaceutical Lab')
                   | (MQD_df_PHI.Manufacturer_GROUPED == 'Lumar Pharmaceutical Laboratory'),
                   'Manufacturer_GROUPED'] = 'Lumar Pharmaceutical Lab'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer_GROUPED == 'Lupin Limited')
                   | (MQD_df_PHI.Manufacturer_GROUPED == 'Lupin Ltd.'),
                   'Manufacturer_GROUPED'] = 'Lupin Ltd.'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer_GROUPED == 'Missing')
                   | (MQD_df_PHI.Manufacturer_GROUPED == 'No Information Available')
                   | (MQD_df_PHI.Manufacturer_GROUPED == 'No information')
                   | (MQD_df_PHI.Manufacturer_GROUPED == 'NA VALUE'),
                   'Manufacturer_GROUPED'] = 'NA VALUE'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer_GROUPED == 'New Myrex Lab., Inc.')
                    | (MQD_df_PHI.Manufacturer_GROUPED == 'New Myrex Laboratories, Inc.'),
                    'Manufacturer_GROUPED'] = 'New Myrex Lab., Inc.'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer_GROUPED == 'Novartis (Bangladesh)')
                    | (MQD_df_PHI.Manufacturer_GROUPED == 'Novartis'),
                    'Manufacturer_GROUPED'] = 'Novartis'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer_GROUPED == 'Pascual Lab. Inc.')
                   | (MQD_df_PHI.Manufacturer_GROUPED == 'Pascual Laboratories, Inc.'),
                   'Manufacturer_GROUPED'] = 'Pascual Lab. Inc.'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer_GROUPED == 'Pharex Health Corp.')
                   | (MQD_df_PHI.Manufacturer_GROUPED == 'Pharex'),
                   'Manufacturer_GROUPED'] = 'Pharex'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer_GROUPED == 'San Marino Lab., Corp.')
                   | (MQD_df_PHI.Manufacturer_GROUPED == 'San Marino Laboratories Corp'),
                   'Manufacturer_GROUPED'] = 'San Marino Laboratories Corp'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer_GROUPED == 'Sandoz South Africa Ltd.')
                   | (MQD_df_PHI.Manufacturer_GROUPED == 'Sandoz Private Ltd.')
                   | (MQD_df_PHI.Manufacturer_GROUPED == 'Sandoz Philippines Corp.')
                   | (MQD_df_PHI.Manufacturer_GROUPED == 'Sandoz GmbH')
                   | (MQD_df_PHI.Manufacturer_GROUPED == 'Sandoz'),
                   'Manufacturer_GROUPED'] = 'Sandoz'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer_GROUPED == 'The Generics Pharmacy Inc.')
                   | (MQD_df_PHI.Manufacturer_GROUPED == 'TGP'),
                   'Manufacturer_GROUPED'] = 'TGP'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer_GROUPED == 'Wyeth Pakistan Limited')
                   | (MQD_df_PHI.Manufacturer_GROUPED == 'Wyeth Pakistan Ltd.'),
                   'Manufacturer_GROUPED'] = 'Wyeth Pakistan Ltd.'

    # SENEGAL
    # Province_Name
    templist = MQD_df_SEN['Province_Name'].tolist()
    MQD_df_SEN['Province_Name_GROUPED'] = templist

    # Facility_Location and Facility_Name are clearly switched for samples from a certain date
    tempOutletNames = MQD_df_SEN[(MQD_df_SEN['Date_Sample_Collected'].isin(['6/4/2010','6/23/2010','6/24/2010'])) & (MQD_df_SEN['Date_Received']=='7/12/2010')]['Facility_Location'].tolist()
    tempLocationNames = MQD_df_SEN[
        (MQD_df_SEN['Date_Sample_Collected'].isin(['6/4/2010', '6/23/2010', '6/24/2010'])) & (
                    MQD_df_SEN['Date_Received'] == '7/12/2010')]['Facility_Name'].tolist()
    MQD_df_SEN.loc[
        (MQD_df_SEN['Date_Sample_Collected'].isin(['6/4/2010', '6/23/2010', '6/24/2010'])) & (
                MQD_df_SEN['Date_Received'] == '7/12/2010'), 'Facility_Name' ] = tempOutletNames
    MQD_df_SEN.loc[
        (MQD_df_SEN['Date_Sample_Collected'].isin(['6/4/2010', '6/23/2010', '6/24/2010'])) & (
                MQD_df_SEN['Date_Received'] == '7/12/2010'), 'Facility_Location'] = tempLocationNames

    # Facility_Location
    MQD_df_SEN = assignlabels(MQD_df_SEN, 'Facility_Location', thresh=90)
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Location_GROUPED == 'Centre de Santé Mbacké')
                   | (MQD_df_SEN.Facility_Location_GROUPED == 'Centre de Santé Mbacké  tel 33976-49-82'),
                   'Facility_Location_GROUPED'] = 'Centre de Santé Mbacké'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Location_GROUPED == 'Centre de Santé Diourbel')
                   | (MQD_df_SEN.Facility_Location_GROUPED == 'Centre de Santé de Diourbel tel : 33971-28-64'),
                   'Facility_Location_GROUPED'] = 'Centre de Santé Diourbel'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Location_GROUPED == 'Guediawaye')
                   | (MQD_df_SEN.Facility_Location_GROUPED == 'GuÃƒÂ©diawaye'),
                   'Facility_Location_GROUPED'] = 'Guediawaye'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Location_GROUPED == 'Hopital Diourbel')
                   | (MQD_df_SEN.Facility_Location_GROUPED == 'HÃ´pital Diourbel tel : 33971-15-35'),
                   'Facility_Location_GROUPED'] = 'Hopital Diourbel'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Location_GROUPED == '150m hpt kaff')
                    | (MQD_df_SEN.Facility_Location_GROUPED == 'Kaffrine (City)'),
                    'Facility_Location_GROUPED'] = 'Kaffrine (City)'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Location_GROUPED == 'Kanel')
                   | (MQD_df_SEN.Facility_Location_GROUPED == 'kanel')
                   | (MQD_df_SEN.Facility_Location_GROUPED == 'kanel, B.P.11. tel 33.966.70.70'),
                   'Facility_Location_GROUPED'] = 'Kanel'
    MQD_df_SEN.loc[
        (MQD_df_SEN.Facility_Location_GROUPED == 'Dr Dame SECK avenue J.F.KENNEDY B.P.157 tel/FAX941.17.11')
        | (MQD_df_SEN.Facility_Location_GROUPED == 'Dr Assane TOURE tel 33.941.28.29 BP.53 Email boubakh@orange.sn')
        | (MQD_df_SEN.Facility_Location_GROUPED == 'Kaolack (City)'),
        'Facility_Location_GROUPED'] = 'Kaolack (City)'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Location_GROUPED == 'kebemer')
                    | (MQD_df_SEN.Facility_Location_GROUPED == 'Kebemer'),
                    'Facility_Location_GROUPED'] = 'Kebemer'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Location_GROUPED == 'Quartier Caba Club Kedougou')
                    | (MQD_df_SEN.Facility_Location_GROUPED == 'Quartier Gomba n° 630 Kedougou')
                    | (MQD_df_SEN.Facility_Location_GROUPED == 'Kedougou')
                    | (MQD_df_SEN.Facility_Location_GROUPED == 'Kedougou (City)'),
                    'Facility_Location_GROUPED'] = 'Kedougou (City)'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Location_GROUPED == 'koumpantoum')
                    | (MQD_df_SEN.Facility_Location_GROUPED == 'Koumpantoum'),
                    'Facility_Location_GROUPED'] = 'Koumpantoum'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Location_GROUPED == 'Quartier: Sare Moussa Tél. 33 996 23 47')
                    | (MQD_df_SEN.Facility_Location_GROUPED == 'Quartier Sikilo, route de Tripano, Kolda')
                    | (MQD_df_SEN.Facility_Location_GROUPED == 'Quartier Centre II Tél : 33 997-11-58')
                    | (MQD_df_SEN.Facility_Location_GROUPED == 'Kolda')
                    | (MQD_df_SEN.Facility_Location_GROUPED == 'KOLDA')
                    | (MQD_df_SEN.Facility_Location_GROUPED == 'kolda Tel : 33 996 86 05')
                    | (MQD_df_SEN.Facility_Location_GROUPED == 'Kolda (City)'),
                    'Facility_Location_GROUPED'] = 'Kolda (City)'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Location_GROUPED == 'Matam')
                   | (MQD_df_SEN.Facility_Location_GROUPED == 'matam')
                   | (MQD_df_SEN.Facility_Location_GROUPED == 'Matam (City)')
                   | (MQD_df_SEN.Facility_Location_GROUPED == 'Matam, B.P.02. tel:33.966.62.79'),
                   'Facility_Location_GROUPED'] = 'Matam (City)'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Location_GROUPED == 'Mbour-Thiès')
                    | (MQD_df_SEN.Facility_Location_GROUPED == 'Mbour-Thies'),
                    'Facility_Location_GROUPED'] = 'Mbour-Thies'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Location_GROUPED == 'Médina')
                    | (MQD_df_SEN.Facility_Location_GROUPED == 'Medina'),
                    'Facility_Location_GROUPED'] = 'Medina'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Location_GROUPED == 'Ouro-Sogui')
                   | (MQD_df_SEN.Facility_Location_GROUPED == 'Ouro-Sogui, Matam, B.P.49. tel:33.966.10.50')
                   | (MQD_df_SEN.Facility_Location_GROUPED == 'Ouro-Sogui, Matam, tel:33.966.11.22')
                   | (MQD_df_SEN.Facility_Location_GROUPED == 'ouro-Sogui, Matam, B.P.120. tel:33.966.12.78'),
                   'Facility_Location_GROUPED'] = 'Ouro-Sogui'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Location_GROUPED == 'PRA Diourbel')
                   | (MQD_df_SEN.Facility_Location_GROUPED == 'PRA Diourbel tel : 33971-23-92'),
                   'Facility_Location_GROUPED'] = 'PRA Diourbel'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Location_GROUPED == 'saint louis')
                    | (MQD_df_SEN.Facility_Location_GROUPED == 'Sor Saint Louis')
                    | (MQD_df_SEN.Facility_Location_GROUPED == 'pharmacie Mame Madia')
                    | (MQD_df_SEN.Facility_Location_GROUPED == 'Saint Louis (Dept)')
                    | (MQD_df_SEN.Facility_Location_GROUPED == 'Sor Saint Louis SAINT LOUIS'),
                    'Facility_Location_GROUPED'] = 'Saint Louis (City)'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Location_GROUPED == 'BP 60 TAMBACOUNDA SENEGAL')
                    | (MQD_df_SEN.Facility_Location_GROUPED == 'Quartier Abattoir Tambacounda')
                    | (MQD_df_SEN.Facility_Location_GROUPED == 'Quartier Pout, Avenue Léopold S, Senghor Tambacounda')
                    | (MQD_df_SEN.Facility_Location_GROUPED == 'Tambacounda'),
                    'Facility_Location_GROUPED'] = 'Tambacounda'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Location_GROUPED == 'Thiès')
                    | (MQD_df_SEN.Facility_Location_GROUPED == 'Thies'),
                    'Facility_Location_GROUPED'] = 'Thies'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Location_GROUPED == 'velingara TÃ©l : 33 997 - 11- 10')
                   | (MQD_df_SEN.Facility_Location_GROUPED == 'Velingara'),
                   'Facility_Location_GROUPED'] = 'Velingara'

    # Facility_Name
    MQD_df_SEN = assignlabels(MQD_df_SEN, 'Facility_Name', thresh=90)
    # Manual adjustments
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'centre de santÃ© de dioum')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Centre de Sante de Dioum'),
                    'Facility_Name_GROUPED'] = 'Centre de Sante de Dioum'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'centre de santÃ© de kanel')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Centre de Sante de Kanel'),
                    'Facility_Name_GROUPED'] = 'Centre de Sante de Kanel'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'centre de sante de kolda')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Centre de Sante de Kolda'),
                    'Facility_Name_GROUPED'] = 'Centre de Sante de Kolda'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'centre de santÃ© de koumpantoum')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Centre de Sante de Koumpantoum'),
                    'Facility_Name_GROUPED'] = 'Centre de Sante de Koumpantoum'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'centre de santé de matam')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Centre de Sante de Matam'),
                    'Facility_Name_GROUPED'] = 'Centre de Sante de Matam'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'centre de sante de velingara')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Centre de Sante de Velingara'),
                    'Facility_Name_GROUPED'] = 'Centre de Sante de Velingara'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'centre de sante de kedougou')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Centre de Sante de Kedougou'),
                    'Facility_Name_GROUPED'] = 'Centre de Sante de Kedougou'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'Centre de SantÃƒï¿½Ã¯Â¿Â½Ãƒï¿½Ã‚Â© de Richard Toll')
                   | (MQD_df_SEN.Facility_Name_GROUPED == 'Centre de Santé de Richard Toll')
                   | (MQD_df_SEN.Facility_Name_GROUPED == 'Centre de Sante de Richard Toll'),
                   'Facility_Name_GROUPED'] = 'Centre de Sante de Richard Toll'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'centre de sante de Tambacounda')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Centre de Sante de Tambacounda'),
                    'Facility_Name_GROUPED'] = 'Centre de Sante de Tambacounda'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'Centre de SantÃ© de Diourbel tel : 33971-28-64')
                   | (MQD_df_SEN.Facility_Name_GROUPED == 'Centre de Santé Diourbel')
                   | (MQD_df_SEN.Facility_Name_GROUPED == 'Centre de Sante Diourbel'),
                   'Facility_Name_GROUPED'] = 'Centre de Sante Diourbel'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'Centre de Santé Mbacké')
                   | (MQD_df_SEN.Facility_Name_GROUPED == 'Centre de Santé Mbacké  tel 33976-49-82')
                   | (MQD_df_SEN.Facility_Name_GROUPED == 'Centre de Sante Mbacke'),
                   'Facility_Name_GROUPED'] = 'Centre de Sante Mbacke'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'Centre de santé Ousmane Ngom')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Centre de Sante Ousmane Ngom'),
                    'Facility_Name_GROUPED'] = 'Centre de Sante Ousmane Ngom'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'Centre de Santé Roi Baudouin')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Centre de Sante Roi Baudouin'),
                    'Facility_Name_GROUPED'] = 'Centre de Sante Roi Baudouin'
    MQD_df_SEN.loc[
        (MQD_df_SEN.Facility_Name_GROUPED == 'Centre de traitement de la tuberculose d eTouba  tel : 33978-13-71')
        | (MQD_df_SEN.Facility_Name_GROUPED == 'Centre de Traitement de la Tuberculose de Touba'),
        'Facility_Name_GROUPED'] = 'Centre de Traitement de la Tuberculose de Touba'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'centre Hospitalier Régional de Thiès')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Centre Hospitalier Regional de Thies'),
                    'Facility_Name_GROUPED'] = 'Centre Hospitalier Regional de Thies'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'District Sanitaire Touba tel: 33-978-13-70')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'District Sanitaire Touba'),
                    'Facility_Name_GROUPED'] = 'District Sanitaire Touba'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'Hopital de Dioum')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Hopital de DIOUM'),
                    'Facility_Name_GROUPED'] = 'Hopital de Dioum'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'HÃ´pital Diourbel tel : 33971-15-35')
                   | (MQD_df_SEN.Facility_Name_GROUPED == 'Hopital Diourbel'),
                   'Facility_Name_GROUPED'] = 'Hopital Diourbel'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'hopitale regionale de koda')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Hopitale Regionale de Koda'),
                    'Facility_Name_GROUPED'] = 'Hopitale Regionale de Koda'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'hôpital régionale de ouro-sogui')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Hopital Regionale de Ouro-Sogui'),
                    'Facility_Name_GROUPED'] = 'Hopital Regionale de Ouro-Sogui'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'hopital rÃƒÂ©gional de saint louis')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Hopital Regional de Saint Louis'),
                    'Facility_Name_GROUPED'] = 'Hopital Regional de Saint Louis'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'Hopitale regionale de Tambacounda')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Hopitale Regionale de Tambacounda'),
                    'Facility_Name_GROUPED'] = 'Hopitale Regionale de Tambacounda'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'Hopital Touba tel: 33-978-13-70')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Hopital Touba'),
                    'Facility_Name_GROUPED'] = 'Hopital Touba'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'pharmacie')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Pharmacie'),
                    'Facility_Name_GROUPED'] = 'Pharmacie'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'pharmacie awa barry')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Pharmacie Awa Barry'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Awa Barry'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'pharmacie Babacar sy')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Pharmacie Babacar Sy'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Babacar Sy'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'pharmacie boubakh')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Pharmacie Boubakh'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Boubakh'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'pharmacie Ceikh Ousmane Mbacké')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Pharmacie Ceikh Ousmane Mbacke'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Ceikh Ousmane Mbacke'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'Pharmacie centrale  Dr A. Camara tel : 33971-11-20 Diourbel')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Pharmacie Centrale Dr A.C.'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Centrale Dr A.C.'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == "pharmacie Château d'Eau")
                    | (MQD_df_SEN.Facility_Name_GROUPED == "Pharmacie Chateau d'Eau"),
                    'Facility_Name_GROUPED'] = "Pharmacie Chateau d'Eau"
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'pharmacie cheikh tidiane')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Pharmacie Cheikh Tidiane'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Cheikh Tidiane'
    MQD_df_SEN.loc[
        (MQD_df_SEN.Facility_Name_GROUPED == 'Pharmacie du Baool Dr EL Badou Cissé tel :  33971-10-58   Diourbel')
        | (MQD_df_SEN.Facility_Name_GROUPED == 'Pharmacie du Baool Dr El-B.C.'),
        'Facility_Name_GROUPED'] = 'Pharmacie du Baool Dr El-B.C.'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'pharmacie du Fleuve')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Pharmacie du Fleuve'),
                    'Facility_Name_GROUPED'] = 'Pharmacie du Fleuve'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'pharmacie El hadj omar Tall')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Pharmacie El Hadj Omar Tall'),
                    'Facility_Name_GROUPED'] = 'Pharmacie El Hadj Omar Tall'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'pharmacie FOULADOU')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Pharmacie Fouladou'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Fouladou'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'pharmacie KANCISSE')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Pharmacie Kancisse'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Kancisse'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'pharmacie KOLDA')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Pharmacie Kolda'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Kolda'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'Pharmacie Mame Diarra Bousso Dr Yéro Diouma Dian  tel: 33-971-34-35 Diourbel')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Pharmacie Mame Diarra Bousso Dr Y.D.D.'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Mame Diarra Bousso Dr Y.D.D.'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'PHARMACIE MAME MADIA')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Pharmacie Mame Madia'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Mame Madia'
    MQD_df_SEN.loc[
        (MQD_df_SEN.Facility_Name_GROUPED == 'Pharmacie Mame  Ibrahima Ndour Dr Alassane Ndour Tél: 339760097 Mbacké')
        | (MQD_df_SEN.Facility_Name_GROUPED == 'Pharmacie Mame Ibrahima Ndour Dr A.N.'),
        'Facility_Name_GROUPED'] = 'Pharmacie Mame Ibrahima Ndour Dr A.N.'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'Pharmacie Ndamatou  Dr Omar Niasse tel : 33978-17-68Touba')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Pharmacie Ndamatou Dr O.N.'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Ndamatou Dr O.N.'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'pharmacie ousmane')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Pharmacie Ousmane'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Ousmane'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == "Pharmacie RÃƒÂ©gionale d' Approvisionnement de Saint Louis")
                    | (MQD_df_SEN.Facility_Name_GROUPED == "Pharmacie Regionale d' Approvisionnement de Saint Louis"),
                    'Facility_Name_GROUPED'] = "Pharmacie Regionale d' Approvisionnement de Saint Louis"
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'pharmacie sogui')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'Pharmacie Sogui'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Sogui'
    MQD_df_SEN.loc[
        (MQD_df_SEN.Facility_Name_GROUPED == 'Pharmacie Touba MosquÃ©e  Dr Amadou Malick Kane Tel : 33974-89-74')
        | (MQD_df_SEN.Facility_Name_GROUPED == 'Pharmacie Touba Mosque Dr A.M.K.'),
        'Facility_Name_GROUPED'] = 'Pharmacie Touba Mosque Dr A.M.K.'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'PRA Diourbel')
                   | (MQD_df_SEN.Facility_Name_GROUPED == 'PRA Diourbel tel : 33971-23-92'),
                   'Facility_Name_GROUPED'] = 'PRA Diourbel'
    MQD_df_SEN.loc[(MQD_df_SEN.Facility_Name_GROUPED == 'PRA Thiès')
                    | (MQD_df_SEN.Facility_Name_GROUPED == 'PRA Thies'),
                    'Facility_Name_GROUPED'] = 'PRA Thies'

    # Manufacturer
    templist = MQD_df_SEN['Manufacturer'].tolist()
    MQD_df_SEN['Manufacturer_GROUPED'] = templist

    '''
    a = MQD_df_SEN['Province_Name_GROUPED'][SEN_df['Date_Received'] == '7/12/2010'].astype('str').unique()
    print(len(a))
    for item in sorted(a):
        print(item)
    '''

    # THAILAND
    # Province_Name: 'Provinces' of Thailand
    templist = MQD_df_THA['Province_Name'].tolist()
    MQD_df_THA['Province_Name_GROUPED'] = templist

    # Facility_Location: Largely missing for Thailand
    templist = MQD_df_THA['Facility_Location'].tolist()
    MQD_df_THA['Facility_Location_GROUPED'] = templist

    # Facility_Name; need to remove unknown characters first before running assignlabels()
    MQD_df_THA.loc[(MQD_df_THA.Facility_Name == '*?????????????????')
                   | (MQD_df_THA.Facility_Name == '?.???????')
                   | (MQD_df_THA.Facility_Name == '?.????????')
                   | (MQD_df_THA.Facility_Name == '??. ?????')
                   | (MQD_df_THA.Facility_Name == '??. ???????')
                   | (MQD_df_THA.Facility_Name == '??.????')
                   | (MQD_df_THA.Facility_Name == '??.???????')
                   | (MQD_df_THA.Facility_Name == '??.????????')
                   | (MQD_df_THA.Facility_Name == '??.??????????????????')
                   | (MQD_df_THA.Facility_Name == '???.4.1 ?????')
                   | (MQD_df_THA.Facility_Name == '???.4.1 ???????')
                   | (MQD_df_THA.Facility_Name == '???.4.1.2')
                   | (MQD_df_THA.Facility_Name == '????')
                   | (MQD_df_THA.Facility_Name == '?????')
                   | (MQD_df_THA.Facility_Name == '?????????????????')
                   | (MQD_df_THA.Facility_Name == '????????????????')
                   | (MQD_df_THA.Facility_Name == '???????????????')
                   | (MQD_df_THA.Facility_Name == '??????????????')
                   | (MQD_df_THA.Facility_Name == '?????????????')
                   | (MQD_df_THA.Facility_Name == '??????')
                   | (MQD_df_THA.Facility_Name == '????????????')
                   | (MQD_df_THA.Facility_Name == '???????????')
                   | (MQD_df_THA.Facility_Name == '?????????2')
                   | (MQD_df_THA.Facility_Name == '??????????')
                   | (MQD_df_THA.Facility_Name == '?????????')
                   | (MQD_df_THA.Facility_Name == '??????? ?????????')
                   | (MQD_df_THA.Facility_Name == '???????')
                   | (MQD_df_THA.Facility_Name == '??.??????'),
                   'Facility_Name'] = 'NA VALUE'

    MQD_df_THA = assignlabels(MQD_df_THA, 'Facility_Name', thresh=90)
    #todo: Manual adjustments

    # Manufacturer
    MQD_df_THA = assignlabels(MQD_df_THA, 'Manufacturer', thresh=90)
    # Manual adjustments
    MQD_df_THA.loc[(MQD_df_THA.Manufacturer_GROUPED == 'Guilin Pharmaceutical Co., Ltd./Atlantic Laboratories Co., Ltd')
                   | (MQD_df_THA.Manufacturer_GROUPED == 'Guilin Pharmaceutical Co., Ltd'),
                   'Manufacturer_GROUPED'] = 'Guilin Pharmaceutical Co., Ltd'
    MQD_df_THA.loc[(MQD_df_THA.Manufacturer_GROUPED == 'Ubison')
                   | (MQD_df_THA.Manufacturer_GROUPED == 'Unison Laboratories Co., Ltd'),
                   'Manufacturer_GROUPED'] = 'Unison Laboratories Co., Ltd'
    MQD_df_THA.loc[(MQD_df_THA.Manufacturer_GROUPED == 'Weatsgo Pharma')
                   | (MQD_df_THA.Manufacturer_GROUPED == 'Wellgo Pharmaceutical')
                   | (MQD_df_THA.Manufacturer_GROUPED == 'Wesgo Pharmacutical Co., Ltd'),
                   'Manufacturer_GROUPED'] = 'Wellgo Pharmaceutical'

    # VIETNAM
    # Province_Name
    templist = MQD_df_VIE['Province_Name'].tolist()
    MQD_df_VIE['Province_Name_GROUPED'] = templist

    # Facility_Location; 'Districts' or 'Communes' in Vietnam
    facilityLocationList = [
        'An Nhon district',
        'An Thoi Dong District',
        'Ba Thuoc District',
        'Bac Quang District',
        'Bim Son Town',
        'Binh Chanh District',
        'Binh Long District',
        'Bo Trach District',
        'Bom Bo District',
        'Bu Dang District',
        'Bu Dop District',
        'Bu Gia Map District',
        'Buon Don District',
        'Buon Me Thuot City',
        'Cam Lo District',
        'Cam Thuy District',
        'Can Gio District',
        'Can Loc District',
        'Chon Thanh District',
        'Chuong Duong Str',
        'Cu Chi District',
        'Cumgar District',
        'Da Krong District',
        'Dai Loc District',
        'Dak Glei District',
        'Dak Ha District',
        'Dak Lak',
        'Dak To District',
        'Dako District',
        'Dakrong District',
        'Dien Bien District',
        'Dien Bien Dong District',
        'Dien Bien Phu City',
        'District No. 5',
        'District No. 6',
        'Dong Ha Market',
        'Dong Ha Town',
        'Dong Phu District',
        'Dong Van District',
        'Dong Xoai District',
        'Du Dang District',
        'Duy Xuyen District',
        'Eahleo District',
        'Eakar District',
        'Easup District',
        'Gia Lai Town',
        'Gia Map District',
        'Gio Linh District',
        'Ha Giang City',
        'Ha Tinh City',
        'Hau Loc District',
        'Hoai An District',
        'Hoai Nhon District',
        'Hooc Mon District',
        'Huong Hoa District',
        'Huong Khe District',
        'IAPA D',
        'Khe Sanh Town',
        'Kon Plong District',
        'Kon Tum Town',
        'Konp Long District',
        'Konplong Town',
        'Kron Plong District',
        'Krong Ana District',
        'Krong Bong District',
        'Krong Buk District',
        'Krong Jing Town',
        'Krong Klang Town',
        'Krong Long District',
        'Krong Na District',
        'Krong Pac District',
        'Krongchro District',
        'Kronp Ray District',
        'Ky Son District',
        'Lang Chanh District',
        'Lao Bao District',
        'Le Thuy District',
        'Lien Son District',
        'Loc Hung Commune',
        'Loc Ninh District',
        'Ma Dak District',
        'Madrak District',
        'Mdrak District',
        'Meo Vac District',
        'Missing',
        'Muong Ang district',
        'Muong Cha district',
        'Muong La district',
        'Muong Lat District',
        'Muong Lay District',
        'Muong Nhe District',
        'Muong Te District',
        'Muong Tra District',
        'Nga Son district',
        'Ngoc Hoi District',
        'Ngoc Lac District',
        'Nguyen Hue Str',
        'Nha Be District',
        'Nha Bich Commune',
        'Phouc Long District',
        'Phu My District',
        'Phuoc Long District',
        'Pleiku City',
        'Quan Ba district',
        'Quan Hoa District',
        'Quang Ba District',
        'Quang Binh City',
        'Quang Tri City',
        'Quang Tri District',
        'Quang Trung District',
        'Quy Nhon City',
        'Son La City',
        'Song Ma district',
        'Tam Ky City',
        'Tan Bien District',
        'Tan Binh Precinct',
        'Tan Chau District',
        'Tay Ninh District',
        'Tay Ninh Town',
        'Tay Son District',
        'Thanh Hoa City',
        'Thanh Luong Commune',
        'Tho Xuan District',
        'Thuan Phu Commune',
        'Trung Ha District',
        'Trung Thanh District',
        'Tua Chua District',
        'Tuan Giao District',
        'Tuong Duong District',
        'Tuy Phuoc district',
        'Unknown',
        'Van Canh district',
        'Vi Xuyen district',
        'Vinh City',
        'Vinh Linh District',
        'Vinh Loc district',
        'Vinh Market',
        'Yen Minh district'
    ]
    MQD_df_VIE = assignlabels(MQD_df_VIE, 'Facility_Location', facilityLocationList, thresh=90)
    # Manual adjustments
    MQD_df_VIE.loc[(MQD_df_VIE.Facility_Location_GROUPED == 'Missing')
                   | (MQD_df_VIE.Facility_Location_GROUPED == 'NA VALUE')
                   | (MQD_df_VIE.Facility_Location_GROUPED == 'Unknown')
                   | (MQD_df_VIE.Facility_Location == 'T'),
                   'Facility_Location_GROUPED'] = 'NA VALUE'
    MQD_df_VIE.loc[(MQD_df_VIE.Facility_Location == 'Buon Ma Thout Town')
                   | (MQD_df_VIE.Facility_Location == 'Buon Me Thuot City'),
                   'Facility_Location_GROUPED'] = 'Buon Me Thuot City'
    MQD_df_VIE.loc[(MQD_df_VIE.Facility_Location == 'Dak Ha Town')
                   | (MQD_df_VIE.Facility_Location == 'Dak Ha District'),
                   'Facility_Location_GROUPED'] = 'Dak Ha District'
    MQD_df_VIE.loc[(MQD_df_VIE.Facility_Location == 'Dakrong Commune')
                   | (MQD_df_VIE.Facility_Location == 'Dakrong District'),
                   'Facility_Location_GROUPED'] = 'Dakrong District'
    MQD_df_VIE.loc[(MQD_df_VIE.Facility_Location == 'Dien Bien Phu Town')
                   | (MQD_df_VIE.Facility_Location == 'Dien Bien Phu City'),
                   'Facility_Location_GROUPED'] = 'Dien Bien Phu City'
    MQD_df_VIE.loc[(MQD_df_VIE.Facility_Location == 'Dong Xoai Town')
                   | (MQD_df_VIE.Facility_Location == 'Dong Xoai District'),
                   'Facility_Location_GROUPED'] = 'Dong Xoai District'
    MQD_df_VIE.loc[(MQD_df_VIE.Facility_Location == 'Ha Giang Town')
                   | (MQD_df_VIE.Facility_Location == 'Ha Giang City'),
                   'Facility_Location_GROUPED'] = 'Ha Giang City'
    MQD_df_VIE.loc[(MQD_df_VIE.Facility_Location == 'Khe Sanh Market')
                   | (MQD_df_VIE.Facility_Location == 'Khe Sanh Str.')
                   | (MQD_df_VIE.Facility_Location == 'Khe Sanh Town'),
                   'Facility_Location_GROUPED'] = 'Khe Sanh Town'
    MQD_df_VIE.loc[(MQD_df_VIE.Facility_Location == 'Quang Trung Str')
                   | (MQD_df_VIE.Facility_Location == 'Quang Trung Town')
                   | (MQD_df_VIE.Facility_Location == 'Quang Trung District'),
                   'Facility_Location_GROUPED'] = 'Quang Trung District'
    MQD_df_VIE.loc[(MQD_df_VIE.Facility_Location == 'Qui Nhon Town')
                   | (MQD_df_VIE.Facility_Location == 'Quy Nhon City'),
                   'Facility_Location_GROUPED'] = 'Quy Nhon City'

    # Facility_Name
    MQD_df_VIE = assignlabels(MQD_df_VIE, 'Facility_Name', thresh=90)
    #todo: Manual adjustments

    # Manufacturer
    MQD_df_VIE = assignlabels(MQD_df_VIE, 'Manufacturer', thresh=90)
    # Manual adjustments; consider setting back the manufacturers with different numbers
    MQD_df_VIE.loc[(MQD_df_VIE.Manufacturer_GROUPED == 'Bin Thuan Medical Materials Pharmaceutical Joint Stock Company')
                   | (MQD_df_VIE.Manufacturer_GROUPED == 'Binh Thuan Pharmaceutical & Material JSC'),
                   'Manufacturer_GROUPED'] = 'Binh Thuan Pharmaceutical & Material JSC'
    MQD_df_VIE.loc[(MQD_df_VIE.Manufacturer_GROUPED == 'Brawn Laboratories Limited')
                   | (MQD_df_VIE.Manufacturer_GROUPED == 'Brawn lab Ltd'),
                   'Manufacturer_GROUPED'] = 'Brawn Laboratories Limited'
    MQD_df_VIE.loc[(MQD_df_VIE.Manufacturer_GROUPED == 'Cuu Long Pharmaceutical JSC')
                   | (MQD_df_VIE.Manufacturer_GROUPED == 'Cuu Long Pharmaceutical Joint-Stock Co.'),
                   'Manufacturer_GROUPED'] = 'Cuu Long Pharmaceutical JSC'
    MQD_df_VIE.loc[(MQD_df_VIE.Manufacturer_GROUPED == 'Danapha Pharmaceutical JSC')
                   | (MQD_df_VIE.Manufacturer_GROUPED == 'Danapha'),
                   'Manufacturer_GROUPED'] = 'Danapha'
    MQD_df_VIE.loc[(MQD_df_VIE.Manufacturer_GROUPED == 'H6 Pharm')
                   | (MQD_df_VIE.Manufacturer_GROUPED == 'HG Pharm')
                   | (MQD_df_VIE.Manufacturer_GROUPED == 'Hau Giang Pharmaceutical JSC')
                   | (MQD_df_VIE.Manufacturer_GROUPED == 'Hau Giang Pharmaceutical Joint-Stock Co.'),
                   'Manufacturer_GROUPED'] = 'HG Pharm'
    MQD_df_VIE.loc[(MQD_df_VIE.Manufacturer_GROUPED == 'ICA Biotechnological Pharmaceutical JSC')
                   | (MQD_df_VIE.Manufacturer_GROUPED == 'ICA Pharmaceutical JSC'),
                   'Manufacturer_GROUPED'] = 'ICA Pharmaceutical JSC'
    MQD_df_VIE.loc[(MQD_df_VIE.Manufacturer_GROUPED == 'Medipharco')
                   | (MQD_df_VIE.Manufacturer_GROUPED == 'Medipharco Tenamyd')
                   | (MQD_df_VIE.Manufacturer_GROUPED == 'Mebiphar')    ,
                   'Manufacturer_GROUPED'] = 'Medipharco'
    MQD_df_VIE.loc[(MQD_df_VIE.Manufacturer_GROUPED == 'Mekopha Pharmaceutical JSC')
                   | (MQD_df_VIE.Manufacturer_GROUPED == 'Mekophar Chemical Pharmaceutical Joint-Stock Co.'),
                   'Manufacturer_GROUPED'] = 'Mekopha Pharmaceutical JSC'
    MQD_df_VIE.loc[(MQD_df_VIE.Manufacturer_GROUPED == 'Micro Labs Ltd')
                   | (MQD_df_VIE.Manufacturer_GROUPED == 'Micro labslemited'),
                   'Manufacturer_GROUPED'] = 'Micro Labs Ltd'
    MQD_df_VIE.loc[(MQD_df_VIE.Manufacturer_GROUPED == 'Minh Dan Pharmaceutical JSC')
                   | (MQD_df_VIE.Manufacturer_GROUPED == 'Minh Dan Pharmaceutical Joint-Stock Co.'),
                   'Manufacturer_GROUPED'] = 'Minh Dan Pharmaceutical JSC'
    MQD_df_VIE.loc[(MQD_df_VIE.Manufacturer_GROUPED == 'Nam Ha Pharmaceutical JSC')
                   | (MQD_df_VIE.Manufacturer_GROUPED == 'Nam Ha Pharmaceutical Joint-Stock Co.'),
                   'Manufacturer_GROUPED'] = 'Nam Ha Pharmaceutical JSC'
    MQD_df_VIE.loc[(MQD_df_VIE.Manufacturer_GROUPED == 'National Pharmaceutical Joint-Stock No 3')
                   | (MQD_df_VIE.Manufacturer_GROUPED == 'National Pharmaceutical company No 3')
                   | (MQD_df_VIE.Manufacturer_GROUPED == 'No. 3 National Pharmaceutical JSC'),
                   'Manufacturer_GROUPED'] = 'National Pharmaceutical Joint-Stock No 3'
    MQD_df_VIE.loc[(MQD_df_VIE.Manufacturer_GROUPED == 'Pune Pharmaceuticals Ltd')
                   | (MQD_df_VIE.Manufacturer_GROUPED == 'Pure Pharma Limited'),
                   'Manufacturer_GROUPED'] = 'Pure Pharma Limited'
    MQD_df_VIE.loc[(MQD_df_VIE.Manufacturer_GROUPED == 'Quang Binh Pharmaceutical JSC')
                   | (MQD_df_VIE.Manufacturer_GROUPED == 'Quang Binh Pharmaceutical Joint-Stock Co.'),
                   'Manufacturer_GROUPED'] = 'Quang Binh Pharmaceutical JSC'
    MQD_df_VIE.loc[(MQD_df_VIE.Manufacturer_GROUPED == 'Standa')
                   | (MQD_df_VIE.Manufacturer_GROUPED == 'Stada Ltd, Co.'),
                   'Manufacturer_GROUPED'] = 'Stada Ltd, Co.'
    MQD_df_VIE.loc[(MQD_df_VIE.Manufacturer_GROUPED == 'Missing')
                   | (MQD_df_VIE.Manufacturer_GROUPED == 'Unknown')
                   | (MQD_df_VIE.Manufacturer_GROUPED == 'NA VALUE'),
                   'Manufacturer_GROUPED'] = 'NA VALUE'
    MQD_df_VIE.loc[(MQD_df_VIE.Manufacturer_GROUPED == 'Vidipha  National Pharmaceutical JSC')
                   | (MQD_df_VIE.Manufacturer_GROUPED == 'Vidipha Pharmaceutical Joint-Stock Co.')
                   | (MQD_df_VIE.Manufacturer_GROUPED == 'Vidiphar Pharmaceutical JSC'),
                   'Manufacturer_GROUPED'] = 'Vidiphar Pharmaceutical JSC'


    '''
    MQD_df_SEN.loc[MQD_df_SEN.Manufacturer == 'nan']
    MQD_df_VIE[(MQD_df_VIE.Manufacturer_GROUPED == 'Standa' )].count()
    a = MQD_df_VIE['Manufacturer_GROUPED'].astype('str').unique()
    print(len(a))
    for item in sorted(a):
        print(item)
    MQD_df_THA.pivot_table(index=['Facility_Name'], columns=['Final_Test_Conclusion'], aggfunc='size', fill_value=0)
    MQD_df_VIE[(MQD_df_VIE.Facility_Location_GROUPED == 'MANUALLY_MODIFY')].pivot_table(
        index=['Facility_Location'], columns=['Facility_Location_GROUPED'], aggfunc='size', fill_value=0)
    '''

    '''
    SUMMARY OF THERAPEUTIC INDICATIONS AND FACILITY TYPES FOR EACH COUNTRY:
        CAMBODIA, 2990 TOTAL OBSVNS:
            THERAPEUTIC INDICATIONS:
                1328 Antibiotic
                557 Antimalarial
            OUTLET-TYPE FACILITIES: 1603
        ETHIOPIA, 663 TOTAL OBSVNS:
            THERAPEUTIC INDICATIONS:
                163 Antibiotic
                281 Antimalarial
            OUTLET-TYPE FACILITIES: 622
        GHANA, 562 TOTAL OBSVNS
            THERAPEUTIC INDICATIONS:
                304 Antimalarial
            OUTLET-TYPE FACILITIES: 525
        KENYA, 904 TOTAL OBSVNS
            THERAPEUTIC INDICATIONS:
                707 Antimalarial
            OUTLET-TYPE FACILITIES: 847
        LAOS, 1784 TOTAL OBSVNS
            THERAPEUTIC INDICATIONS:
                901 Antibiotic
                263 Antimalarial
            OUTLET-TYPE FACILITIES: 1565
        MOZAMBIQUE, 1536 TOTAL OBSVNS
            THERAPEUTIC INDICATIONS:
                520 Antibiotic
                175 Antimalarial
            OUTLET-TYPE FACILITIES: 1240
        PERU, 462 TOTAL OBSVNS
            THERAPEUTIC INDICATIONS:
                271 Antibiotic
            OUTLET-TYPE FACILITIES: 382
        PHILIPPINES, 1870 TOTAL OBSVNS
            THERAPEUTIC INDICATIONS:
                1863 Antituberculosis
            OUTLET-TYPE FACILITIES: 1778
        SENEGAL, 623 TOTAL OBSVNS
            THERAPEUTIC INDICATIONS:
                155 Antimalarial
                110 Antiretroviral
            OUTLET-TYPE FACILITIES: 557
        THAILAND, 2287 TOTAL OBSVNS
            THERAPEUTIC INDICATIONS:
                220 Antibiotic
                263 Antimalarial
            OUTLET-TYPE FACILITIES: 2196
        VIETNAM, 2683 TOTAL OBSVNS
            THERAPEUTIC INDICATIONS:
                666 Antibiotic
                532 Antimalarial
            OUTLET-TYPE FACILITIES: 2161                
    '''
    # Facility-filtered for outlet-type facilities
    MQD_df_CAM_facilityfilter = MQD_df_CAM[MQD_df_CAM['Facility_Type_Name'].isin(
        ['Depot of Pharmacy', 'Consultation Office', 'Consultation cabinet', 'Health care service', 'Health Cabinet',
         'Health Clinic', 'Pharmacie', 'Pharmacy', 'Pharmacy Depot', 'Private Clinic', 'Retail-drug Outlet',
         'Retail drug outlet', 'Clinic'])].copy()
    MQD_df_ETH_facilityfilter = MQD_df_ETH[MQD_df_ETH['Facility_Type_Name'].isin(
        ['Clinic', 'Drug store', 'Health Center', 'Health Clinic', 'Hospital', 'Medium Clinic', 'Other Public',
         'Pharmacy', 'Retail Shop', 'drug shop', 'health office',])].copy()
    MQD_df_GHA_facilityfilter = MQD_df_GHA[MQD_df_GHA['Facility_Type_Name'].isin(
        ['Health Clinic', 'Hospital', 'Pharmacy', 'Retail Shop', 'Retail-drug Outlet'])].copy()
    MQD_df_KEN_facilityfilter = MQD_df_KEN[MQD_df_KEN['Facility_Type_Name'].isin(
        ['Clinic', 'Dispensary', 'Health Centre', 'Health Clinic', 'Hospital', 'Mission Hospital', 'Pharmacy',
         'RETAIL CHEMIST', 'Retail Shop', 'Retailer'])].copy()
    MQD_df_LAO_facilityfilter = MQD_df_LAO[MQD_df_LAO['Facility_Type_Name'].isin(
        ['Clinic', 'Health Clinic', 'Hospital', 'Pharmacy'])].copy()
    MQD_df_MOZ_facilityfilter = MQD_df_MOZ[MQD_df_MOZ['Facility_Type_Name'].isin(
        ['Depot', 'Health Cabinet', 'Health Center', 'Health Center Depot', 'Health Clinic',
         'Health Post Depot', 'Hospital', 'Hospital Depot', 'Pharmacy'])].copy()
    MQD_df_PER_facilityfilter = MQD_df_PER[MQD_df_PER['Facility_Type_Name'].isin(
        ['Hospital', 'Pharmacy', 'Pharmacy ESSALUD'])].copy()
    MQD_df_PHI_facilityfilter = MQD_df_PHI[MQD_df_PHI['Facility_Type_Name'].isin(
        ['Health Center', 'Health Clinic', 'Hospital', 'Hospital Pharmacy', 'Pharmacy',
         'Retail-drug Outlet', 'health office'])].copy()
    MQD_df_SEN_facilityfilter = MQD_df_SEN[MQD_df_SEN['Facility_Type_Name'].isin(
        ['Health Clinic', 'Hospital', 'Pharmacy'])].copy()
    MQD_df_THA_facilityfilter = MQD_df_THA[MQD_df_THA['Facility_Type_Name'].isin(
        ['Health Clinic', 'Hospital', 'Pharmacy', 'Retail-drug Outlet'])].copy()
    MQD_df_VIE_facilityfilter = MQD_df_VIE[MQD_df_VIE['Facility_Type_Name'].isin(
        ['General Clinic', 'Health Clinic', 'Hospital', 'Medical centre', 'Medical station', 'Pharmacy'])].copy()

    # For each desired data set, generate lists suitable for use with logistigate
    # Overall data, #nofilter
    dataTbl_CAM_GEO1 = MQD_df_CAM[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_CAM_GEO1 = [[i[0],i[1],1] if i[2]=='Fail' else [i[0],i[1],0] for i in dataTbl_CAM_GEO1]
    dataTbl_CAM_GEO2 = MQD_df_CAM[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_CAM_GEO2 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_CAM_GEO2]
    dataTbl_CAM_GEO3 = MQD_df_CAM[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_CAM_GEO3 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_CAM_GEO3]
    dataTbl_ETH_GEO1 = MQD_df_ETH[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_ETH_GEO1 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_ETH_GEO1]
    dataTbl_ETH_GEO2 = MQD_df_ETH[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_ETH_GEO2 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_ETH_GEO2]
    dataTbl_ETH_GEO3 = MQD_df_ETH[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_ETH_GEO3 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_ETH_GEO3]
    dataTbl_GHA_GEO1 = MQD_df_GHA[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_GHA_GEO1 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_GHA_GEO1]
    dataTbl_GHA_GEO2 = MQD_df_GHA[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_GHA_GEO2 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_GHA_GEO2]
    dataTbl_GHA_GEO3 = MQD_df_GHA[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_GHA_GEO3 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_GHA_GEO3]
    dataTbl_KEN_GEO1 = MQD_df_KEN[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_KEN_GEO1 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_KEN_GEO1]
    dataTbl_KEN_GEO2 = MQD_df_KEN[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_KEN_GEO2 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_KEN_GEO2]
    dataTbl_KEN_GEO3 = MQD_df_KEN[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_KEN_GEO3 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_KEN_GEO3]
    dataTbl_LAO_GEO1 = MQD_df_LAO[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_LAO_GEO1 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_LAO_GEO1]
    dataTbl_LAO_GEO2 = MQD_df_LAO[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_LAO_GEO2 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_LAO_GEO2]
    dataTbl_LAO_GEO3 = MQD_df_LAO[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_LAO_GEO3 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_LAO_GEO3]
    dataTbl_MOZ_GEO1 = MQD_df_MOZ[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_MOZ_GEO1 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_MOZ_GEO1]
    dataTbl_MOZ_GEO2 = MQD_df_MOZ[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_MOZ_GEO2 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_MOZ_GEO2]
    dataTbl_MOZ_GEO3 = MQD_df_MOZ[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_MOZ_GEO3 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_MOZ_GEO3]
    dataTbl_PER_GEO1 = MQD_df_PER[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_PER_GEO1 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_PER_GEO1]
    dataTbl_PER_GEO2 = MQD_df_PER[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_PER_GEO2 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_PER_GEO2]
    dataTbl_PER_GEO3 = MQD_df_PER[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_PER_GEO3 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_PER_GEO3]
    dataTbl_PHI_GEO1 = MQD_df_PHI[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_PHI_GEO1 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_PHI_GEO1]
    dataTbl_PHI_GEO2 = MQD_df_PHI[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_PHI_GEO2 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_PHI_GEO2]
    dataTbl_PHI_GEO3 = MQD_df_PHI[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_PHI_GEO3 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_PHI_GEO3]
    dataTbl_SEN_GEO1 = MQD_df_SEN[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_SEN_GEO1 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_SEN_GEO1]
    dataTbl_SEN_GEO2 = MQD_df_SEN[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_SEN_GEO2 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_SEN_GEO2]
    dataTbl_SEN_GEO3 = MQD_df_SEN[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_SEN_GEO3 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_SEN_GEO3]
    dataTbl_THA_GEO1 = MQD_df_THA[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_THA_GEO1 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_THA_GEO1]
    dataTbl_THA_GEO2 = MQD_df_THA[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_THA_GEO2 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_THA_GEO2]
    dataTbl_THA_GEO3 = MQD_df_THA[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_THA_GEO3 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_THA_GEO3]
    dataTbl_VIE_GEO1 = MQD_df_VIE[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_VIE_GEO1 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_VIE_GEO1]
    dataTbl_VIE_GEO2 = MQD_df_VIE[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_VIE_GEO2 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_VIE_GEO2]
    dataTbl_VIE_GEO3 = MQD_df_VIE[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_VIE_GEO3 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_VIE_GEO3]

    # Facility-filtered data
    dataTbl_CAM_GEO1_ff = MQD_df_CAM_facilityfilter[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_CAM_GEO1_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_CAM_GEO1_ff]
    dataTbl_CAM_GEO2_ff = MQD_df_CAM_facilityfilter[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_CAM_GEO2_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_CAM_GEO2_ff]
    dataTbl_CAM_GEO3_ff = MQD_df_CAM_facilityfilter[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_CAM_GEO3_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_CAM_GEO3_ff]
    dataTbl_ETH_GEO1_ff = MQD_df_ETH_facilityfilter[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_ETH_GEO1_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_ETH_GEO1_ff]
    dataTbl_ETH_GEO2_ff = MQD_df_ETH_facilityfilter[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_ETH_GEO2_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_ETH_GEO2_ff]
    dataTbl_ETH_GEO3_ff = MQD_df_ETH_facilityfilter[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_ETH_GEO3_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_ETH_GEO3_ff]
    dataTbl_GHA_GEO1_ff = MQD_df_GHA_facilityfilter[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_GHA_GEO1_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_GHA_GEO1_ff]
    dataTbl_GHA_GEO2_ff = MQD_df_GHA_facilityfilter[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_GHA_GEO2_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_GHA_GEO2_ff]
    dataTbl_GHA_GEO3_ff = MQD_df_GHA_facilityfilter[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_GHA_GEO3_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_GHA_GEO3_ff]
    dataTbl_KEN_GEO1_ff = MQD_df_KEN_facilityfilter[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_KEN_GEO1_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_KEN_GEO1_ff]
    dataTbl_KEN_GEO2_ff = MQD_df_KEN_facilityfilter[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_KEN_GEO2_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_KEN_GEO2_ff]
    dataTbl_KEN_GEO3_ff = MQD_df_KEN_facilityfilter[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_KEN_GEO3_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_KEN_GEO3_ff]
    dataTbl_LAO_GEO1_ff = MQD_df_LAO_facilityfilter[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_LAO_GEO1_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_LAO_GEO1_ff]
    dataTbl_LAO_GEO2_ff = MQD_df_LAO_facilityfilter[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_LAO_GEO2_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_LAO_GEO2_ff]
    dataTbl_LAO_GEO3_ff = MQD_df_LAO_facilityfilter[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_LAO_GEO3_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_LAO_GEO3_ff]
    dataTbl_MOZ_GEO1_ff = MQD_df_MOZ_facilityfilter[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_MOZ_GEO1_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_MOZ_GEO1_ff]
    dataTbl_MOZ_GEO2_ff = MQD_df_MOZ_facilityfilter[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_MOZ_GEO2_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_MOZ_GEO2_ff]
    dataTbl_MOZ_GEO3_ff = MQD_df_MOZ_facilityfilter[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_MOZ_GEO3_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_MOZ_GEO3_ff]
    dataTbl_PER_GEO1_ff = MQD_df_PER_facilityfilter[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_PER_GEO1_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_PER_GEO1_ff]
    dataTbl_PER_GEO2_ff = MQD_df_PER_facilityfilter[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_PER_GEO2_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_PER_GEO2_ff]
    dataTbl_PER_GEO3_ff = MQD_df_PER_facilityfilter[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_PER_GEO3_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_PER_GEO3_ff]
    dataTbl_PHI_GEO1_ff = MQD_df_PHI_facilityfilter[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_PHI_GEO1_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_PHI_GEO1_ff]
    dataTbl_PHI_GEO2_ff = MQD_df_PHI_facilityfilter[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_PHI_GEO2_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_PHI_GEO2_ff]
    dataTbl_PHI_GEO3_ff = MQD_df_PHI_facilityfilter[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_PHI_GEO3_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_PHI_GEO3_ff]
    dataTbl_SEN_GEO1_ff = MQD_df_SEN_facilityfilter[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_SEN_GEO1_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_SEN_GEO1_ff]
    dataTbl_SEN_GEO2_ff = MQD_df_SEN_facilityfilter[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_SEN_GEO2_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_SEN_GEO2_ff]
    dataTbl_SEN_GEO3_ff = MQD_df_SEN_facilityfilter[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_SEN_GEO3_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_SEN_GEO3_ff]
    dataTbl_THA_GEO1_ff = MQD_df_THA_facilityfilter[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_THA_GEO1_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_THA_GEO1_ff]
    dataTbl_THA_GEO2_ff = MQD_df_THA_facilityfilter[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_THA_GEO2_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_THA_GEO2_ff]
    dataTbl_THA_GEO3_ff = MQD_df_THA_facilityfilter[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_THA_GEO3_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_THA_GEO3_ff]
    dataTbl_VIE_GEO1_ff = MQD_df_VIE_facilityfilter[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_VIE_GEO1_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_VIE_GEO1_ff]
    dataTbl_VIE_GEO2_ff = MQD_df_VIE_facilityfilter[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_VIE_GEO2_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_VIE_GEO2_ff]
    dataTbl_VIE_GEO3_ff = MQD_df_VIE_facilityfilter[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_VIE_GEO3_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_VIE_GEO3_ff]

    # Therapeutic indication filter
    dataTbl_CAM_GEO1_antibiotic = MQD_df_CAM[MQD_df_CAM['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_CAM_GEO1_antibiotic = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_CAM_GEO1_antibiotic]
    dataTbl_CAM_GEO2_antibiotic = MQD_df_CAM[MQD_df_CAM['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_CAM_GEO2_antibiotic = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_CAM_GEO2_antibiotic]
    dataTbl_CAM_GEO3_antibiotic = MQD_df_CAM[MQD_df_CAM['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_CAM_GEO3_antibiotic = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_CAM_GEO3_antibiotic]
    dataTbl_CAM_GEO1_antimalarial = MQD_df_CAM[MQD_df_CAM['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_CAM_GEO1_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_CAM_GEO1_antimalarial]
    dataTbl_CAM_GEO2_antimalarial = MQD_df_CAM[MQD_df_CAM['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_CAM_GEO2_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_CAM_GEO2_antimalarial]
    dataTbl_CAM_GEO3_antimalarial = MQD_df_CAM[MQD_df_CAM['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_CAM_GEO3_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_CAM_GEO3_antimalarial]
    dataTbl_ETH_GEO1_antibiotic = MQD_df_ETH[MQD_df_ETH['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_ETH_GEO1_antibiotic = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_ETH_GEO1_antibiotic]
    dataTbl_ETH_GEO2_antibiotic = MQD_df_ETH[MQD_df_ETH['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_ETH_GEO2_antibiotic = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_ETH_GEO2_antibiotic]
    dataTbl_ETH_GEO3_antibiotic = MQD_df_ETH[MQD_df_ETH['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_ETH_GEO3_antibiotic = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_ETH_GEO3_antibiotic]
    dataTbl_ETH_GEO1_antimalarial = MQD_df_ETH[MQD_df_ETH['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_ETH_GEO1_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_ETH_GEO1_antimalarial]
    dataTbl_ETH_GEO2_antimalarial = MQD_df_ETH[MQD_df_ETH['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_ETH_GEO2_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_ETH_GEO2_antimalarial]
    dataTbl_ETH_GEO3_antimalarial = MQD_df_ETH[MQD_df_ETH['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_ETH_GEO3_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_ETH_GEO3_antimalarial]
    dataTbl_GHA_GEO1_antimalarial = MQD_df_GHA[MQD_df_GHA['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_GHA_GEO1_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_GHA_GEO1_antimalarial]
    dataTbl_GHA_GEO2_antimalarial = MQD_df_GHA[MQD_df_GHA['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_GHA_GEO2_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_GHA_GEO2_antimalarial]
    dataTbl_GHA_GEO3_antimalarial = MQD_df_GHA[MQD_df_GHA['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_GHA_GEO3_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_GHA_GEO3_antimalarial]
    dataTbl_KEN_GEO1_antimalarial = MQD_df_KEN[MQD_df_KEN['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_KEN_GEO1_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_KEN_GEO1_antimalarial]
    dataTbl_KEN_GEO2_antimalarial = MQD_df_KEN[MQD_df_KEN['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_KEN_GEO2_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_KEN_GEO2_antimalarial]
    dataTbl_KEN_GEO3_antimalarial = MQD_df_KEN[MQD_df_KEN['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_KEN_GEO3_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_KEN_GEO3_antimalarial]
    dataTbl_LAO_GEO1_antibiotic = MQD_df_LAO[MQD_df_LAO['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_LAO_GEO1_antibiotic = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_LAO_GEO1_antibiotic]
    dataTbl_LAO_GEO2_antibiotic = MQD_df_LAO[MQD_df_LAO['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_LAO_GEO2_antibiotic = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_LAO_GEO2_antibiotic]
    dataTbl_LAO_GEO3_antibiotic = MQD_df_LAO[MQD_df_LAO['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_LAO_GEO3_antibiotic = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_LAO_GEO3_antibiotic]
    dataTbl_LAO_GEO1_antimalarial = MQD_df_LAO[MQD_df_LAO['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_LAO_GEO1_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_LAO_GEO1_antimalarial]
    dataTbl_LAO_GEO2_antimalarial = MQD_df_LAO[MQD_df_LAO['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_LAO_GEO2_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_LAO_GEO2_antimalarial]
    dataTbl_LAO_GEO3_antimalarial = MQD_df_LAO[MQD_df_LAO['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_LAO_GEO3_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_LAO_GEO3_antimalarial]
    dataTbl_MOZ_GEO1_antibiotic = MQD_df_MOZ[MQD_df_MOZ['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_MOZ_GEO1_antibiotic = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_MOZ_GEO1_antibiotic]
    dataTbl_MOZ_GEO2_antibiotic = MQD_df_MOZ[MQD_df_MOZ['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_MOZ_GEO2_antibiotic = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_MOZ_GEO2_antibiotic]
    dataTbl_MOZ_GEO3_antibiotic = MQD_df_MOZ[MQD_df_MOZ['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_MOZ_GEO3_antibiotic = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_MOZ_GEO3_antibiotic]
    dataTbl_MOZ_GEO1_antimalarial = MQD_df_MOZ[MQD_df_MOZ['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_MOZ_GEO1_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_MOZ_GEO1_antimalarial]
    dataTbl_MOZ_GEO2_antimalarial = MQD_df_MOZ[MQD_df_MOZ['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_MOZ_GEO2_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_MOZ_GEO2_antimalarial]
    dataTbl_MOZ_GEO3_antimalarial = MQD_df_MOZ[MQD_df_MOZ['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_MOZ_GEO3_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_MOZ_GEO3_antimalarial]
    dataTbl_PER_GEO1_antibiotic = MQD_df_PER[MQD_df_PER['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_PER_GEO1_antibiotic = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_PER_GEO1_antibiotic]
    dataTbl_PER_GEO2_antibiotic = MQD_df_PER[MQD_df_PER['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_PER_GEO2_antibiotic = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_PER_GEO2_antibiotic]
    dataTbl_PER_GEO3_antibiotic = MQD_df_PER[MQD_df_PER['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_PER_GEO3_antibiotic = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_PER_GEO3_antibiotic]
    dataTbl_PHI_GEO1_antituberculosis = MQD_df_PHI[MQD_df_PHI['Indication_GROUPED'].isin(['Antituberculosis'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_PHI_GEO1_antituberculosis = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_PHI_GEO1_antituberculosis]
    dataTbl_PHI_GEO2_antituberculosis = MQD_df_PHI[MQD_df_PHI['Indication_GROUPED'].isin(['Antituberculosis'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_PHI_GEO2_antituberculosis = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_PHI_GEO2_antituberculosis]
    dataTbl_PHI_GEO3_antituberculosis = MQD_df_PHI[MQD_df_PHI['Indication_GROUPED'].isin(['Antituberculosis'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_PHI_GEO3_antituberculosis = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_PHI_GEO3_antituberculosis]
    dataTbl_SEN_GEO1_antimalarial = MQD_df_SEN[MQD_df_SEN['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_SEN_GEO1_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_SEN_GEO1_antimalarial]
    dataTbl_SEN_GEO2_antimalarial = MQD_df_SEN[MQD_df_SEN['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_SEN_GEO2_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_SEN_GEO2_antimalarial]
    dataTbl_SEN_GEO3_antimalarial = MQD_df_SEN[MQD_df_SEN['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_SEN_GEO3_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_SEN_GEO3_antimalarial]
    dataTbl_SEN_GEO1_antiretroviral = MQD_df_SEN[MQD_df_SEN['Indication_GROUPED'].isin(['Antiretroviral'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_SEN_GEO1_antiretroviral = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_SEN_GEO1_antiretroviral]
    dataTbl_SEN_GEO2_antiretroviral = MQD_df_SEN[MQD_df_SEN['Indication_GROUPED'].isin(['Antiretroviral'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_SEN_GEO2_antiretroviral = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_SEN_GEO2_antiretroviral]
    dataTbl_SEN_GEO3_antiretroviral = MQD_df_SEN[MQD_df_SEN['Indication_GROUPED'].isin(['Antiretroviral'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_SEN_GEO3_antiretroviral = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_SEN_GEO3_antiretroviral]
    dataTbl_THA_GEO1_antibiotic = MQD_df_THA[MQD_df_THA['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_THA_GEO1_antibiotic = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_THA_GEO1_antibiotic]
    dataTbl_THA_GEO2_antibiotic = MQD_df_THA[MQD_df_THA['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_THA_GEO2_antibiotic = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_THA_GEO2_antibiotic]
    dataTbl_THA_GEO3_antibiotic = MQD_df_THA[MQD_df_THA['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_THA_GEO3_antibiotic = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_THA_GEO3_antibiotic]
    dataTbl_THA_GEO1_antimalarial = MQD_df_THA[MQD_df_THA['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_THA_GEO1_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_THA_GEO1_antimalarial]
    dataTbl_THA_GEO2_antimalarial = MQD_df_THA[MQD_df_THA['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_THA_GEO2_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_THA_GEO2_antimalarial]
    dataTbl_THA_GEO3_antimalarial = MQD_df_THA[MQD_df_THA['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_THA_GEO3_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_THA_GEO3_antimalarial]
    dataTbl_VIE_GEO1_antibiotic = MQD_df_VIE[MQD_df_VIE['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_VIE_GEO1_antibiotic = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_VIE_GEO1_antibiotic]
    dataTbl_VIE_GEO2_antibiotic = MQD_df_VIE[MQD_df_VIE['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_VIE_GEO2_antibiotic = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_VIE_GEO2_antibiotic]
    dataTbl_VIE_GEO3_antibiotic = MQD_df_VIE[MQD_df_VIE['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_VIE_GEO3_antibiotic = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_VIE_GEO3_antibiotic]
    dataTbl_VIE_GEO1_antimalarial = MQD_df_VIE[MQD_df_VIE['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_VIE_GEO1_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_VIE_GEO1_antimalarial]
    dataTbl_VIE_GEO2_antimalarial = MQD_df_VIE[MQD_df_VIE['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_VIE_GEO2_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_VIE_GEO2_antimalarial]
    dataTbl_VIE_GEO3_antimalarial = MQD_df_VIE[MQD_df_VIE['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_VIE_GEO3_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_VIE_GEO3_antimalarial]

    # Therapeutic indication filter, with outlet-type facility filter
    dataTbl_CAM_GEO1_antibiotic_ff = MQD_df_CAM_facilityfilter[MQD_df_CAM_facilityfilter['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_CAM_GEO1_antibiotic_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_CAM_GEO1_antibiotic_ff]
    dataTbl_CAM_GEO2_antibiotic_ff = MQD_df_CAM_facilityfilter[MQD_df_CAM_facilityfilter['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_CAM_GEO2_antibiotic_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_CAM_GEO2_antibiotic_ff]
    dataTbl_CAM_GEO3_antibiotic_ff = MQD_df_CAM_facilityfilter[MQD_df_CAM_facilityfilter['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_CAM_GEO3_antibiotic_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_CAM_GEO3_antibiotic_ff]
    dataTbl_CAM_GEO1_antimalarial_ff = MQD_df_CAM_facilityfilter[MQD_df_CAM_facilityfilter['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_CAM_GEO1_antimalarial_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_CAM_GEO1_antimalarial_ff]
    dataTbl_CAM_GEO2_antimalarial_ff = MQD_df_CAM_facilityfilter[MQD_df_CAM_facilityfilter['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_CAM_GEO2_antimalarial_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_CAM_GEO2_antimalarial_ff]
    dataTbl_CAM_GEO3_antimalarial_ff = MQD_df_CAM_facilityfilter[MQD_df_CAM_facilityfilter['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_CAM_GEO3_antimalarial_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_CAM_GEO3_antimalarial_ff]
    dataTbl_ETH_GEO1_antibiotic_ff = MQD_df_ETH_facilityfilter[MQD_df_ETH_facilityfilter['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_ETH_GEO1_antibiotic_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_ETH_GEO1_antibiotic_ff]
    dataTbl_ETH_GEO2_antibiotic_ff = MQD_df_ETH_facilityfilter[MQD_df_ETH_facilityfilter['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_ETH_GEO2_antibiotic_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_ETH_GEO2_antibiotic_ff]
    dataTbl_ETH_GEO3_antibiotic_ff = MQD_df_ETH_facilityfilter[MQD_df_ETH_facilityfilter['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_ETH_GEO3_antibiotic_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_ETH_GEO3_antibiotic_ff]
    dataTbl_ETH_GEO1_antimalarial_ff = MQD_df_ETH_facilityfilter[MQD_df_ETH_facilityfilter['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_ETH_GEO1_antimalarial_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_ETH_GEO1_antimalarial_ff]
    dataTbl_ETH_GEO2_antimalarial_ff = MQD_df_ETH_facilityfilter[MQD_df_ETH_facilityfilter['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_ETH_GEO2_antimalarial_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_ETH_GEO2_antimalarial_ff]
    dataTbl_ETH_GEO3_antimalarial_ff = MQD_df_ETH_facilityfilter[MQD_df_ETH_facilityfilter['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_ETH_GEO3_antimalarial_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_ETH_GEO3_antimalarial_ff]
    dataTbl_GHA_GEO1_antimalarial_ff = MQD_df_GHA_facilityfilter[MQD_df_GHA_facilityfilter['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_GHA_GEO1_antimalarial_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_GHA_GEO1_antimalarial_ff]
    dataTbl_GHA_GEO2_antimalarial_ff = MQD_df_GHA_facilityfilter[MQD_df_GHA_facilityfilter['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_GHA_GEO2_antimalarial_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_GHA_GEO2_antimalarial_ff]
    dataTbl_GHA_GEO3_antimalarial_ff = MQD_df_GHA_facilityfilter[MQD_df_GHA_facilityfilter['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_GHA_GEO3_antimalarial_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_GHA_GEO3_antimalarial_ff]
    dataTbl_KEN_GEO1_antimalarial_ff = MQD_df_KEN_facilityfilter[MQD_df_KEN_facilityfilter['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_KEN_GEO1_antimalarial_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_KEN_GEO1_antimalarial_ff]
    dataTbl_KEN_GEO2_antimalarial_ff = MQD_df_KEN_facilityfilter[MQD_df_KEN_facilityfilter['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_KEN_GEO2_antimalarial_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_KEN_GEO2_antimalarial_ff]
    dataTbl_KEN_GEO3_antimalarial_ff = MQD_df_KEN_facilityfilter[MQD_df_KEN_facilityfilter['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_KEN_GEO3_antimalarial_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_KEN_GEO3_antimalarial_ff]
    dataTbl_LAO_GEO1_antibiotic_ff = MQD_df_LAO_facilityfilter[MQD_df_LAO_facilityfilter['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_LAO_GEO1_antibiotic_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_LAO_GEO1_antibiotic_ff]
    dataTbl_LAO_GEO2_antibiotic_ff = MQD_df_LAO_facilityfilter[MQD_df_LAO_facilityfilter['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_LAO_GEO2_antibiotic_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_LAO_GEO2_antibiotic_ff]
    dataTbl_LAO_GEO3_antibiotic_ff = MQD_df_LAO_facilityfilter[MQD_df_LAO_facilityfilter['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_LAO_GEO3_antibiotic_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_LAO_GEO3_antibiotic_ff]
    dataTbl_LAO_GEO1_antimalarial_ff = MQD_df_LAO_facilityfilter[MQD_df_LAO_facilityfilter['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_LAO_GEO1_antimalarial_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_LAO_GEO1_antimalarial_ff]
    dataTbl_LAO_GEO2_antimalarial_ff = MQD_df_LAO_facilityfilter[MQD_df_LAO_facilityfilter['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_LAO_GEO2_antimalarial_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_LAO_GEO2_antimalarial_ff]
    dataTbl_LAO_GEO3_antimalarial_ff = MQD_df_LAO_facilityfilter[MQD_df_LAO_facilityfilter['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_LAO_GEO3_antimalarial_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_LAO_GEO3_antimalarial_ff]
    dataTbl_MOZ_GEO1_antibiotic_ff = MQD_df_MOZ_facilityfilter[MQD_df_MOZ_facilityfilter['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_MOZ_GEO1_antibiotic_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_MOZ_GEO1_antibiotic_ff]
    dataTbl_MOZ_GEO2_antibiotic_ff = MQD_df_MOZ_facilityfilter[MQD_df_MOZ_facilityfilter['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_MOZ_GEO2_antibiotic_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_MOZ_GEO2_antibiotic_ff]
    dataTbl_MOZ_GEO3_antibiotic_ff = MQD_df_MOZ_facilityfilter[MQD_df_MOZ_facilityfilter['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_MOZ_GEO3_antibiotic_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_MOZ_GEO3_antibiotic_ff]
    dataTbl_MOZ_GEO1_antimalarial_ff = MQD_df_MOZ_facilityfilter[MQD_df_MOZ_facilityfilter['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_MOZ_GEO1_antimalarial_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_MOZ_GEO1_antimalarial_ff]
    dataTbl_MOZ_GEO2_antimalarial_ff = MQD_df_MOZ_facilityfilter[MQD_df_MOZ_facilityfilter['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_MOZ_GEO2_antimalarial_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_MOZ_GEO2_antimalarial_ff]
    dataTbl_MOZ_GEO3_antimalarial_ff = MQD_df_MOZ_facilityfilter[MQD_df_MOZ_facilityfilter['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_MOZ_GEO3_antimalarial_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_MOZ_GEO3_antimalarial_ff]
    dataTbl_PER_GEO1_antibiotic_ff = MQD_df_PER_facilityfilter[MQD_df_PER_facilityfilter['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_PER_GEO1_antibiotic_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_PER_GEO1_antibiotic_ff]
    dataTbl_PER_GEO2_antibiotic_ff = MQD_df_PER_facilityfilter[MQD_df_PER_facilityfilter['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_PER_GEO2_antibiotic_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_PER_GEO2_antibiotic_ff]
    dataTbl_PER_GEO3_antibiotic_ff = MQD_df_PER_facilityfilter[MQD_df_PER_facilityfilter['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_PER_GEO3_antibiotic_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_PER_GEO3_antibiotic_ff]
    dataTbl_PHI_GEO1_antituberculosis_ff = MQD_df_PHI_facilityfilter[MQD_df_PHI_facilityfilter['Indication_GROUPED'].isin(['Antituberculosis'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_PHI_GEO1_antituberculosis_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                         dataTbl_PHI_GEO1_antituberculosis_ff]
    dataTbl_PHI_GEO2_antituberculosis_ff = MQD_df_PHI_facilityfilter[MQD_df_PHI_facilityfilter['Indication_GROUPED'].isin(['Antituberculosis'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_PHI_GEO2_antituberculosis_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                         dataTbl_PHI_GEO2_antituberculosis_ff]
    dataTbl_PHI_GEO3_antituberculosis_ff = MQD_df_PHI_facilityfilter[MQD_df_PHI_facilityfilter['Indication_GROUPED'].isin(['Antituberculosis'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_PHI_GEO3_antituberculosis_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                         dataTbl_PHI_GEO3_antituberculosis_ff]
    dataTbl_SEN_GEO1_antimalarial_ff = MQD_df_SEN_facilityfilter[MQD_df_SEN_facilityfilter['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_SEN_GEO1_antimalarial_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_SEN_GEO1_antimalarial_ff]
    dataTbl_SEN_GEO2_antimalarial_ff = MQD_df_SEN_facilityfilter[MQD_df_SEN_facilityfilter['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_SEN_GEO2_antimalarial_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_SEN_GEO2_antimalarial_ff]
    dataTbl_SEN_GEO3_antimalarial_ff = MQD_df_SEN_facilityfilter[MQD_df_SEN_facilityfilter['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_SEN_GEO3_antimalarial_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_SEN_GEO3_antimalarial_ff]
    dataTbl_SEN_GEO1_antiretroviral_ff = MQD_df_SEN_facilityfilter[MQD_df_SEN_facilityfilter['Indication_GROUPED'].isin(['Antiretroviral'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_SEN_GEO1_antiretroviral_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                       dataTbl_SEN_GEO1_antiretroviral_ff]
    dataTbl_SEN_GEO2_antiretroviral_ff = MQD_df_SEN_facilityfilter[MQD_df_SEN_facilityfilter['Indication_GROUPED'].isin(['Antiretroviral'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_SEN_GEO2_antiretroviral_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                       dataTbl_SEN_GEO2_antiretroviral_ff]
    dataTbl_SEN_GEO3_antiretroviral_ff = MQD_df_SEN_facilityfilter[MQD_df_SEN_facilityfilter['Indication_GROUPED'].isin(['Antiretroviral'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_SEN_GEO3_antiretroviral_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                       dataTbl_SEN_GEO3_antiretroviral_ff]
    dataTbl_THA_GEO1_antibiotic_ff = MQD_df_THA_facilityfilter[MQD_df_THA_facilityfilter['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_THA_GEO1_antibiotic_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_THA_GEO1_antibiotic_ff]
    dataTbl_THA_GEO2_antibiotic_ff = MQD_df_THA_facilityfilter[MQD_df_THA_facilityfilter['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_THA_GEO2_antibiotic_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_THA_GEO2_antibiotic_ff]
    dataTbl_THA_GEO3_antibiotic_ff = MQD_df_THA_facilityfilter[MQD_df_THA_facilityfilter['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_THA_GEO3_antibiotic_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_THA_GEO3_antibiotic_ff]
    dataTbl_THA_GEO1_antimalarial_ff = MQD_df_THA_facilityfilter[MQD_df_THA_facilityfilter['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_THA_GEO1_antimalarial_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_THA_GEO1_antimalarial_ff]
    dataTbl_THA_GEO2_antimalarial_ff = MQD_df_THA_facilityfilter[MQD_df_THA_facilityfilter['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_THA_GEO2_antimalarial_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_THA_GEO2_antimalarial_ff]
    dataTbl_THA_GEO3_antimalarial_ff = MQD_df_THA_facilityfilter[MQD_df_THA_facilityfilter['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_THA_GEO3_antimalarial_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_THA_GEO3_antimalarial_ff]
    dataTbl_VIE_GEO1_antibiotic_ff = MQD_df_VIE_facilityfilter[MQD_df_VIE_facilityfilter['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_VIE_GEO1_antibiotic_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_VIE_GEO1_antibiotic_ff]
    dataTbl_VIE_GEO2_antibiotic_ff = MQD_df_VIE_facilityfilter[MQD_df_VIE_facilityfilter['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_VIE_GEO2_antibiotic_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_VIE_GEO2_antibiotic_ff]
    dataTbl_VIE_GEO3_antibiotic_ff = MQD_df_VIE_facilityfilter[MQD_df_VIE_facilityfilter['Indication_GROUPED'].isin(['Antibiotic'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_VIE_GEO3_antibiotic_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                   dataTbl_VIE_GEO3_antibiotic_ff]
    dataTbl_VIE_GEO1_antimalarial_ff = MQD_df_VIE_facilityfilter[MQD_df_VIE_facilityfilter['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_VIE_GEO1_antimalarial_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_VIE_GEO1_antimalarial_ff]
    dataTbl_VIE_GEO2_antimalarial_ff = MQD_df_VIE_facilityfilter[MQD_df_VIE_facilityfilter['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_VIE_GEO2_antimalarial_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_VIE_GEO2_antimalarial_ff]
    dataTbl_VIE_GEO3_antimalarial_ff = MQD_df_VIE_facilityfilter[MQD_df_VIE_facilityfilter['Indication_GROUPED'].isin(['Antimalarial'])][
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    dataTbl_VIE_GEO3_antimalarial_ff = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in
                                     dataTbl_VIE_GEO3_antimalarial_ff]





    # Put the databases and lists into a dictionary
    outputDict = {}
    outputDict.update({# all raw pandas dataframes
       'df_ALL':MQD_df, 'df_CAM':MQD_df_CAM, 'df_ETH':MQD_df_ETH, 'df_GHA':MQD_df_GHA,
       'df_KEN':MQD_df_KEN, 'df_LAO':MQD_df_LAO, 'df_MOZ':MQD_df_MOZ, 'df_PER':MQD_df_PER,
       'df_PHI':MQD_df_PHI, 'df_SEN':MQD_df_SEN, 'df_THA':MQD_df_THA, 'df_VIE':MQD_df_VIE,
       'df_CAM_ff':MQD_df_CAM_facilityfilter, 'df_ETH_ff':MQD_df_ETH_facilityfilter,
       'df_GHA_ff':MQD_df_GHA_facilityfilter, 'df_KEN_ff':MQD_df_KEN_facilityfilter,
       'df_LAO_ff':MQD_df_LAO_facilityfilter, 'df_MOZ_ff':MQD_df_MOZ_facilityfilter,
       'df_PER_ff':MQD_df_PER_facilityfilter, 'df_PHI_ff':MQD_df_PHI_facilityfilter,
       'df_SEN_ff':MQD_df_SEN_facilityfilter, 'df_THA_ff':MQD_df_THA_facilityfilter,
       'df_VIE_ff':MQD_df_VIE_facilityfilter,
       # raw data tables for use with Logistigate
       'tbl_CAM_G1': dataTbl_CAM_GEO1, 'tbl_CAM_G2': dataTbl_CAM_GEO2, 'tbl_CAM_G3': dataTbl_CAM_GEO3,
       'tbl_ETH_G1': dataTbl_ETH_GEO1, 'tbl_ETH_G2': dataTbl_ETH_GEO2, 'tbl_ETH_G3': dataTbl_ETH_GEO3,
       'tbl_GHA_G1': dataTbl_GHA_GEO1, 'tbl_GHA_G2': dataTbl_GHA_GEO2, 'tbl_GHA_G3': dataTbl_GHA_GEO3,
       'tbl_KEN_G1': dataTbl_KEN_GEO1, 'tbl_KEN_G2': dataTbl_KEN_GEO2, 'tbl_KEN_G3': dataTbl_KEN_GEO3,
       'tbl_LAO_G1': dataTbl_LAO_GEO1, 'tbl_LAO_G2': dataTbl_LAO_GEO2, 'tbl_LAO_G3': dataTbl_LAO_GEO3,
       'tbl_MOZ_G1': dataTbl_MOZ_GEO1, 'tbl_MOZ_G2': dataTbl_MOZ_GEO2, 'tbl_MOZ_G3': dataTbl_MOZ_GEO3,
       'tbl_PER_G1': dataTbl_PER_GEO1, 'tbl_PER_G2': dataTbl_PER_GEO2, 'tbl_PER_G3': dataTbl_PER_GEO3,
       'tbl_PHI_G1': dataTbl_PHI_GEO1, 'tbl_PHI_G2': dataTbl_PHI_GEO2, 'tbl_PHI_G3': dataTbl_PHI_GEO3,
       'tbl_SEN_G1': dataTbl_SEN_GEO1, 'tbl_SEN_G2': dataTbl_SEN_GEO2, 'tbl_SEN_G3': dataTbl_SEN_GEO3,
       'tbl_THA_G1': dataTbl_THA_GEO1, 'tbl_THA_G2': dataTbl_THA_GEO2, 'tbl_THA_G3': dataTbl_THA_GEO3,
       'tbl_VIE_G1': dataTbl_VIE_GEO1, 'tbl_VIE_G2': dataTbl_VIE_GEO2, 'tbl_VIE_G3': dataTbl_VIE_GEO3,
       # data tables with outlet-type facility filter
       'tbl_CAM_G1_ff': dataTbl_CAM_GEO1_ff, 'tbl_CAM_G2_ff': dataTbl_CAM_GEO2_ff, 'tbl_CAM_G3_ff': dataTbl_CAM_GEO3_ff,
       'tbl_ETH_G1_ff': dataTbl_ETH_GEO1_ff, 'tbl_ETH_G2_ff': dataTbl_ETH_GEO2_ff, 'tbl_ETH_G3_ff': dataTbl_ETH_GEO3_ff,
       'tbl_GHA_G1_ff': dataTbl_GHA_GEO1_ff, 'tbl_GHA_G2_ff': dataTbl_GHA_GEO2_ff, 'tbl_GHA_G3_ff': dataTbl_GHA_GEO3_ff,
       'tbl_KEN_G1_ff': dataTbl_KEN_GEO1_ff, 'tbl_KEN_G2_ff': dataTbl_KEN_GEO2_ff, 'tbl_KEN_G3_ff': dataTbl_KEN_GEO3_ff,
       'tbl_LAO_G1_ff': dataTbl_LAO_GEO1_ff, 'tbl_LAO_G2_ff': dataTbl_LAO_GEO2_ff, 'tbl_LAO_G3_ff': dataTbl_LAO_GEO3_ff,
       'tbl_MOZ_G1_ff': dataTbl_MOZ_GEO1_ff, 'tbl_MOZ_G2_ff': dataTbl_MOZ_GEO2_ff, 'tbl_MOZ_G3_ff': dataTbl_MOZ_GEO3_ff,
       'tbl_PER_G1_ff': dataTbl_PER_GEO1_ff, 'tbl_PER_G2_ff': dataTbl_PER_GEO2_ff, 'tbl_PER_G3_ff': dataTbl_PER_GEO3_ff,
       'tbl_PHI_G1_ff': dataTbl_PHI_GEO1_ff, 'tbl_PHI_G2_ff': dataTbl_PHI_GEO2_ff, 'tbl_PHI_G3_ff': dataTbl_PHI_GEO3_ff,
       'tbl_SEN_G1_ff': dataTbl_SEN_GEO1_ff, 'tbl_SEN_G2_ff': dataTbl_SEN_GEO2_ff, 'tbl_SEN_G3_ff': dataTbl_SEN_GEO3_ff,
       'tbl_THA_G1_ff': dataTbl_THA_GEO1_ff, 'tbl_THA_G2_ff': dataTbl_THA_GEO2_ff, 'tbl_THA_G3_ff': dataTbl_THA_GEO3_ff,
       'tbl_VIE_G1_ff': dataTbl_VIE_GEO1_ff, 'tbl_VIE_G2_ff': dataTbl_VIE_GEO2_ff, 'tbl_VIE_G3_ff': dataTbl_VIE_GEO3_ff,
       # data tables for different therapeutic indications
       'tbl_CAM_G1_antibiotic': dataTbl_CAM_GEO1_antibiotic, 'tbl_CAM_G2_antibiotic': dataTbl_CAM_GEO2_antibiotic, 'tbl_CAM_G3_antibiotic': dataTbl_CAM_GEO3_antibiotic,
       'tbl_CAM_G1_antimalarial': dataTbl_CAM_GEO1_antimalarial, 'tbl_CAM_G2_antimalarial': dataTbl_CAM_GEO2_antimalarial, 'tbl_CAM_G3_antimalarial': dataTbl_CAM_GEO3_antimalarial,
       'tbl_ETH_G1_antibiotic': dataTbl_ETH_GEO1_antibiotic, 'tbl_ETH_G2_antibiotic': dataTbl_ETH_GEO2_antibiotic, 'tbl_ETH_G3_antibiotic': dataTbl_ETH_GEO3_antibiotic,
       'tbl_ETH_G1_antimalarial': dataTbl_ETH_GEO1_antimalarial, 'tbl_ETH_G2_antimalarial': dataTbl_ETH_GEO2_antimalarial, 'tbl_ETH_G3_antimalarial': dataTbl_ETH_GEO3_antimalarial,
       'tbl_GHA_G1_antimalarial': dataTbl_GHA_GEO1_antimalarial, 'tbl_GHA_G2_antimalarial': dataTbl_GHA_GEO2_antimalarial, 'tbl_GHA_G3_antimalarial': dataTbl_GHA_GEO3_antimalarial,
       'tbl_KEN_G1_antimalarial': dataTbl_KEN_GEO1_antimalarial, 'tbl_KEN_G2_antimalarial': dataTbl_KEN_GEO2_antimalarial, 'tbl_KEN_G3_antimalarial': dataTbl_KEN_GEO3_antimalarial,
       'tbl_LAO_G1_antibiotic': dataTbl_LAO_GEO1_antibiotic, 'tbl_LAO_G2_antibiotic': dataTbl_LAO_GEO2_antibiotic, 'tbl_LAO_G3_antibiotic': dataTbl_LAO_GEO3_antibiotic,
       'tbl_LAO_G1_antimalarial': dataTbl_LAO_GEO1_antimalarial, 'tbl_LAO_G2_antimalarial': dataTbl_LAO_GEO2_antimalarial, 'tbl_LAO_G3_antimalarial': dataTbl_LAO_GEO3_antimalarial,
       'tbl_MOZ_G1_antibiotic': dataTbl_MOZ_GEO1_antibiotic, 'tbl_MOZ_G2_antibiotic': dataTbl_MOZ_GEO2_antibiotic, 'tbl_MOZ_G3_antibiotic': dataTbl_MOZ_GEO3_antibiotic,
       'tbl_MOZ_G1_antimalarial': dataTbl_MOZ_GEO1_antimalarial, 'tbl_MOZ_G2_antimalarial': dataTbl_MOZ_GEO2_antimalarial, 'tbl_MOZ_G3_antimalarial': dataTbl_MOZ_GEO3_antimalarial,
       'tbl_PER_G1_antibiotic': dataTbl_PER_GEO1_antibiotic, 'tbl_PER_G2_antibiotic': dataTbl_PER_GEO2_antibiotic, 'tbl_PER_G3_antibiotic': dataTbl_PER_GEO3_antibiotic,
       'tbl_PHI_G1_antituberculosis': dataTbl_PHI_GEO1_antituberculosis, 'tbl_PHI_G2_antituberculosis': dataTbl_PHI_GEO2_antituberculosis, 'tbl_PHI_G3_antituberculosis': dataTbl_PHI_GEO3_antituberculosis,
       'tbl_SEN_G1_antimalarial': dataTbl_SEN_GEO1_antimalarial, 'tbl_SEN_G2_antimalarial': dataTbl_SEN_GEO2_antimalarial, 'tbl_SEN_G3_antimalarial': dataTbl_SEN_GEO3_antimalarial,
       'tbl_SEN_G1_antiretroviral': dataTbl_SEN_GEO1_antiretroviral, 'tbl_SEN_G2_antiretroviral': dataTbl_SEN_GEO2_antiretroviral, 'tbl_SEN_G3_antiretroviral': dataTbl_SEN_GEO3_antiretroviral,
       'tbl_THA_G1_antibiotic': dataTbl_THA_GEO1_antibiotic, 'tbl_THA_G2_antibiotic': dataTbl_THA_GEO2_antibiotic, 'tbl_THA_G3_antibiotic': dataTbl_THA_GEO3_antibiotic,
       'tbl_THA_G1_antimalarial': dataTbl_THA_GEO1_antimalarial, 'tbl_THA_G2_antimalarial': dataTbl_THA_GEO2_antimalarial, 'tbl_THA_G3_antimalarial': dataTbl_THA_GEO3_antimalarial,
       'tbl_VIE_G1_antibiotic': dataTbl_VIE_GEO1_antibiotic, 'tbl_VIE_G2_antibiotic': dataTbl_VIE_GEO2_antibiotic, 'tbl_VIE_G3_antibiotic': dataTbl_VIE_GEO3_antibiotic,
       'tbl_VIE_G1_antimalarial': dataTbl_VIE_GEO1_antimalarial, 'tbl_VIE_G2_antimalarial': dataTbl_VIE_GEO2_antimalarial, 'tbl_VIE_G3_antimalarial': dataTbl_VIE_GEO3_antimalarial,
       # data tables for therapeutic indications, with outlet-type facility filter
      'tbl_CAM_G1_antibiotic_ff': dataTbl_CAM_GEO1_antibiotic_ff, 'tbl_CAM_G2_antibiotic_ff': dataTbl_CAM_GEO2_antibiotic_ff, 'tbl_CAM_G3_antibiotic_ff': dataTbl_CAM_GEO3_antibiotic_ff,
      'tbl_CAM_G1_antimalarial_ff': dataTbl_CAM_GEO1_antimalarial_ff, 'tbl_CAM_G2_antimalarial_ff': dataTbl_CAM_GEO2_antimalarial_ff, 'tbl_CAM_G3_antimalarial_ff': dataTbl_CAM_GEO3_antimalarial_ff,
      'tbl_ETH_G1_antibiotic_ff': dataTbl_ETH_GEO1_antibiotic_ff, 'tbl_ETH_G2_antibiotic_ff': dataTbl_ETH_GEO2_antibiotic_ff, 'tbl_ETH_G3_antibiotic_ff': dataTbl_ETH_GEO3_antibiotic_ff,
      'tbl_ETH_G1_antimalarial_ff': dataTbl_ETH_GEO1_antimalarial_ff, 'tbl_ETH_G2_antimalarial_ff': dataTbl_ETH_GEO2_antimalarial_ff, 'tbl_ETH_G3_antimalarial_ff': dataTbl_ETH_GEO3_antimalarial_ff,
      'tbl_GHA_G1_antimalarial_ff': dataTbl_GHA_GEO1_antimalarial_ff, 'tbl_GHA_G2_antimalarial_ff': dataTbl_GHA_GEO2_antimalarial_ff, 'tbl_GHA_G3_antimalarial_ff': dataTbl_GHA_GEO3_antimalarial_ff,
      'tbl_KEN_G1_antimalarial_ff': dataTbl_KEN_GEO1_antimalarial_ff, 'tbl_KEN_G2_antimalarial_ff': dataTbl_KEN_GEO2_antimalarial_ff, 'tbl_KEN_G3_antimalarial_ff': dataTbl_KEN_GEO3_antimalarial_ff,
      'tbl_LAO_G1_antibiotic_ff': dataTbl_LAO_GEO1_antibiotic_ff, 'tbl_LAO_G2_antibiotic_ff': dataTbl_LAO_GEO2_antibiotic_ff, 'tbl_LAO_G3_antibiotic_ff': dataTbl_LAO_GEO3_antibiotic_ff,
      'tbl_LAO_G1_antimalarial_ff': dataTbl_LAO_GEO1_antimalarial_ff, 'tbl_LAO_G2_antimalarial_ff': dataTbl_LAO_GEO2_antimalarial_ff, 'tbl_LAO_G3_antimalarial_ff': dataTbl_LAO_GEO3_antimalarial_ff,
      'tbl_MOZ_G1_antibiotic_ff': dataTbl_MOZ_GEO1_antibiotic_ff, 'tbl_MOZ_G2_antibiotic_ff': dataTbl_MOZ_GEO2_antibiotic_ff, 'tbl_MOZ_G3_antibiotic_ff': dataTbl_MOZ_GEO3_antibiotic_ff,
      'tbl_MOZ_G1_antimalarial_ff': dataTbl_MOZ_GEO1_antimalarial_ff, 'tbl_MOZ_G2_antimalarial_ff': dataTbl_MOZ_GEO2_antimalarial_ff, 'tbl_MOZ_G3_antimalarial_ff': dataTbl_MOZ_GEO3_antimalarial_ff,
      'tbl_PER_G1_antibiotic_ff': dataTbl_PER_GEO1_antibiotic_ff, 'tbl_PER_G2_antibiotic_ff': dataTbl_PER_GEO2_antibiotic_ff, 'tbl_PER_G3_antibiotic_ff': dataTbl_PER_GEO3_antibiotic_ff,
      'tbl_PHI_G1_antituberculosis_ff': dataTbl_PHI_GEO1_antituberculosis_ff, 'tbl_PHI_G2_antituberculosis_ff': dataTbl_PHI_GEO2_antituberculosis_ff, 'tbl_PHI_G3_antituberculosis_ff': dataTbl_PHI_GEO3_antituberculosis_ff,
      'tbl_SEN_G1_antimalarial_ff': dataTbl_SEN_GEO1_antimalarial_ff, 'tbl_SEN_G2_antimalarial_ff': dataTbl_SEN_GEO2_antimalarial_ff, 'tbl_SEN_G3_antimalarial_ff': dataTbl_SEN_GEO3_antimalarial_ff,
      'tbl_SEN_G1_antiretroviral_ff': dataTbl_SEN_GEO1_antiretroviral_ff, 'tbl_SEN_G2_antiretroviral_ff': dataTbl_SEN_GEO2_antiretroviral_ff, 'tbl_SEN_G3_antiretroviral_ff': dataTbl_SEN_GEO3_antiretroviral_ff,
      'tbl_THA_G1_antibiotic_ff': dataTbl_THA_GEO1_antibiotic_ff, 'tbl_THA_G2_antibiotic_ff': dataTbl_THA_GEO2_antibiotic_ff, 'tbl_THA_G3_antibiotic_ff': dataTbl_THA_GEO3_antibiotic_ff,
      'tbl_THA_G1_antimalarial_ff': dataTbl_THA_GEO1_antimalarial_ff, 'tbl_THA_G2_antimalarial_ff': dataTbl_THA_GEO2_antimalarial_ff, 'tbl_THA_G3_antimalarial_ff': dataTbl_THA_GEO3_antimalarial_ff,
      'tbl_VIE_G1_antibiotic_ff': dataTbl_VIE_GEO1_antibiotic_ff, 'tbl_VIE_G2_antibiotic_ff': dataTbl_VIE_GEO2_antibiotic_ff, 'tbl_VIE_G3_antibiotic_ff': dataTbl_VIE_GEO3_antibiotic_ff,
      'tbl_VIE_G1_antimalarial_ff': dataTbl_VIE_GEO1_antimalarial_ff, 'tbl_VIE_G2_antimalarial_ff': dataTbl_VIE_GEO2_antimalarial_ff, 'tbl_VIE_G3_antimalarial_ff': dataTbl_VIE_GEO3_antimalarial_ff
    })

    return outputDict

'''
FOR STORING THE OUTPUT DICTIONARY AS A SAVED OBJECT SO WE DON'T HAVE TO RUN THE PROCESSOR EVERY TIME; ABOUT 28 MB

import pickle
import os
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
filesPath = os.path.join(SCRIPT_DIR, 'MQDfiles')
outputFileName = os.path.join(filesPath, 'pickleOutput')
pickle.dump(outputDict, open(outputFileName,'wb'))
'''

def SenegalDataScript():
    '''
    This script runs everything needed for the paper 'Inferring sources of substandard and falsified products in
    pharmaceutical supply chains'.
    '''
    import os
    import pickle
    #import pandas as pd

    SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
    filesPath = os.path.join(SCRIPT_DIR, 'MQDfiles')
    outputFileName = os.path.join(filesPath, 'pickleOutput')

    openFile = open(outputFileName, 'rb')  # Read the file
    dataDict = pickle.load(openFile)

    numPostSamps = 1000
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    meanSFPrate = dataDict['df_ALL'][dataDict['df_ALL']['Final_Test_Conclusion'] == 'Fail']['Sample_ID'].count() / \
                  dataDict['df_ALL']['Sample_ID'].count()
    priorMean = spsp.logit(meanSFPrate)  # Mean SFP rate of the MQDB data
    priorVar = 1.416197468

    SEN_df = dataDict['df_SEN']
    # 7 unique Province_Name_GROUPED; 23 unique Facility_Location_GROUPED; 66 unique Facility_Name_GROUPED
    # Remove 'Missing' and 'Unknown' labels
    SEN_df_2010 = SEN_df[(SEN_df['Date_Received'] == '7/12/2010') & (SEN_df['Manufacturer_GROUPED'] != 'Unknown') & (
                SEN_df['Facility_Location_GROUPED'] != 'Missing')].copy()
    tbl_SEN_G1_2010 = SEN_df_2010[['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    tbl_SEN_G1_2010 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in tbl_SEN_G1_2010]
    tbl_SEN_G2_2010 = SEN_df_2010[['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    tbl_SEN_G2_2010 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in tbl_SEN_G2_2010]
    tbl_SEN_G3_2010 = SEN_df_2010[['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    tbl_SEN_G3_2010 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in tbl_SEN_G3_2010]

    # Print some overall summaries of the data
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

    # DEIDENTIFICATION

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
        newName = 'Mnfr. ' + str(i + 1)
        for ind, item in enumerate(tbl_SEN_G1_2010):
            if item[1] == currName:
                tbl_SEN_G1_2010[ind][1] = newName
        for ind, item in enumerate(tbl_SEN_G2_2010):
            if item[1] == currName:
                tbl_SEN_G2_2010[ind][1] = newName
        for ind, item in enumerate(tbl_SEN_G3_2010):
            if item[1] == currName:
                tbl_SEN_G3_2010[ind][1] = newName
    # Replace Province
    orig_PROV_lst = ['Dakar', 'Kaffrine', 'Kaolack', 'Kedougou', 'Kolda', 'Matam', 'Saint Louis']
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

    # Replace Facility Name
    orig_NAME_lst = ['CHR', 'CTA-Fann', 'Centre Hospitalier Regional de Thies', 'Centre de Sante Diourbel',
                     'Centre de Sante Mbacke', 'Centre de Sante Ousmane Ngom', 'Centre de Sante Roi Baudouin',
                     'Centre de Sante de Dioum', 'Centre de Sante de Kanel', 'Centre de Sante de Kedougou',
                     'Centre de Sante de Kolda', 'Centre de Sante de Koumpantoum', 'Centre de Sante de Matam',
                     'Centre de Sante de Richard Toll', 'Centre de Sante de Tambacounda',
                     'Centre de Sante de Velingara',
                     'Centre de Traitement de la Tuberculose de Touba', 'District Sanitaire Touba',
                     'District Sanitaire de Mbour',
                     'District Sanitaire de Rufisque', 'District Sanitaire de Tivaoune', 'District Sud',
                     'Hopital Diourbel',
                     'Hopital Regional de Saint Louis', 'Hopital Regionale de Ouro-Sogui', 'Hopital Touba',
                     'Hopital de Dioum',
                     'Hopitale Regionale de Koda', 'Hopitale Regionale de Tambacounda', 'PNA', 'PRA', 'PRA Diourbel',
                     'PRA Thies',
                     'Pharmacie', 'Pharmacie Awa Barry', 'Pharmacie Babacar Sy', 'Pharmacie Boubakh',
                     'Pharmacie Ceikh Ousmane Mbacke', 'Pharmacie Centrale Dr A.C.', "Pharmacie Chateau d'Eau",
                     'Pharmacie Cheikh Tidiane', 'Pharmacie El Hadj Omar Tall', 'Pharmacie Fouladou',
                     'Pharmacie Kancisse',
                     'Pharmacie Keneya', 'Pharmacie Kolda', 'Pharmacie Koldoise',
                     'Pharmacie Mame Diarra Bousso Dr Y.D.D.',
                     'Pharmacie Mame Fatou Diop Yoro', 'Pharmacie Mame Ibrahima Ndour Dr A.N.', 'Pharmacie Mame Madia',
                     'Pharmacie Ndamatou Dr O.N.', 'Pharmacie Oriantale', 'Pharmacie Oumou Khairy Ndiaye',
                     'Pharmacie Ousmane',
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
        newName = 'Facility ' + str(i + 1)
        for ind, item in enumerate(tbl_SEN_G3_2010):
            if item[0] == currName:
                tbl_SEN_G3_2010[ind][0] = newName

    # RUN 1: s=1.0, r=1.0, prior is laplace(-2.5,3.5)
    priorMean = -2.5
    priorVar = 3.5

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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

    # RUN 1b: s=1.0, r=1.0, prior is normal(-2.5,3.5)
    priorMean = -2.5
    priorVar = 3.5

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
    SNinds = lgDict['importerNames'].index('Mnfr. 5')
    print('Manufacturer 5: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    SNinds = lgDict['importerNames'].index('Mnfr. 8')
    print('Manufacturer 8: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    SNinds = lgDict['importerNames'].index('Mnfr. 10')
    print('Manufacturer 10: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    TNinds = lgDict['outletNames'].index('District 7')
    print('District 7: (' + str(
        np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.05))[
                            :5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.95))[:5] + ')')
    TNinds = lgDict['outletNames'].index('District 8')
    print('District 8: (' + str(
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

    # RUN 1c: s=0.8, r=1.0, prior is normal(-2.5,3.5)
    priorMean = -2.5
    priorVar = 3.5
    s, r = 0.8, 1.0
    lgDict = util.testresultsfiletotable(tbl_SEN_G2_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
    SNinds = lgDict['importerNames'].index('Mnfr. 5')
    print('Manufacturer 5: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    SNinds = lgDict['importerNames'].index('Mnfr. 8')
    print('Manufacturer 8: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    SNinds = lgDict['importerNames'].index('Mnfr. 10')
    print('Manufacturer 10: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    TNinds = lgDict['outletNames'].index('District 7')
    print('District 7: (' + str(
        np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.05))[
                            :5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.95))[:5] + ')')
    TNinds = lgDict['outletNames'].index('District 8')
    print('District 8: (' + str(
        np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.05))[
                            :5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.95))[:5] + ')')

    s, r = 1.0, 0.95
    lgDict = util.testresultsfiletotable(tbl_SEN_G2_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
    SNinds = lgDict['importerNames'].index('Mnfr. 5')
    print('Manufacturer 5: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    SNinds = lgDict['importerNames'].index('Mnfr. 8')
    print('Manufacturer 8: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    SNinds = lgDict['importerNames'].index('Mnfr. 10')
    print('Manufacturer 10: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    TNinds = lgDict['outletNames'].index('District 7')
    print('District 7: (' + str(
        np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.05))[
                            :5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.95))[:5] + ')')
    TNinds = lgDict['outletNames'].index('District 8')
    print('District 8: (' + str(
        np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.05))[
                            :5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.95))[:5] + ')')

    # RUN 2: s=1.0, r=1.0, prior is laplace(-2.5,1.5)
    priorMean = -2.5
    priorVar = 1.5

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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

    ##### USE THIS RUN TO GENERATE PLOTS #####
    priorMean = -2.5
    priorVar = 3.5
    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    numSN, numTN = lgDict['importerNum'], lgDict['outletNum']

    lowerQuant, upperQuant = 0.05, 0.95
    priorSamps = lgDict['prior'].expitrand(5000)
    # priorLower, priorUpper = np.quantile(priorSamps, lowerQuant), np.quantile(priorSamps, upperQuant)
    priorLower = spsp.expit(sps.laplace.ppf(lowerQuant, loc=priorMean, scale=np.sqrt(priorVar / 2)))
    priorUpper = spsp.expit(sps.laplace.ppf(upperQuant, loc=priorMean, scale=np.sqrt(priorVar / 2)))

    SNindsSubset = range(numSN)
    SNnames = [lgDict['importerNames'][i] for i in SNindsSubset]
    SNlowers = [np.quantile(lgDict['postSamples'][:, l], lowerQuant) for l in SNindsSubset]
    SNuppers = [np.quantile(lgDict['postSamples'][:, l], upperQuant) for l in SNindsSubset]
    floorVal = 0.05
    ceilVal = 0.3
    # First group
    SNlowers1 = [i for i in SNlowers if i > floorVal]
    SNuppers1 = [SNuppers[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    SNnames1 = [SNnames[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    midpoints1 = [SNuppers1[i] - (SNuppers1[i] - SNlowers1[i]) / 2 for i in range(len(SNuppers1))]
    zippedList1 = zip(midpoints1, SNuppers1, SNlowers1, SNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    SNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    SNuppers2 = [i for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers2 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNnames2 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    midpoints2 = [SNuppers2[i] - (SNuppers2[i] - SNlowers2[i]) / 2 for i in range(len(SNuppers2))]
    zippedList2 = zip(midpoints2, SNuppers2, SNlowers2, SNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    SNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    SNuppers3 = [i for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers3 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNnames3 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    midpoints3 = [SNuppers3[i] - (SNuppers3[i] - SNlowers3[i]) / 2 for i in range(len(SNuppers3))]
    zippedList3 = zip(midpoints3, SNuppers3, SNlowers3, SNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    SNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    SNnamesSorted = SNnamesSorted1.copy()
    # sorted_pairs = sorted_pairs1.copy()
    SNnamesSorted.append(' ')
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted = SNnamesSorted + SNnamesSorted2
    # sorted_pairs = sorted_pairs + sorted_pairs2
    SNnamesSorted.append(' ')
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted = SNnamesSorted + SNnamesSorted3
    # sorted_pairs = sorted_pairs + sorted_pairs3
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted.append(' ')
    SNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(10, 10), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o-', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o-', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o-', color='red')
    plt.plot((SNnamesSorted[-1], SNnamesSorted[-1]), (priorLower, priorUpper), 'o--', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(SNnamesSorted)), SNnamesSorted, rotation=90)
    plt.title('Supply Node 90% Intervals\nManufacturer-Province Analysis',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Supply Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    fig.tight_layout()
    plt.show()
    plt.close()

    TNindsSubset = range(numTN)
    TNnames = [lgDict['outletNames'][i] for i in TNindsSubset]
    TNlowers = [np.quantile(lgDict['postSamples'][:, numSN + l], lowerQuant) for l in TNindsSubset]
    TNuppers = [np.quantile(lgDict['postSamples'][:, numSN + l], upperQuant) for l in TNindsSubset]
    floorVal = 0.05
    ceilVal = 0.3
    # First group
    TNlowers1 = [i for i in TNlowers if i > floorVal]
    TNuppers1 = [TNuppers[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    TNnames1 = [TNnames[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    midpoints1 = [TNuppers1[i] - (TNuppers1[i] - TNlowers1[i]) / 2 for i in range(len(TNuppers1))]
    zippedList1 = zip(midpoints1, TNuppers1, TNlowers1, TNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    TNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    TNuppers2 = [i for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers2 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNnames2 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    midpoints2 = [TNuppers2[i] - (TNuppers2[i] - TNlowers2[i]) / 2 for i in range(len(TNuppers2))]
    zippedList2 = zip(midpoints2, TNuppers2, TNlowers2, TNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    TNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    TNuppers3 = [i for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers3 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNnames3 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    midpoints3 = [TNuppers3[i] - (TNuppers3[i] - TNlowers3[i]) / 2 for i in range(len(TNuppers3))]
    zippedList3 = zip(midpoints3, TNuppers3, TNlowers3, TNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    TNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    TNnamesSorted = TNnamesSorted1.copy()
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted2
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted3
    TNnamesSorted.append(' ')
    TNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(10, 10), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o-', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o-', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o-', color='red')
    plt.plot((TNnamesSorted[-1], TNnamesSorted[-1]), (priorLower, priorUpper), 'o--', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(TNnamesSorted)), TNnamesSorted, rotation=90)
    plt.title('Test Node 90% Intervals\nManufacturer-Province Analysis',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Test Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    fig.tight_layout()
    plt.show()
    plt.close()

    # District as TNs; TRACKED
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    numPostSamps = 1000
    priorMean = -2.5
    priorVar = 3.5
    lowerQuant, upperQuant = 0.05, 0.95
    priorLower = spsp.expit(sps.laplace.ppf(lowerQuant, loc=priorMean, scale=np.sqrt(priorVar / 2)))
    priorUpper = spsp.expit(sps.laplace.ppf(upperQuant, loc=priorMean, scale=np.sqrt(priorVar / 2)))

    lgDict = util.testresultsfiletotable(tbl_SEN_G2_2010, csvName=False)
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    numSN, numTN = lgDict['importerNum'], lgDict['outletNum']

    SNindsSubset = range(numSN)
    SNnames = [lgDict['importerNames'][i] for i in SNindsSubset]
    SNlowers = [np.quantile(lgDict['postSamples'][:, l], lowerQuant) for l in SNindsSubset]
    SNuppers = [np.quantile(lgDict['postSamples'][:, l], upperQuant) for l in SNindsSubset]
    floorVal = 0.05
    ceilVal = 0.3
    # First group
    SNlowers1 = [i for i in SNlowers if i > floorVal]
    SNuppers1 = [SNuppers[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    SNnames1 = [SNnames[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    midpoints1 = [SNuppers1[i] - (SNuppers1[i] - SNlowers1[i]) / 2 for i in range(len(SNuppers1))]
    zippedList1 = zip(midpoints1, SNuppers1, SNlowers1, SNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    SNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    SNuppers2 = [i for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers2 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNnames2 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    midpoints2 = [SNuppers2[i] - (SNuppers2[i] - SNlowers2[i]) / 2 for i in range(len(SNuppers2))]
    zippedList2 = zip(midpoints2, SNuppers2, SNlowers2, SNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    SNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    SNuppers3 = [i for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers3 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNnames3 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    midpoints3 = [SNuppers3[i] - (SNuppers3[i] - SNlowers3[i]) / 2 for i in range(len(SNuppers3))]
    zippedList3 = zip(midpoints3, SNuppers3, SNlowers3, SNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    SNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    SNnamesSorted = SNnamesSorted1.copy()
    # sorted_pairs = sorted_pairs1.copy()
    SNnamesSorted.append(' ')
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted = SNnamesSorted + SNnamesSorted2
    # sorted_pairs = sorted_pairs + sorted_pairs2
    SNnamesSorted.append(' ')
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted = SNnamesSorted + SNnamesSorted3
    # sorted_pairs = sorted_pairs + sorted_pairs3
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted.append(' ')
    SNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(10, 6), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o--', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o--', color='orange')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o:', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o:', color='green')
    plt.plot((SNnamesSorted[-1], SNnamesSorted[-1]), (priorLower, priorUpper), 'o-', color='gray')
    plt.ylim([0, 1])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xticks(range(len(SNnamesSorted)), SNnamesSorted, rotation=90)
    plt.title('Supply Node 90% Intervals\nManufacturer-District Analysis, Tracked Setting',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Supply Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    plt.axhline(y=floorVal, color='r', linestyle='-', alpha=0.1)  # line for 'l'
    plt.axhline(y=ceilVal, color='blue', linestyle='-', alpha=0.1)  # line for 'u'
    plt.text(26.3, ceilVal + .015, 'u=30%', color='blue', alpha=0.5, size=9)
    plt.text(26.3, floorVal + .015, 'l=5%', color='r', alpha=0.5, size=9)
    fig.tight_layout()
    plt.show()
    plt.close()

    TNindsSubset = range(numTN)
    TNnames = [lgDict['outletNames'][i] for i in TNindsSubset]
    TNlowers = [np.quantile(lgDict['postSamples'][:, numSN + l], lowerQuant) for l in TNindsSubset]
    TNuppers = [np.quantile(lgDict['postSamples'][:, numSN + l], upperQuant) for l in TNindsSubset]
    floorVal = 0.05
    ceilVal = 0.3
    # First group
    TNlowers1 = [i for i in TNlowers if i > floorVal]
    TNuppers1 = [TNuppers[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    TNnames1 = [TNnames[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    midpoints1 = [TNuppers1[i] - (TNuppers1[i] - TNlowers1[i]) / 2 for i in range(len(TNuppers1))]
    zippedList1 = zip(midpoints1, TNuppers1, TNlowers1, TNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    TNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    TNuppers2 = [i for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers2 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNnames2 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    midpoints2 = [TNuppers2[i] - (TNuppers2[i] - TNlowers2[i]) / 2 for i in range(len(TNuppers2))]
    zippedList2 = zip(midpoints2, TNuppers2, TNlowers2, TNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    TNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    TNuppers3 = [i for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers3 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNnames3 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    midpoints3 = [TNuppers3[i] - (TNuppers3[i] - TNlowers3[i]) / 2 for i in range(len(TNuppers3))]
    zippedList3 = zip(midpoints3, TNuppers3, TNlowers3, TNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    TNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    TNnamesSorted = TNnamesSorted1.copy()
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted2
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted3
    TNnamesSorted.append(' ')
    TNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(10, 6), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o--', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o--', color='orange')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o:', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o:', color='green')
    plt.plot((TNnamesSorted[-1], TNnamesSorted[-1]), (priorLower, priorUpper), 'o-', color='gray')
    plt.ylim([0, 1])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xticks(range(len(TNnamesSorted)), TNnamesSorted, rotation=90)
    plt.title('Test Node 90% Intervals\nManufacturer-District Analysis, Tracked Setting',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Test Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    plt.axhline(y=floorVal, color='r', linestyle='-', alpha=0.1)  # line for 'l'
    plt.axhline(y=ceilVal, color='blue', linestyle='-', alpha=0.1)  # line for 'u'
    plt.text(24.4, ceilVal + .015, 'u=30%', color='blue', alpha=0.5, size=9)
    plt.text(24.4, floorVal + .015, 'l=5%', color='r', alpha=0.5, size=9)
    fig.tight_layout()
    plt.show()
    plt.close()


    # District as TNs; UNTRACKED
    priorMean = -2.5
    priorVar = 3.5
    lowerQuant, upperQuant = 0.05, 0.95
    priorLower = spsp.expit(sps.laplace.ppf(lowerQuant, loc=priorMean, scale=np.sqrt(priorVar / 2)))
    priorUpper = spsp.expit(sps.laplace.ppf(upperQuant, loc=priorMean, scale=np.sqrt(priorVar / 2)))
    numPostSamps = 5000
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}

    lgDict = util.testresultsfiletotable(tbl_SEN_G2_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    numSN, numTN = lgDict['importerNum'], lgDict['outletNum']

    SNindsSubset = range(numSN)
    SNnames = [lgDict['importerNames'][i] for i in SNindsSubset]
    SNlowers = [np.quantile(lgDict['postSamples'][:, l], lowerQuant) for l in SNindsSubset]
    SNuppers = [np.quantile(lgDict['postSamples'][:, l], upperQuant) for l in SNindsSubset]
    floorVal = 0.05
    ceilVal = 0.3
    # First group
    SNlowers1 = [i for i in SNlowers if i > floorVal]
    SNuppers1 = [SNuppers[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    SNnames1 = [SNnames[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    midpoints1 = [SNuppers1[i] - (SNuppers1[i] - SNlowers1[i]) / 2 for i in range(len(SNuppers1))]
    zippedList1 = zip(midpoints1, SNuppers1, SNlowers1, SNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    SNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    SNuppers2 = [i for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers2 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNnames2 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    midpoints2 = [SNuppers2[i] - (SNuppers2[i] - SNlowers2[i]) / 2 for i in range(len(SNuppers2))]
    zippedList2 = zip(midpoints2, SNuppers2, SNlowers2, SNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    SNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    SNuppers3 = [i for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers3 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNnames3 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    midpoints3 = [SNuppers3[i] - (SNuppers3[i] - SNlowers3[i]) / 2 for i in range(len(SNuppers3))]
    zippedList3 = zip(midpoints3, SNuppers3, SNlowers3, SNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    SNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    SNnamesSorted = SNnamesSorted1.copy()
    # sorted_pairs = sorted_pairs1.copy()
    SNnamesSorted.append(' ')
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted = SNnamesSorted + SNnamesSorted2
    # sorted_pairs = sorted_pairs + sorted_pairs2
    SNnamesSorted.append(' ')
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted = SNnamesSorted + SNnamesSorted3
    # sorted_pairs = sorted_pairs + sorted_pairs3
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted.append(' ')
    SNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(10, 6), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o--', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o--', color='orange')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o:', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o:', color='green')
    plt.plot((SNnamesSorted[-1], SNnamesSorted[-1]), (priorLower, priorUpper), 'o-', color='gray')
    plt.ylim([0, 1])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xticks(range(len(SNnamesSorted)), SNnamesSorted, rotation=90)
    plt.title('Supply Node 90% Intervals\nManufacturer-District Analysis, Untracked Setting',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Supply Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    plt.axhline(y=floorVal, color='r', linestyle='-', alpha=0.1)  # line for 'l'
    plt.axhline(y=ceilVal, color='blue', linestyle='-', alpha=0.1)  # line for 'u'
    plt.text(26.3, ceilVal + .015, 'u=30%', color='blue', alpha=0.5, size=9)
    plt.text(26.3, floorVal + .015, 'l=5%', color='r', alpha=0.5, size=9)
    fig.tight_layout()
    plt.show()
    plt.close()

    TNindsSubset = range(numTN)
    TNnames = [lgDict['outletNames'][i] for i in TNindsSubset]
    TNlowers = [np.quantile(lgDict['postSamples'][:, numSN + l], lowerQuant) for l in TNindsSubset]
    TNuppers = [np.quantile(lgDict['postSamples'][:, numSN + l], upperQuant) for l in TNindsSubset]
    floorVal = 0.05
    ceilVal = 0.3
    # First group
    TNlowers1 = [i for i in TNlowers if i > floorVal]
    TNuppers1 = [TNuppers[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    TNnames1 = [TNnames[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    midpoints1 = [TNuppers1[i] - (TNuppers1[i] - TNlowers1[i]) / 2 for i in range(len(TNuppers1))]
    zippedList1 = zip(midpoints1, TNuppers1, TNlowers1, TNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    TNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    TNuppers2 = [i for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers2 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNnames2 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    midpoints2 = [TNuppers2[i] - (TNuppers2[i] - TNlowers2[i]) / 2 for i in range(len(TNuppers2))]
    zippedList2 = zip(midpoints2, TNuppers2, TNlowers2, TNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    TNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    TNuppers3 = [i for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers3 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNnames3 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    midpoints3 = [TNuppers3[i] - (TNuppers3[i] - TNlowers3[i]) / 2 for i in range(len(TNuppers3))]
    zippedList3 = zip(midpoints3, TNuppers3, TNlowers3, TNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    TNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    TNnamesSorted = TNnamesSorted1.copy()
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted2
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted3
    TNnamesSorted.append(' ')
    TNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(10, 6), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o--', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o--', color='orange')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o:', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o:', color='green')
    plt.plot((TNnamesSorted[-1], TNnamesSorted[-1]), (priorLower, priorUpper), 'o-', color='gray')
    plt.ylim([0, 1])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xticks(range(len(TNnamesSorted)), TNnamesSorted, rotation=90)
    plt.title('Test Node 90% Intervals\nManufacturer-District Analysis, Untracked Setting',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Test Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    plt.axhline(y=floorVal, color='r', linestyle='-', alpha=0.1)  # line for 'l'
    plt.axhline(y=ceilVal, color='blue', linestyle='-', alpha=0.1)  # line for 'u'
    plt.text(24.4, ceilVal + .015, 'u=30%', color='blue', alpha=0.5, size=9)
    plt.text(24.4, floorVal + .015, 'l=5%', color='r', alpha=0.5, size=9)
    fig.tight_layout()
    plt.show()
    plt.close()

    # District as TNs; UNTRACKED; what if Q looked different?
    lgDict = util.testresultsfiletotable(tbl_SEN_G2_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    random.seed(31)
    for i, Nrow in enumerate(lgDict['N']):
        tempRow = Nrow / np.sum(Nrow)
        random.shuffle(tempRow)
        Q[i] = tempRow
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    numSN, numTN = lgDict['importerNum'], lgDict['outletNum']

    SNindsSubset = range(numSN)
    SNnames = [lgDict['importerNames'][i] for i in SNindsSubset]
    SNlowers = [np.quantile(lgDict['postSamples'][:, l], lowerQuant) for l in SNindsSubset]
    SNuppers = [np.quantile(lgDict['postSamples'][:, l], upperQuant) for l in SNindsSubset]
    floorVal = 0.05
    ceilVal = 0.3
    # First group
    SNlowers1 = [i for i in SNlowers if i > floorVal]
    SNuppers1 = [SNuppers[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    SNnames1 = [SNnames[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    midpoints1 = [SNuppers1[i] - (SNuppers1[i] - SNlowers1[i]) / 2 for i in range(len(SNuppers1))]
    zippedList1 = zip(midpoints1, SNuppers1, SNlowers1, SNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    SNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    SNuppers2 = [i for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers2 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNnames2 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    midpoints2 = [SNuppers2[i] - (SNuppers2[i] - SNlowers2[i]) / 2 for i in range(len(SNuppers2))]
    zippedList2 = zip(midpoints2, SNuppers2, SNlowers2, SNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    SNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    SNuppers3 = [i for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers3 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNnames3 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    midpoints3 = [SNuppers3[i] - (SNuppers3[i] - SNlowers3[i]) / 2 for i in range(len(SNuppers3))]
    zippedList3 = zip(midpoints3, SNuppers3, SNlowers3, SNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    SNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    SNnamesSorted = SNnamesSorted1.copy()
    # sorted_pairs = sorted_pairs1.copy()
    SNnamesSorted.append(' ')
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted = SNnamesSorted + SNnamesSorted2
    # sorted_pairs = sorted_pairs + sorted_pairs2
    SNnamesSorted.append(' ')
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted = SNnamesSorted + SNnamesSorted3
    # sorted_pairs = sorted_pairs + sorted_pairs3
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted.append(' ')
    SNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(10, 6), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o--', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o--', color='orange')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o:', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o:', color='green')
    plt.plot((SNnamesSorted[-1], SNnamesSorted[-1]), (priorLower, priorUpper), 'o-', color='gray')
    plt.ylim([0, 1])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xticks(range(len(SNnamesSorted)), SNnamesSorted, rotation=90)
    plt.title('Supply Node 90% Intervals\nManufacturer-District Analysis, Untracked Setting',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Supply Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    plt.axhline(y=floorVal, color='r', linestyle='-', alpha=0.1)  # line for 'l'
    plt.axhline(y=ceilVal, color='blue', linestyle='-', alpha=0.1)  # line for 'u'
    plt.text(26.3, ceilVal + .015, 'u=30%', color='blue', alpha=0.5, size=9)
    plt.text(26.3, floorVal + .015, 'l=5%', color='r', alpha=0.5, size=9)
    fig.tight_layout()
    plt.show()
    plt.close()

    TNindsSubset = range(numTN)
    TNnames = [lgDict['outletNames'][i] for i in TNindsSubset]
    TNlowers = [np.quantile(lgDict['postSamples'][:, numSN + l], lowerQuant) for l in TNindsSubset]
    TNuppers = [np.quantile(lgDict['postSamples'][:, numSN + l], upperQuant) for l in TNindsSubset]
    floorVal = 0.05
    ceilVal = 0.3
    # First group
    TNlowers1 = [i for i in TNlowers if i > floorVal]
    TNuppers1 = [TNuppers[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    TNnames1 = [TNnames[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    midpoints1 = [TNuppers1[i] - (TNuppers1[i] - TNlowers1[i]) / 2 for i in range(len(TNuppers1))]
    zippedList1 = zip(midpoints1, TNuppers1, TNlowers1, TNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    TNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    TNuppers2 = [i for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers2 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNnames2 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    midpoints2 = [TNuppers2[i] - (TNuppers2[i] - TNlowers2[i]) / 2 for i in range(len(TNuppers2))]
    zippedList2 = zip(midpoints2, TNuppers2, TNlowers2, TNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    TNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    TNuppers3 = [i for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers3 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNnames3 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    midpoints3 = [TNuppers3[i] - (TNuppers3[i] - TNlowers3[i]) / 2 for i in range(len(TNuppers3))]
    zippedList3 = zip(midpoints3, TNuppers3, TNlowers3, TNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    TNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    TNnamesSorted = TNnamesSorted1.copy()
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted2
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted3
    TNnamesSorted.append(' ')
    TNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(10, 6), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o--', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o--', color='orange')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o:', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o:', color='green')
    plt.plot((TNnamesSorted[-1], TNnamesSorted[-1]), (priorLower, priorUpper), 'o-', color='gray')
    plt.ylim([0, 1])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xticks(range(len(TNnamesSorted)), TNnamesSorted, rotation=90)
    plt.title('Test Node 90% Intervals\nManufacturer-District Analysis, Untracked Setting',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Test Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    plt.axhline(y=floorVal, color='r', linestyle='-', alpha=0.1)  # line for 'l'
    plt.axhline(y=ceilVal, color='blue', linestyle='-', alpha=0.1)  # line for 'u'
    plt.text(24.4, ceilVal + .015, 'u=30%', color='blue', alpha=0.5, size=9)
    plt.text(24.4, floorVal + .015, 'l=5%', color='r', alpha=0.5, size=9)
    fig.tight_layout()
    plt.show()
    plt.close()

    # District as TNs; UNTRACKED; what if Q looked different? USE BOOTSTRAP SAMPLES FOR ESTIMATES OF Q
    priorMean, priorVar = -2.5, 3.5
    lowerQuant, upperQuant = 0.05, 0.95
    btlowerQuant, btupperQuant = 0.05, 0.95
    priorLower = spsp.expit(sps.laplace.ppf(lowerQuant, loc=priorMean, scale=np.sqrt(priorVar / 2)))
    priorUpper = spsp.expit(sps.laplace.ppf(upperQuant, loc=priorMean, scale=np.sqrt(priorVar / 2)))

    # First get posterior draws from original Q estimate for later comparison
    lgDict = util.testresultsfiletotable(tbl_SEN_G2_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    origpostdraws = lgDict['postSamples']

    # Now use bootstrapped Q estimates
    lgDict = util.testresultsfiletotable(tbl_SEN_G2_2010, csvName=False)
    outNum, impNum = lgDict['N'].shape
    # Update N and Y
    Nmat = lgDict['N']
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'importerNum': impNum, 'outletNum': outNum})
    # Initialize empty Q
    Q = np.zeros(shape=(outNum, impNum))
    random.seed(313)  # Set a seed
    btuppers, btlowers = [], []
    for rep in range(100):  # SET BOOTSTRAP REPLICATIONS HERE
        print('Bootstrap sample: '+str(rep)+'...')
        # Every row needs at least one observation; add 1 for each row first before bootstrap (23 total)
        for rw in range(Q.shape[0]):
            currProbs = Nmat[rw] / np.sum(Nmat[rw])
            Q[rw, random.choices(range(Q.shape[1]), currProbs)] += 1  # Add one to the chosen element
        # Now pull bootstrap samples
        for btsamp in range(len(lgDict['dataTbl']) - Q.shape[
            0]):  # Pull samples equal to number of data points in original data set
            currSamp = random.choices(lgDict['dataTbl'])[0]
            rw, col = lgDict['outletNames'].index(currSamp[0]), lgDict['importerNames'].index(currSamp[1])
            Q[rw, col] += 1
        # Normalize
        for ind, rw in enumerate(Q):
            newrw = rw / np.sum(rw)
            Q[ind] = newrw
        # Generate posterior draws using this bootstrapped Q
        lgDict.update({'transMat': Q})
        lgDict = methods.GeneratePostSamples(lgDict) # Overwrites any old samples
        # Grab 90% upper and lower bounds for each node
        currpostdraws = lgDict['postSamples']
        newuppers = [np.quantile(currpostdraws[:, i], upperQuant) for i in range(impNum + outNum)]
        newlowers = [np.quantile(currpostdraws[:, i], lowerQuant) for i in range(impNum + outNum)]
        btuppers.append(newuppers)
        btlowers.append(newlowers)

    btuppers = np.array(btuppers)
    btlowers = np.array(btlowers)
    # Generate plots that show the range of inference when using the bootstrap samples
    numSN, numTN = lgDict['importerNum'], lgDict['outletNum']

    SNindsSubset = range(numSN)
    SNnames = [lgDict['importerNames'][i] for i in SNindsSubset]
    SNlowers = [np.quantile(origpostdraws[:, l], lowerQuant) for l in SNindsSubset]
    SNuppers = [np.quantile(origpostdraws[:, l], upperQuant) for l in SNindsSubset]
    # Bootstrap ranges
    SNlowers_bthigh = [np.quantile(btlowers[:, l], btupperQuant) for l in SNindsSubset]
    SNlowers_btlow = [np.quantile(btlowers[:, l], btlowerQuant) for l in SNindsSubset]
    SNuppers_bthigh = [np.quantile(btuppers[:, l], btupperQuant) for l in SNindsSubset]
    SNuppers_btlow = [np.quantile(btuppers[:, l], btlowerQuant) for l in SNindsSubset]

    floorVal = 0.05
    ceilVal = 0.3
    # First group
    SNlowers1 = [i for i in SNlowers if i > floorVal]
    SNuppers1 = [SNuppers[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    SNnames1 = [SNnames[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    midpoints1 = [SNuppers1[i] - (SNuppers1[i] - SNlowers1[i]) / 2 for i in range(len(SNuppers1))]
    zippedList1 = zip(midpoints1, SNuppers1, SNlowers1, SNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    SNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    SNuppers2 = [i for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers2 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNnames2 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    midpoints2 = [SNuppers2[i] - (SNuppers2[i] - SNlowers2[i]) / 2 for i in range(len(SNuppers2))]
    zippedList2 = zip(midpoints2, SNuppers2, SNlowers2, SNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    SNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    SNuppers3 = [i for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers3 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNnames3 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    midpoints3 = [SNuppers3[i] - (SNuppers3[i] - SNlowers3[i]) / 2 for i in range(len(SNuppers3))]
    zippedList3 = zip(midpoints3, SNuppers3, SNlowers3, SNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    SNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    SNnamesSorted = SNnamesSorted1.copy()
    # sorted_pairs = sorted_pairs1.copy()
    SNnamesSorted.append(' ')
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted = SNnamesSorted + SNnamesSorted2
    # sorted_pairs = sorted_pairs + sorted_pairs2
    SNnamesSorted.append(' ')
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted = SNnamesSorted + SNnamesSorted3
    # sorted_pairs = sorted_pairs + sorted_pairs3
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted.append(' ')
    SNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(10, 6), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o--', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o--', color='orange')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o:', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o:', color='green')
    plt.plot((SNnamesSorted[-1], SNnamesSorted[-1]), (priorLower, priorUpper), 'o-', color='gray')
    plt.ylim([0, 1])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xticks(range(len(SNnamesSorted)), SNnamesSorted, rotation=90)
    plt.title('Supply Node 90% Intervals w/ Bounds from 100 Bootstrap Samples\nManufacturer-District Analysis, Untracked Setting',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Supply Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    plt.axhline(y=floorVal, color='r', linestyle='-', alpha=0.1)  # line for 'l'
    plt.axhline(y=ceilVal, color='blue', linestyle='-', alpha=0.1)  # line for 'u'
    plt.text(26.3, ceilVal + .015, 'u=30%', color='blue', alpha=0.5, size=9)
    plt.text(26.3, floorVal + .015, 'l=5%', color='r', alpha=0.5, size=9)
    # Put in bootstrap upper and lower intervals
    SNnames_bt = SNnamesSorted1 + SNnamesSorted2 + SNnamesSorted3
    btuppers_high_sorted = [SNuppers_bthigh[SNnames.index(nm)] for nm in SNnames_bt]
    btuppers_low_sorted = [SNuppers_btlow[SNnames.index(nm)] for nm in SNnames_bt]
    btlowers_high_sorted = [SNlowers_bthigh[SNnames.index(nm)] for nm in SNnames_bt]
    btlowers_low_sorted = [SNlowers_btlow[SNnames.index(nm)] for nm in SNnames_bt]
    for ind, nm in enumerate(SNnames_bt):
        plt.plot((nm, nm), (btuppers_low_sorted[ind], btuppers_high_sorted[ind]), '_-',
                 color='k',alpha=0.3)
    for ind, nm in enumerate(SNnames_bt):
        plt.plot((nm, nm), (btlowers_low_sorted[ind], btlowers_high_sorted[ind]), '_-',
                 color='k', alpha=0.3)
    ###
    fig.tight_layout()
    plt.show()
    plt.close()

    TNindsSubset = range(numTN)
    TNnames = [lgDict['outletNames'][i] for i in TNindsSubset]
    TNlowers = [np.quantile(origpostdraws[:, numSN + l], lowerQuant) for l in TNindsSubset]
    TNuppers = [np.quantile(origpostdraws[:, numSN + l], upperQuant) for l in TNindsSubset]
    # Bootstrap ranges
    TNlowers_bthigh = [np.quantile(btlowers[:, numSN + l], btupperQuant) for l in TNindsSubset]
    TNlowers_btlow = [np.quantile(btlowers[:, numSN + l], btlowerQuant) for l in TNindsSubset]
    TNuppers_bthigh = [np.quantile(btuppers[:, numSN + l], btupperQuant) for l in TNindsSubset]
    TNuppers_btlow = [np.quantile(btuppers[:, numSN + l], btlowerQuant) for l in TNindsSubset]

    floorVal = 0.05
    ceilVal = 0.3
    # First group
    TNlowers1 = [i for i in TNlowers if i > floorVal]
    TNuppers1 = [TNuppers[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    TNnames1 = [TNnames[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    midpoints1 = [TNuppers1[i] - (TNuppers1[i] - TNlowers1[i]) / 2 for i in range(len(TNuppers1))]
    zippedList1 = zip(midpoints1, TNuppers1, TNlowers1, TNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    TNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    TNuppers2 = [i for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers2 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNnames2 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    midpoints2 = [TNuppers2[i] - (TNuppers2[i] - TNlowers2[i]) / 2 for i in range(len(TNuppers2))]
    zippedList2 = zip(midpoints2, TNuppers2, TNlowers2, TNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    TNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    TNuppers3 = [i for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers3 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNnames3 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    midpoints3 = [TNuppers3[i] - (TNuppers3[i] - TNlowers3[i]) / 2 for i in range(len(TNuppers3))]
    zippedList3 = zip(midpoints3, TNuppers3, TNlowers3, TNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    TNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    TNnamesSorted = TNnamesSorted1.copy()
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted2
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted3
    TNnamesSorted.append(' ')
    TNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(10, 6), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o--', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o--', color='orange')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o:', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o:', color='green')
    plt.plot((TNnamesSorted[-1], TNnamesSorted[-1]), (priorLower, priorUpper), 'o-', color='gray')
    plt.ylim([0, 1])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xticks(range(len(TNnamesSorted)), TNnamesSorted, rotation=90)
    plt.title('Test Node 90% Intervals w/ Bounds from 100 Bootstrap Samples\nManufacturer-District Analysis, Untracked Setting',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Test Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    plt.axhline(y=floorVal, color='r', linestyle='-', alpha=0.1)  # line for 'l'
    plt.axhline(y=ceilVal, color='blue', linestyle='-', alpha=0.1)  # line for 'u'
    plt.text(24.4, ceilVal + .015, 'u=30%', color='blue', alpha=0.5, size=9)
    plt.text(24.4, floorVal + .015, 'l=5%', color='r', alpha=0.5, size=9)
    # Put in bootstrap upper and lower intervals
    TNnames_bt = TNnamesSorted1 + TNnamesSorted2 + TNnamesSorted3
    btuppers_high_sorted = [TNuppers_bthigh[TNnames.index(nm)] for nm in TNnames_bt]
    btuppers_low_sorted = [TNuppers_btlow[TNnames.index(nm)] for nm in TNnames_bt]
    btlowers_high_sorted = [TNlowers_bthigh[TNnames.index(nm)] for nm in TNnames_bt]
    btlowers_low_sorted = [TNlowers_btlow[TNnames.index(nm)] for nm in TNnames_bt]
    for ind, nm in enumerate(TNnames_bt):
        plt.plot((nm, nm), (btuppers_low_sorted[ind], btuppers_high_sorted[ind]), '_-',
                 color='k', alpha=0.3)
    for ind, nm in enumerate(TNnames_bt):
        plt.plot((nm, nm), (btlowers_low_sorted[ind], btlowers_high_sorted[ind]), '_-',
                 color='k', alpha=0.3)
    ###
    fig.tight_layout()
    plt.show()
    plt.close()

    #########################
    #########################
    # Facility Location as TNs
    lgDict = util.testresultsfiletotable(tbl_SEN_G3_2010, csvName=False)
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    numSN, numTN = lgDict['importerNum'], lgDict['outletNum']

    SNindsSubset = range(numSN)
    SNnames = [lgDict['importerNames'][i] for i in SNindsSubset]
    SNlowers = [np.quantile(lgDict['postSamples'][:, l], lowerQuant) for l in SNindsSubset]
    SNuppers = [np.quantile(lgDict['postSamples'][:, l], upperQuant) for l in SNindsSubset]
    floorVal = 0.05
    ceilVal = 0.3
    # First group
    SNlowers1 = [i for i in SNlowers if i > floorVal]
    SNuppers1 = [SNuppers[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    SNnames1 = [SNnames[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    midpoints1 = [SNuppers1[i] - (SNuppers1[i] - SNlowers1[i]) / 2 for i in range(len(SNuppers1))]
    zippedList1 = zip(midpoints1, SNuppers1, SNlowers1, SNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    SNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    SNuppers2 = [i for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers2 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNnames2 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    midpoints2 = [SNuppers2[i] - (SNuppers2[i] - SNlowers2[i]) / 2 for i in range(len(SNuppers2))]
    zippedList2 = zip(midpoints2, SNuppers2, SNlowers2, SNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    SNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    SNuppers3 = [i for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers3 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNnames3 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    midpoints3 = [SNuppers3[i] - (SNuppers3[i] - SNlowers3[i]) / 2 for i in range(len(SNuppers3))]
    zippedList3 = zip(midpoints3, SNuppers3, SNlowers3, SNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    SNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    SNnamesSorted = SNnamesSorted1.copy()
    # sorted_pairs = sorted_pairs1.copy()
    SNnamesSorted.append(' ')
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted = SNnamesSorted + SNnamesSorted2
    # sorted_pairs = sorted_pairs + sorted_pairs2
    SNnamesSorted.append(' ')
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted = SNnamesSorted + SNnamesSorted3
    # sorted_pairs = sorted_pairs + sorted_pairs3
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted.append(' ')
    SNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(10, 10), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o-', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o-', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o-', color='red')
    plt.plot((SNnamesSorted[-1], SNnamesSorted[-1]), (priorLower, priorUpper), 'o--', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(SNnamesSorted)), SNnamesSorted, rotation=90)
    plt.title('Supply Node 90% Intervals\nManufacturer-Facility Analysis',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Supply Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    fig.tight_layout()
    plt.show()
    plt.close()

    TNindsSubset = range(numTN)
    TNnames = [lgDict['outletNames'][i] for i in TNindsSubset]
    TNlowers = [np.quantile(lgDict['postSamples'][:, numSN + l], lowerQuant) for l in TNindsSubset]
    TNuppers = [np.quantile(lgDict['postSamples'][:, numSN + l], upperQuant) for l in TNindsSubset]
    floorVal = 0.05
    ceilVal = 0.3
    # First group
    TNlowers1 = [i for i in TNlowers if i > floorVal]
    TNuppers1 = [TNuppers[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    TNnames1 = [TNnames[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    midpoints1 = [TNuppers1[i] - (TNuppers1[i] - TNlowers1[i]) / 2 for i in range(len(TNuppers1))]
    zippedList1 = zip(midpoints1, TNuppers1, TNlowers1, TNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    TNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    TNuppers2 = [i for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers2 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNnames2 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    midpoints2 = [TNuppers2[i] - (TNuppers2[i] - TNlowers2[i]) / 2 for i in range(len(TNuppers2))]
    zippedList2 = zip(midpoints2, TNuppers2, TNlowers2, TNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    TNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    TNuppers3 = [i for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers3 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNnames3 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    midpoints3 = [TNuppers3[i] - (TNuppers3[i] - TNlowers3[i]) / 2 for i in range(len(TNuppers3))]
    zippedList3 = zip(midpoints3, TNuppers3, TNlowers3, TNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    TNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    TNnamesSorted = TNnamesSorted1.copy()
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted2
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted3
    TNnamesSorted.append(' ')
    TNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(10, 10), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o-', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o-', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o-', color='red')
    plt.plot((TNnamesSorted[-1], TNnamesSorted[-1]), (priorLower, priorUpper), 'o--', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(TNnamesSorted)), TNnamesSorted, rotation=90)
    plt.title('Test Node 90% Intervals\nManufacturer-Facility Analysis',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Test Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(9)
    fig.tight_layout()
    plt.show()
    plt.close()

    # What does a good prior look like?
    mean = -2.5
    var = 1.5
    s = np.random.laplace(mean, np.sqrt(var / 2), 10000)
    t = np.exp(s) / (1 + np.exp(s))
    print(np.mean(t))
    import matplotlib.pyplot as plt
    plt.hist(s, density=True, bins=30)
    plt.show()
    plt.hist(t, density=True, bins=30)
    plt.show()

    mean = -2.5
    var = 1.5
    s = np.random.normal(mean, np.sqrt(var), 10000)
    t = np.exp(s) / (1 + np.exp(s))
    print(np.mean(t))
    plt.hist(s, density=True, bins=30)
    plt.show()
    plt.hist(t, density=True, bins=30)
    plt.show()

    import scipy.stats as sps
    import scipy.special as spsp
    int50 = sps.laplace.ppf(0.50, loc=mean, scale=np.sqrt(var / 2))
    int05 = sps.laplace.ppf(0.05, loc=mean, scale=np.sqrt(var / 2))
    int95 = sps.laplace.ppf(0.95, loc=mean, scale=np.sqrt(var / 2))
    int70 = sps.laplace.ppf(0.70, loc=mean, scale=np.sqrt(var / 2))
    print(spsp.expit(int05), spsp.expit(int50), spsp.expit(int70), spsp.expit(int95))
    print(spsp.expit(int05), spsp.expit(int95))

    # Generate samples for paper example in Section 3, to be used in Section 5
    lgDict = {}
    priorMean, priorVar = -2, 1
    numPostSamps = 1000
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    int50 = sps.norm.ppf(0.50, loc=priorMean, scale=np.sqrt(priorVar))
    int05 = sps.norm.ppf(0.05, loc=priorMean, scale=np.sqrt(priorVar))
    int95 = sps.norm.ppf(0.95, loc=priorMean, scale=np.sqrt(priorVar))
    int70 = sps.norm.ppf(0.70, loc=priorMean, scale=np.sqrt(priorVar))
    print(spsp.expit(int05), spsp.expit(int50), spsp.expit(int70), spsp.expit(int95))
    Ntoy = np.array([[6, 11], [12, 6], [2, 13]])
    Ytoy = np.array([[3, 0], [6, 0], [0, 0]])
    TNnames, SNnames = ['Test Node 1', 'Test Node 2', 'Test Node 3'], ['Supply Node 1', 'Supply Node 2']
    lgDict.update({'type': 'Tracked', 'outletNum': 3, 'importerNum': 2, 'diagSens': 1.0, 'diagSpec': 1.0,
                   'N': Ntoy, 'Y': Ytoy, 'numPostSamples': numPostSamps, 'MCMCdict': MCMCdict,
                   'outletNames': TNnames, 'importerNames': SNnames,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar)})
    lgDict = methods.GeneratePostSamples(lgDict)
    numSN, numTN = lgDict['importerNum'], lgDict['outletNum']
    lowerQuant, upperQuant = 0.05, 0.95
    priorLower = spsp.expit(sps.norm.ppf(lowerQuant, loc=priorMean, scale=np.sqrt(priorVar)))
    priorUpper = spsp.expit(sps.norm.ppf(upperQuant, loc=priorMean, scale=np.sqrt(priorVar)))
    SNindsSubset = range(numSN)

    SNnames = [lgDict['importerNames'][i] for i in SNindsSubset]
    SNlowers = [np.quantile(lgDict['postSamples'][:, l], lowerQuant) for l in SNindsSubset]
    SNuppers = [np.quantile(lgDict['postSamples'][:, l], upperQuant) for l in SNindsSubset]
    floorVal = 0.05
    ceilVal = 0.2
    # First group
    SNlowers1 = [i for i in SNlowers if i > floorVal]
    SNuppers1 = [SNuppers[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    SNnames1 = [SNnames[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    midpoints1 = [SNuppers1[i] - (SNuppers1[i] - SNlowers1[i]) / 2 for i in range(len(SNuppers1))]
    zippedList1 = zip(midpoints1, SNuppers1, SNlowers1, SNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    SNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    SNuppers2 = [i for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers2 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNnames2 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    midpoints2 = [SNuppers2[i] - (SNuppers2[i] - SNlowers2[i]) / 2 for i in range(len(SNuppers2))]
    zippedList2 = zip(midpoints2, SNuppers2, SNlowers2, SNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    SNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    SNuppers3 = [i for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers3 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNnames3 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    midpoints3 = [SNuppers3[i] - (SNuppers3[i] - SNlowers3[i]) / 2 for i in range(len(SNuppers3))]
    zippedList3 = zip(midpoints3, SNuppers3, SNlowers3, SNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    SNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    SNnamesSorted = SNnamesSorted1.copy()
    # sorted_pairs = sorted_pairs1.copy()
    SNnamesSorted.append(' ')
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted = SNnamesSorted + SNnamesSorted2
    # sorted_pairs = sorted_pairs + sorted_pairs2
    SNnamesSorted.append(' ')
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted = SNnamesSorted + SNnamesSorted3
    # sorted_pairs = sorted_pairs + sorted_pairs3
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted.append(' ')
    SNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(5, 5), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o--', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o--', color='orange')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o:', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o:', color='green')
    plt.plot((SNnamesSorted[-1], SNnamesSorted[-1]), (priorLower, priorUpper), 'o-', color='gray')
    plt.ylim([0, 1])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xticks(range(len(SNnamesSorted)), SNnamesSorted, rotation=90)
    plt.title('Supply Node 90% Intervals\nExample',
              fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Supply Node Name', fontdict={'fontsize': 12, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 12, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(11)
    plt.axhline(y=floorVal, color='r', linestyle='-', alpha=0.3)  # line for 'l'
    plt.axhline(y=ceilVal, color='blue', linestyle='-', alpha=0.3)  # line for 'u'
    fig.tight_layout()
    plt.show()
    plt.close()

    TNindsSubset = range(numTN)
    TNnames = [lgDict['outletNames'][i] for i in TNindsSubset]
    TNlowers = [np.quantile(lgDict['postSamples'][:, numSN + l], lowerQuant) for l in TNindsSubset]
    TNuppers = [np.quantile(lgDict['postSamples'][:, numSN + l], upperQuant) for l in TNindsSubset]
    floorVal = 0.05
    ceilVal = 0.2
    # First group
    TNlowers1 = [i for i in TNlowers if i > floorVal]
    TNuppers1 = [TNuppers[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    TNnames1 = [TNnames[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    midpoints1 = [TNuppers1[i] - (TNuppers1[i] - TNlowers1[i]) / 2 for i in range(len(TNuppers1))]
    zippedList1 = zip(midpoints1, TNuppers1, TNlowers1, TNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    TNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    TNuppers2 = [i for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers2 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNnames2 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    midpoints2 = [TNuppers2[i] - (TNuppers2[i] - TNlowers2[i]) / 2 for i in range(len(TNuppers2))]
    zippedList2 = zip(midpoints2, TNuppers2, TNlowers2, TNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    TNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    TNuppers3 = [i for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers3 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNnames3 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    midpoints3 = [TNuppers3[i] - (TNuppers3[i] - TNlowers3[i]) / 2 for i in range(len(TNuppers3))]
    zippedList3 = zip(midpoints3, TNuppers3, TNlowers3, TNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    TNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    TNnamesSorted = TNnamesSorted1.copy()
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted2
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted3
    TNnamesSorted.append(' ')
    TNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(5, 5), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o--', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o--', color='orange')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o:', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o:', color='green')
    plt.plot((TNnamesSorted[-1], TNnamesSorted[-1]), (priorLower, priorUpper), 'o-', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(TNnamesSorted)), TNnamesSorted, rotation=90)
    plt.title('Test Node 90% Intervals\nExample',
              fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Test Node Name', fontdict={'fontsize': 12, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 12, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(11)
    plt.axhline(y=floorVal, color='r', linestyle='-', alpha=0.3)  # line for 'l'
    plt.axhline(y=ceilVal, color='blue', linestyle='-', alpha=0.3)  # line for 'u'
    fig.tight_layout()
    plt.show()
    plt.close()

    # COMBINED INTO ONE PLOT; FORMATTED FOR VERY PARTICULAR DATA SET, E.G., SKIPS HIGH RISK INTERVALS FOR TEST NODES
    floorVal = 0.05
    ceilVal = 0.2
    # First group
    SNlowers1 = [i for i in SNlowers if i > floorVal]
    SNuppers1 = [SNuppers[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    SNnames1 = [SNnames[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    midpoints1 = [SNuppers1[i] - (SNuppers1[i] - SNlowers1[i]) / 2 for i in range(len(SNuppers1))]
    zippedList1 = zip(midpoints1, SNuppers1, SNlowers1, SNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    SNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    TNuppers2 = [i for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers2 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNnames2 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    midpoints2 = [TNuppers2[i] - (TNuppers2[i] - TNlowers2[i]) / 2 for i in range(len(TNuppers2))]
    zippedList2 = zip(midpoints2, TNuppers2, TNlowers2, TNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    TNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    TNuppers3 = [i for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers3 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNnames3 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNmidpoints3 = [TNuppers3[i] - (TNuppers3[i] - TNlowers3[i]) / 2 for i in range(len(TNuppers3))]
    TNzippedList3 = zip(TNmidpoints3, TNuppers3, TNlowers3, TNnames3)
    TNsorted_pairs3 = sorted(TNzippedList3, reverse=True)
    TNnamesSorted3 = [tup[-1] for tup in TNsorted_pairs3]

    SNuppers3 = [i for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers3 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNnames3 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNmidpoints3 = [SNuppers3[i] - (SNuppers3[i] - SNlowers3[i]) / 2 for i in range(len(SNuppers3))]
    SNzippedList3 = zip(SNmidpoints3, SNuppers3, SNlowers3, SNnames3)
    SNsorted_pairs3 = sorted(SNzippedList3, reverse=True)
    SNnamesSorted3 = [tup[-1] for tup in SNsorted_pairs3]

    midpoints3 = TNmidpoints3 + SNmidpoints3
    uppers3 = TNuppers3 + SNuppers3
    lowers3 = TNlowers3 + SNlowers3
    names3 = TNnames3 + SNnames3
    zippedList3 = zip(midpoints3, uppers3, lowers3, names3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)

    # Combine groups
    namesSorted = SNnamesSorted1.copy()
    # namesSorted.append(' ')
    namesSorted = namesSorted + TNnamesSorted2
    # namesSorted.append(' ')
    namesSorted = namesSorted + TNnamesSorted3 + SNnamesSorted3
    namesSorted.append(' ')
    namesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(5, 5), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        # plt.plot((name, name), (lower, upper), 'o-', color='red')
        plt.plot((name, name), (lower, upper), 'o-', color='blue')
    #   plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        # plt.plot((name, name), (lower, upper), 'o--', color='orange')
        plt.plot((name, name), (lower, upper), 'o-', color='blue')
    # plt.plot((' ', ' '), (np.nan, np.nan), 'o--', color='orange')
    for _, upper, lower, name in sorted_pairs3:
        # plt.plot((name, name), (lower, upper), 'o:', color='green')
        plt.plot((name, name), (lower, upper), 'o-', color='blue')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o:', color='green')
    plt.plot((namesSorted[-1], namesSorted[-1]), (priorLower, priorUpper), 'o-', color='gray')
    plt.ylim([0, 0.6])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xticks(range(len(namesSorted)), namesSorted, rotation=90)
    plt.title('Node 90% Intervals',
              fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Node Name', fontdict={'fontsize': 12, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 12, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(11)
    # plt.axhline(y=floorVal, color='r', linestyle='-', alpha=0.1)  # line for 'l'
    # plt.axhline(y=ceilVal, color='blue', linestyle='-', alpha=0.1)  # line for 'u'
    # plt.text(6.7, 0.215, 'u=0.20', color='blue', alpha=0.5)
    # plt.text(6.7, 0.065, 'l=0.05', color='r', alpha=0.5)
    fig.tight_layout()
    plt.show()
    plt.close()

    # 90% CI VALUES USING NEWTON, 2009
    for i in range(numSN):  # sum across TNs to see totals for SNs
        currTotal = np.sum(lgDict['N'], axis=0)[i]
        currPos = np.sum(lgDict['Y'], axis=0)[i]
        pHat = currPos / currTotal
        lowerBd = pHat - (1.645 * np.sqrt(pHat * (1 - pHat) / currTotal))
        upperBd = pHat + (1.645 * np.sqrt(pHat * (1 - pHat) / currTotal))
        print(lgDict['importerNames'][i] + ': (' + str(lowerBd)[:5] + ', ' + str(upperBd)[:5] + ')')
    # Test nodes
    for i in range(numTN):  # sum across SNs to see totals for TNs
        currTotal = np.sum(lgDict['N'], axis=1)[i]
        currPos = np.sum(lgDict['Y'], axis=1)[i]
        pHat = currPos / currTotal
        lowerBd = pHat - (1.645 * np.sqrt(pHat * (1 - pHat) / currTotal))
        upperBd = pHat + (1.645 * np.sqrt(pHat * (1 - pHat) / currTotal))
        print(lgDict['outletNames'][i] + ': (' + str(lowerBd)[:5] + ', ' + str(upperBd)[:5] + ')')

    # TIMING ANALYSIS
    import time

    numPostSamps = 1000
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}

    # Look at difference in runtimes for HMC and LMC
    times1 = []
    times2 = []
    for runs in range(20):
        testSysDict = util.generateRandDataDict(numImp=50, numOut=50, diagSens=0.90,
                                                diagSpec=0.99, numSamples=50 * 20,
                                                dataType='Tracked', transMatLambda=1.1,
                                                randSeed=-1, trueRates=[])
        testSysDict = util.GetVectorForms(testSysDict)
        print(np.count_nonzero(testSysDict['N']))
        priorMean, priorVar = -2.4, 1
        MCMCdict.update({'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4})
        testSysDict.update({'numPostSamples': numPostSamps, 'MCMCdict': MCMCdict,
                            'prior': methods.prior_normal(mu=priorMean, var=priorVar)})
        startTime = time.time()
        testSysDict = methods.GeneratePostSamples(testSysDict)
        endTime = time.time()
        print(endTime - startTime)
        times1.append(endTime - startTime)
        MCMCdict.update({'MCMCtype': 'Langevin'})
        testSysDict.update({'MCMCdict': MCMCdict})
        startTime = time.time()
        testSysDict = methods.GeneratePostSamples(testSysDict)
        endTime = time.time()
        print(endTime - startTime)
        times2.append(endTime - startTime)
    print(np.max(times1), np.min(times1), np.mean(times1))
    print(np.max(times2), np.min(times2), np.mean(times2))
    # Look at effect of more supply-chain traces
    baseN = [346, 318, 332, 331, 361, 348, 351, 321, 334, 341, 322, 328, 315, 307, 341, 333, 331, 344, 334, 323]
    print(np.mean(baseN) / (50 * 50))
    MCMCdict.update({'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4})
    times3 = []  # Less supply-chain traces
    lowerN = []
    for runs in range(20):
        testSysDict = util.generateRandDataDict(numImp=50, numOut=50, diagSens=0.90,
                                                diagSpec=0.99, numSamples=50 * 20,
                                                dataType='Tracked', transMatLambda=0.5,
                                                randSeed=-1, trueRates=[])
        testSysDict = util.GetVectorForms(testSysDict)
        lowerN.append(np.count_nonzero(testSysDict['N']))
        priorMean, priorVar = -2.4, 1
        testSysDict.update({'numPostSamples': numPostSamps, 'MCMCdict': MCMCdict,
                            'prior': methods.prior_normal(mu=priorMean, var=priorVar)})
        startTime = time.time()
        testSysDict = methods.GeneratePostSamples(testSysDict)
        endTime = time.time()
        print(endTime - startTime)
        times3.append(endTime - startTime)
    print(np.max(times3), np.min(times3), np.mean(times3))
    print(np.average(lowerN) / (50 * 50))
    times4 = []  # More supply-chain traces
    upperN = []
    for runs in range(20):
        testSysDict = util.generateRandDataDict(numImp=50, numOut=50, diagSens=0.90,
                                                diagSpec=0.99, numSamples=50 * 20,
                                                dataType='Tracked', transMatLambda=4.5,
                                                randSeed=-1, trueRates=[])
        testSysDict = util.GetVectorForms(testSysDict)
        upperN.append(np.count_nonzero(testSysDict['N']))
        priorMean, priorVar = -2.4, 1
        testSysDict.update({'numPostSamples': numPostSamps, 'MCMCdict': MCMCdict,
                            'prior': methods.prior_normal(mu=priorMean, var=priorVar)})
        startTime = time.time()
        testSysDict = methods.GeneratePostSamples(testSysDict)
        endTime = time.time()
        print(endTime - startTime)
        times4.append(endTime - startTime)
    print(np.max(times4), np.min(times4), np.mean(times4))
    print(np.average(upperN) / (50 * 50))
    # Look at effect of less or more nodes
    times5 = []  # Less nodes
    upperN = []
    for runs in range(20):
        testSysDict = util.generateRandDataDict(numImp=25, numOut=25, diagSens=0.90,
                                                diagSpec=0.99, numSamples=50 * 20,
                                                dataType='Tracked', transMatLambda=1.1,
                                                randSeed=-1, trueRates=[])
        testSysDict = util.GetVectorForms(testSysDict)
        upperN.append(np.count_nonzero(testSysDict['N']))
        priorMean, priorVar = -2.4, 1
        testSysDict.update({'numPostSamples': numPostSamps, 'MCMCdict': MCMCdict,
                            'prior': methods.prior_normal(mu=priorMean, var=priorVar)})
        startTime = time.time()
        testSysDict = methods.GeneratePostSamples(testSysDict)
        endTime = time.time()
        print(endTime - startTime)
        times5.append(endTime - startTime)
    print(np.max(times5), np.min(times5), np.mean(times5))
    print(np.average(upperN) / (25 * 25))
    times6 = []  # More nodes
    upperN = []
    for runs in range(20):
        testSysDict = util.generateRandDataDict(numImp=100, numOut=100, diagSens=0.90,
                                                diagSpec=0.99, numSamples=50 * 20,
                                                dataType='Tracked', transMatLambda=1.1,
                                                randSeed=-1, trueRates=[])
        testSysDict = util.GetVectorForms(testSysDict)
        upperN.append(np.count_nonzero(testSysDict['N']))
        priorMean, priorVar = -2.4, 1
        testSysDict.update({'numPostSamples': numPostSamps, 'MCMCdict': MCMCdict,
                            'prior': methods.prior_normal(mu=priorMean, var=priorVar)})
        startTime = time.time()
        testSysDict = methods.GeneratePostSamples(testSysDict)
        endTime = time.time()
        print(endTime - startTime)
        times6.append(endTime - startTime)
    print(np.max(times6), np.min(times6), np.mean(times6))
    print(np.average(upperN) / (100 * 100))

    ##### END OF MANUAL PLOT GENERATION #####

    # RUN 3: s=1.0, r=1.0, prior is laplace(-3.5, 3.5)
    priorMean = -3.5
    priorVar = 3.5

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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

    # RUN 4: s=1.0, r=1.0, prior is laplace(-3.5, 1.5)
    priorMean = -3.5
    priorVar = 1.5

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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

    import numpy as np

    # RUN 5: s=1.0, r=1.0, prior is laplace(-2.5, 3.5 ) ; UNTRACKED
    priorMean = -2.5
    priorVar = 3.5

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Name', '\nSenegal - Facility Name'])

    # RUN 6: s=1.0, r=1.0, prior is laplace(-2.5, 1.5 ) ; UNTRACKED
    priorMean = -2.5
    priorVar = 1.5

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Name', '\nSenegal - Facility Name'])

    # RUN 7: s=1.0, r=1.0, prior is laplace(-2.5, 1.5 ) ; UNTRACKED
    priorMean = -3.5
    priorVar = 3.5

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Name', '\nSenegal - Facility Name'])

    # RUN 8: s=1.0, r=1.0, prior is laplace(-2.5, 1.5 ) ; UNTRACKED
    priorMean = -3.5
    priorVar = 1.5

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Name', '\nSenegal - Facility Name'])

    # RUN 9: s=0.8, r=0.95, prior is MQDB countries with at least 1 SFP
    priorMean = -2.5
    priorVar = 3.5
    s, r = 0.8, 0.95

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Name', '\nSenegal - Facility Name'])

    # RUN 10: s=0.8, r=0.95, prior is MQDB countries with at least 1 SFP, with 5 times the variance
    priorMean = -2.5
    priorVar = 1.5
    s, r = 0.8, 0.95

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Name', '\nSenegal - Facility Name'])

    # RUN 11: s=0.8, r=0.95, prior is Ozawa Africa countries with n>=150
    priorMean = -3.5
    priorVar = 3.5
    s, r = 0.8, 0.95

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Name', '\nSenegal - Facility Name'])

    # RUN 12: s=0.8, r=0.95, prior is Ozawa Africa countries with n>=150, with 5 times the variance
    priorMean = -3.5
    priorVar = 1.5
    s, r = 0.8, 0.95

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Name', '\nSenegal - Facility Name'])

    import numpy as np

    # RUN 13: s=0.8, r=0.95, prior is MQDB countries with at least 1 SFP; UNTRACKED
    priorMean = -2.5
    priorVar = 3.5
    s, r = 0.8, 0.95

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Name', '\nSenegal - Facility Name'])

    # RUN 14: s=0.8, r=0.95, prior is MQDB countries with at least 1 SFP, with 5 times the variance; UNTRACKED
    priorMean = -2.5
    priorVar = 1.5
    s, r = 0.8, 0.95

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Name', '\nSenegal - Facility Name'])

    # RUN 15: s=0.8, r=0.95, prior is Ozawa Africa studies w n>=150; UNTRACKED
    priorMean = -3.5
    priorVar = 3.5
    s, r = 0.8, 0.95

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Name', '\nSenegal - Facility Name'])

    # RUN 16: s=0.8, r=0.95, prior is MQDB countries with at least 1 SFP; UNTRACKED
    priorMean = -3.5
    priorVar = 1.5
    s, r = 0.8, 0.95

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Name', '\nSenegal - Facility Name'])

    # Rerun the 2010 data using untracked data; use N to estimate a sourcing probability matrix, Q
    import numpy as np
    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    util.plotPostSamples(lgDict, 'int90',
                         subTitleStr=['\nSenegal [Untracked] - Province', '\nSenegal [Untracked] - Province'])
    SNinds = lgDict['importerNames'].index('Aurobindo Pharmaceuticals Ltd')
    print('Aurobindo Pharmaceuticals Ltd: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    SNinds = lgDict['importerNames'].index('Macleods Pharmaceuticals Ltd')
    print('Macleods Pharmaceuticals Ltd: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    SNinds = lgDict['importerNames'].index('Lupin Limited')
    print('Lupin Limited: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    TNinds = lgDict['outletNames'].index('Dakar')
    print('Dakar: (' + str(np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.05))[
                       :5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.95))[:5] + ')')

    lgDict = util.testresultsfiletotable(tbl_SEN_G2_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # np.count_nonzero(Q)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal [Untracked] - Facility Location',
                                                       '\nSenegal [Untracked] - Facility Location'])

    lgDict = util.testresultsfiletotable(tbl_SEN_G3_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal [Untracked] - Facility Name',
                                                       '\nSenegal [Untracked] - Facility Name'])

    # Rerun the 2010 data using untracked data; use N to estimate a sourcing probability matrix, Q, and add a
    # "flattening" parameter to Q to make the sourcing probabilities less sharp
    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Add a constant across Q
    flatParam = 0.05
    Q = Q + flatParam
    for i, Qrow in enumerate(Q):
        Q[i] = Qrow / np.sum(Qrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    util.plotPostSamples(lgDict, 'int90',
                         subTitleStr=['\nSenegal [Untracked] - Province', '\nSenegal [Untracked] - Province'])

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Add a constant across Q
    flatParam = 0.01
    Q = Q + flatParam
    for i, Qrow in enumerate(Q):
        Q[i] = Qrow / np.sum(Qrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    util.plotPostSamples(lgDict, 'int90',
                         subTitleStr=['\nSenegal [Untracked] - Province', '\nSenegal [Untracked] - Province'])

    lgDict = util.testresultsfiletotable(tbl_SEN_G2_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Add a constant across Q
    flatParam = 0.02
    Q = Q + flatParam
    for i, Qrow in enumerate(Q):
        Q[i] = Qrow / np.sum(Qrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    util.plotPostSamples(lgDict, 'int90',
                         subTitleStr=['\nSenegal [Untracked] - Province', '\nSenegal [Untracked] - Province'])





    # Rerun 2010 data using different testing tool accuracy
    newSens, newSpec = 0.8, 0.95
    # newSens, newSpec = 0.6, 0.9

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': newSens, 'diagSpec': newSpec, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    subTitle = '\nSenegal [s=' + str(newSens) + ',r=' + str(newSpec) + '] - Province'
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[subTitle, subTitle])

    lgDict = util.testresultsfiletotable(tbl_SEN_G2_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': newSens, 'diagSpec': newSpec, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    subTitle = '\nSenegal [s=' + str(newSens) + ',r=' + str(newSpec) + '] - Facility Location'
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[subTitle, subTitle])

    lgDict = util.testresultsfiletotable(tbl_SEN_G3_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': newSens, 'diagSpec': newSpec, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    subTitle = '\nSenegal [s=' + str(newSens) + ',r=' + str(newSpec) + '] - Facility Name'
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[subTitle, subTitle])

    # Look at sensitivity to prior selection
    priorMean, priorVar = -0.7, 2  # 0.7,2 is from Ozawa Africa SFP studies with n>149 samples

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90',
                         subTitleStr=['\nSenegal [mu=' + str(priorMean) + ',var=' + str(priorVar) + '] - Province',
                                      '\nSenegal [mu=-2,var=1] - Province'])
    macleodInd = lgDict['importerNames'].index('Macleods Pharmaceuticals Ltd')
    np.quantile(lgDict['postSamples'][:, macleodInd], 0.05)
    np.quantile(lgDict['postSamples'][:, macleodInd], 0.95)
    np.quantile(lgDict['postSamples'][:, macleodInd], 0.05)
    np.quantile(lgDict['postSamples'][:, macleodInd], 0.95)

    priorMean, priorVar = -2., 5.

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90',
                         subTitleStr=['\nSenegal [mu=' + str(priorMean) + ',var=' + str(priorVar) + '] - Province',
                                      '\nSenegal [mu=-2,var=1] - Province'])

    priorMean, priorVar = -1., 5.

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90',
                         subTitleStr=['\nSenegal [mu=' + str(priorMean) + ',var=' + str(priorVar) + '] - Province',
                                      '\nSenegal [mu=-2,var=1] - Province'])

    priorMean, priorVar = -1., -1.

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90',
                         subTitleStr=['\nSenegal [mu=' + str(priorMean) + ',var=' + str(priorVar) + '] - Province',
                                      '\nSenegal [mu=-2,var=1] - Province'])

    # Use RUN 1 to explore the sufficiency of using different numbers of MCMC draws
    # RUN 1: s=1.0, r=1.0, prior is laplace(-2.5,3.5)
    priorMean = -2.5
    priorVar = 3.5

    # How much Madapt to use?
    numPostSamps = 1000
    sampMat = []
    for madapt in [100,500,1000,5000,10000]:
        for delt in [0.4,0.5,0.6]:
            MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': madapt, 'delta': delt}
            lgDict = util.testresultsfiletotable(tbl_SEN_G2_2010, csvName=False)
            lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
            lgDict = lg.runlogistigate(lgDict)
            sampMat.append(lgDict['postSamples'])

    # Trace plots along a few key nodes: Mftr. 5, Dist. 10; indices 20, 26
    plt.figure(figsize=(9,5))
    plt.plot(sampMat[0][:, 20],'b-.',linewidth=0.4, label='Manuf. 5; $M^{adapt}=100$')
    plt.plot(sampMat[12][:, 20],'b-.',linewidth=2.,alpha=0.2,label='Manuf. 5; $M^{adapt}=10000$')
    plt.plot(sampMat[0][:, 26], 'g--', linewidth=0.5, label='Dist. 1; $M^{adapt}=100$')
    plt.plot(sampMat[12][:, 26], 'g--', linewidth=2,alpha=0.3,label='Dist. 1; $M^{adapt}=10000$')
    plt.ylim([0,0.6])
    plt.xlabel('MCMC Draw',fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.ylabel('SFP rate',fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.legend()
    plt.title('Traces of MCMC draws for Manufacturer 5 and District 10\nUsing $M^{adapt}$ of 100 and 10000',fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.0%}'.format(x) for x in current_values])
    plt.tight_layout()
    plt.show()
    plt.close()

    # Now illustrate where the quantiles converge
    numPostSamps = 2000
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    plt.figure(figsize=(9, 5))
    for rep in range(20):
        lgDict = util.testresultsfiletotable(tbl_SEN_G2_2010, csvName=False)
        lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                       'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
        lgDict = lg.runlogistigate(lgDict)
        targSamps = lgDict['postSamples'][:, 20]  # Manufacturer 5
        cumQuant05 = [np.quantile(targSamps[:i + 1], 0.05) for i in range(len(targSamps))]
        cumQuant95 = [np.quantile(targSamps[:i + 1], 0.95) for i in range(len(targSamps))]
        plt.plot(cumQuant05, 'b-', linewidth=0.4)
        plt.plot(cumQuant95, 'b--', linewidth=0.4)
        targSamps = lgDict['postSamples'][:,26] # District 10
        cumQuant05 = [np.quantile(targSamps[:i + 1], 0.05) for i in range(len(targSamps))]
        cumQuant95 = [np.quantile(targSamps[:i + 1], 0.95) for i in range(len(targSamps))]
        plt.plot(cumQuant05,'g-',linewidth=0.4)
        plt.plot(cumQuant95,'g--',linewidth=0.4)
        print('Rep '+str(rep)+' done')

    plt.ylim([0, 0.48])
    plt.xlabel('MCMC Draw', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.ylabel('SFP rate', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.title('Traces of $5\%$ and $95\%$ quantiles of MCMC draws\nManufacturer 5 and District 10',
              fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.0%}'.format(x) for x in current_values])
    plt.tight_layout()
    plt.show()
    plt.close()




    return

def MQDdataScript():
    '''Script looking at the MQD data'''
    import scipy.special as sps
    import numpy as np
    import os
    SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
    filesPath = os.path.join(SCRIPT_DIR, 'MQDfiles')
    outputFileName = os.path.join(filesPath, 'pickleOutput')
    #sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, 'logistigate', 'exmples','data')))

    # Grab processed data tables
    '''
    dataDict = cleanMQD()
    '''
    import pickle
    openFile = open(outputFileName, 'rb')  # Read the file
    dataDict = pickle.load(openFile)
    '''
    DATA DICTIONARY KEYS:
        'df_ALL', 'df_CAM', 'df_ETH', 'df_GHA', 'df_KEN', 'df_LAO', 'df_MOZ', 'df_PER', 'df_PHI', 'df_SEN', 'df_THA', 
        'df_VIE', 'df_CAM_ff', 'df_ETH_ff', 'df_GHA_ff', 'df_KEN_ff', 'df_LAO_ff', 'df_MOZ_ff', 'df_PER_ff', 
        'df_PHI_ff', 'df_SEN_ff', 'df_THA_ff', 'df_VIE_ff', 
        'tbl_CAM_G1', 'tbl_CAM_G2', 'tbl_CAM_G3', 'tbl_ETH_G1', 'tbl_ETH_G2', 'tbl_ETH_G3', 
        'tbl_GHA_G1', 'tbl_GHA_G2', 'tbl_GHA_G3', 'tbl_KEN_G1', 'tbl_KEN_G2', 'tbl_KEN_G3', 
        'tbl_LAO_G1', 'tbl_LAO_G2', 'tbl_LAO_G3', 'tbl_MOZ_G1', 'tbl_MOZ_G2', 'tbl_MOZ_G3', 
        'tbl_PER_G1', 'tbl_PER_G2', 'tbl_PER_G3', 'tbl_PHI_G1', 'tbl_PHI_G2', 'tbl_PHI_G3', 
        'tbl_SEN_G1', 'tbl_SEN_G2', 'tbl_SEN_G3', 'tbl_THA_G1', 'tbl_THA_G2', 'tbl_THA_G3', 
        'tbl_VIE_G1', 'tbl_VIE_G2', 'tbl_VIE_G3', 
        'tbl_CAM_G1_ff', 'tbl_CAM_G2_ff', 'tbl_CAM_G3_ff', 'tbl_ETH_G1_ff', 'tbl_ETH_G2_ff', 'tbl_ETH_G3_ff', 
        'tbl_GHA_G1_ff', 'tbl_GHA_G2_ff', 'tbl_GHA_G3_ff', 'tbl_KEN_G1_ff', 'tbl_KEN_G2_ff', 'tbl_KEN_G3_ff', 
        'tbl_LAO_G1_ff', 'tbl_LAO_G2_ff', 'tbl_LAO_G3_ff', 'tbl_MOZ_G1_ff', 'tbl_MOZ_G2_ff', 'tbl_MOZ_G3_ff', 
        'tbl_PER_G1_ff', 'tbl_PER_G2_ff', 'tbl_PER_G3_ff', 'tbl_PHI_G1_ff', 'tbl_PHI_G2_ff', 'tbl_PHI_G3_ff', 
        'tbl_SEN_G1_ff', 'tbl_SEN_G2_ff', 'tbl_SEN_G3_ff', 'tbl_THA_G1_ff', 'tbl_THA_G2_ff', 'tbl_THA_G3_ff', 
        'tbl_VIE_G1_ff', 'tbl_VIE_G2_ff', 'tbl_VIE_G3_ff', 
        'tbl_CAM_G1_antibiotic', 'tbl_CAM_G2_antibiotic', 'tbl_CAM_G3_antibiotic', 
        'tbl_CAM_G1_antimalarial', 'tbl_CAM_G2_antimalarial', 'tbl_CAM_G3_antimalarial', 
        'tbl_ETH_G1_antibiotic', 'tbl_ETH_G2_antibiotic', 'tbl_ETH_G3_antibiotic', 
        'tbl_ETH_G1_antimalarial', 'tbl_ETH_G2_antimalarial', 'tbl_ETH_G3_antimalarial', 
        'tbl_GHA_G1_antimalarial', 'tbl_GHA_G2_antimalarial', 'tbl_GHA_G3_antimalarial', 
        'tbl_KEN_G1_antimalarial', 'tbl_KEN_G2_antimalarial', 'tbl_KEN_G3_antimalarial',
        'tbl_LAO_G1_antibiotic', 'tbl_LAO_G2_antibiotic', 'tbl_LAO_G3_antibiotic', 
        'tbl_LAO_G1_antimalarial', 'tbl_LAO_G2_antimalarial', 'tbl_LAO_G3_antimalarial', 
        'tbl_MOZ_G1_antibiotic', 'tbl_MOZ_G2_antibiotic', 'tbl_MOZ_G3_antibiotic', 
        'tbl_MOZ_G1_antimalarial', 'tbl_MOZ_G2_antimalarial', 'tbl_MOZ_G3_antimalarial', 
        'tbl_PER_G1_antibiotic', 'tbl_PER_G2_antibiotic', 'tbl_PER_G3_antibiotic', 
        'tbl_PHI_G1_antituberculosis', 'tbl_PHI_G2_antituberculosis', 'tbl_PHI_G3_antituberculosis', 
        'tbl_SEN_G1_antimalarial', 'tbl_SEN_G2_antimalarial', 'tbl_SEN_G3_antimalarial', 
        'tbl_SEN_G1_antiretroviral', 'tbl_SEN_G2_antiretroviral', 'tbl_SEN_G3_antiretroviral', 
        'tbl_THA_G1_antibiotic', 'tbl_THA_G2_antibiotic', 'tbl_THA_G3_antibiotic', 
        'tbl_THA_G1_antimalarial', 'tbl_THA_G2_antimalarial', 'tbl_THA_G3_antimalarial', 
        'tbl_VIE_G1_antibiotic', 'tbl_VIE_G2_antibiotic', 'tbl_VIE_G3_antibiotic', 
        'tbl_VIE_G1_antimalarial', 'tbl_VIE_G2_antimalarial', 'tbl_VIE_G3_antimalarial', 
        'tbl_CAM_G1_antibiotic_ff', 'tbl_CAM_G2_antibiotic_ff', 'tbl_CAM_G3_antibiotic_ff', 
        'tbl_CAM_G1_antimalarial_ff', 'tbl_CAM_G2_antimalarial_ff', 'tbl_CAM_G3_antimalarial_ff', 
        'tbl_ETH_G1_antibiotic_ff', 'tbl_ETH_G2_antibiotic_ff', 'tbl_ETH_G3_antibiotic_ff', 
        'tbl_ETH_G1_antimalarial_ff', 'tbl_ETH_G2_antimalarial_ff', 'tbl_ETH_G3_antimalarial_ff', 
        'tbl_GHA_G1_antimalarial_ff', 'tbl_GHA_G2_antimalarial_ff', 'tbl_GHA_G3_antimalarial_ff', 
        'tbl_KEN_G1_antimalarial_ff', 'tbl_KEN_G2_antimalarial_ff', 'tbl_KEN_G3_antimalarial_ff', 
        'tbl_LAO_G1_antibiotic_ff', 'tbl_LAO_G2_antibiotic_ff', 'tbl_LAO_G3_antibiotic_ff', 
        'tbl_LAO_G1_antimalarial_ff', 'tbl_LAO_G2_antimalarial_ff', 'tbl_LAO_G3_antimalarial_ff', 
        'tbl_MOZ_G1_antibiotic_ff', 'tbl_MOZ_G2_antibiotic_ff', 'tbl_MOZ_G3_antibiotic_ff', 
        'tbl_MOZ_G1_antimalarial_ff', 'tbl_MOZ_G2_antimalarial_ff', 'tbl_MOZ_G3_antimalarial_ff', 
        'tbl_PER_G1_antibiotic_ff', 'tbl_PER_G2_antibiotic_ff', 'tbl_PER_G3_antibiotic_ff', 
        'tbl_PHI_G1_antituberculosis_ff', 'tbl_PHI_G2_antituberculosis_ff', 'tbl_PHI_G3_antituberculosis_ff', 
        'tbl_SEN_G1_antimalarial_ff', 'tbl_SEN_G2_antimalarial_ff', 'tbl_SEN_G3_antimalarial_ff', 
        'tbl_SEN_G1_antiretroviral_ff', 'tbl_SEN_G2_antiretroviral_ff', 'tbl_SEN_G3_antiretroviral_ff', 
        'tbl_THA_G1_antibiotic_ff', 'tbl_THA_G2_antibiotic_ff', 'tbl_THA_G3_antibiotic_ff', 
        'tbl_THA_G1_antimalarial_ff', 'tbl_THA_G2_antimalarial_ff', 'tbl_THA_G3_antimalarial_ff', 
        'tbl_VIE_G1_antibiotic_ff', 'tbl_VIE_G2_antibiotic_ff', 'tbl_VIE_G3_antibiotic_ff', 
        'tbl_VIE_G1_antimalarial_ff', 'tbl_VIE_G2_antimalarial_ff', 'tbl_VIE_G3_antimalarial_ff'
    '''

    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    meanSFPrate = dataDict['df_ALL'][dataDict['df_ALL']['Final_Test_Conclusion']=='Fail']['Sample_ID'].count() / dataDict['df_ALL']['Sample_ID'].count()
    priorMean = sps.logit(meanSFPrate) # Mean SFP rate of the MQDB data
    # priorMean = sps.logit(np.sum(dataTblDict_CAM['Y']) / np.sum(dataTblDict_CAM['N']))

    ##### CAMBODIA #####
    lgDict = util.testresultsfiletotable(dataDict['tbl_CAM_G1'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0,  'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict,'int90',subTitleStr=[' Cambodia',' Cambodia'])
    # Break up the SNs
    sampMedians = [np.median(lgDict['postSamples'][:, i]) for i in range(lgDict['importerNum'] + lgDict['outletNum'])]
    highSNInds = [i for i, x in enumerate(sampMedians[:lgDict['importerNum']]) if x > 0.1]
    util.plotPostSamples(lgDict, 'int90', importerIndsSubset=highSNInds, subTitleStr=[' Cambodia', ' Cambodia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_CAM_G2'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Cambodia', ' Cambodia'])
    # Break up the SNs
    sampMedians = [np.median(lgDict['postSamples'][:, i]) for i in range(lgDict['importerNum'] + lgDict['outletNum'])]
    highSNInds = [i for i, x in enumerate(sampMedians[:lgDict['importerNum']]) if x > 0.1]
    util.plotPostSamples(lgDict, 'int90', importerIndsSubset=highSNInds, subTitleStr=[' Cambodia', ' Cambodia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_CAM_G3'], csvName=False)
    print('size: '+str(lgDict['N'].shape)+', obsvns: '+str(lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Cambodia', ' Cambodia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_CAM_G1_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500, 'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Cambodia', ' Cambodia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_CAM_G2_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500, 'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Cambodia', ' Cambodia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_CAM_G3_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500, 'prior': methods.prior_normal(mu=priorMean),'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Cambodia', ' Cambodia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_CAM_G1_antibiotic'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500, 'prior': methods.prior_normal(mu=priorMean),
                   'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Cambodia', ' Cambodia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_CAM_G2_antibiotic'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500, 'prior': methods.prior_normal(mu=priorMean),
                   'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Cambodia', ' Cambodia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_CAM_G3_antibiotic'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500, 'prior': methods.prior_normal(mu=priorMean),
                   'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Cambodia', ' Cambodia'])
    # Break up the TNs
    sampMedians = [np.median(lgDict['postSamples'][:, i]) for i in range(lgDict['importerNum'] + lgDict['outletNum'])]
    highTNInds = [i for i, x in enumerate(sampMedians[lgDict['importerNum']:]) if x > 0.1]
    util.plotPostSamples(lgDict, 'int90', outletIndsSubset=highTNInds, subTitleStr=[' Cambodia', ' Cambodia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_CAM_G1_antimalarial'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500, 'prior': methods.prior_normal(mu=priorMean),
                   'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Cambodia', ' Cambodia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_CAM_G2_antimalarial'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500, 'prior': methods.prior_normal(mu=priorMean),
                   'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Cambodia', ' Cambodia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_CAM_G3_antimalarial'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500, 'prior': methods.prior_normal(mu=priorMean),
                   'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Cambodia', ' Cambodia'])
    # Break up the TNs
    sampMedians = [np.median(lgDict['postSamples'][:, i]) for i in range(lgDict['importerNum'] + lgDict['outletNum'])]
    highTNInds = [i for i, x in enumerate(sampMedians[lgDict['importerNum']:]) if x > 0.1]
    util.plotPostSamples(lgDict, 'int90', outletIndsSubset=highTNInds, subTitleStr=[' Cambodia', ' Cambodia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_CAM_G1_antibiotic_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500, 'prior': methods.prior_normal(mu=priorMean),
                   'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Cambodia', ' Cambodia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_CAM_G2_antibiotic_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500, 'prior': methods.prior_normal(mu=priorMean),
                   'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Cambodia', ' Cambodia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_CAM_G3_antibiotic_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500, 'prior': methods.prior_normal(mu=priorMean),
                   'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Cambodia', ' Cambodia'])
    # Break up the TNs
    sampMedians = [np.median(lgDict['postSamples'][:, i]) for i in range(lgDict['importerNum'] + lgDict['outletNum'])]
    highTNInds = [i for i, x in enumerate(sampMedians[lgDict['importerNum']:]) if x > 0.1]
    util.plotPostSamples(lgDict, 'int90', outletIndsSubset=highTNInds, subTitleStr=[' Cambodia', ' Cambodia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_CAM_G1_antimalarial_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500, 'prior': methods.prior_normal(mu=priorMean),
                   'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Cambodia', ' Cambodia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_CAM_G2_antimalarial_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500, 'prior': methods.prior_normal(mu=priorMean),
                   'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Cambodia', ' Cambodia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_CAM_G3_antimalarial_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500, 'prior': methods.prior_normal(mu=priorMean),
                   'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Cambodia', ' Cambodia'])
    # Break up the TNs
    sampMedians = [np.median(lgDict['postSamples'][:, i]) for i in range(lgDict['importerNum'] + lgDict['outletNum'])]
    highTNInds = [i for i, x in enumerate(sampMedians[lgDict['importerNum']:]) if x > 0.1]
    util.plotPostSamples(lgDict, 'int90', outletIndsSubset=highTNInds, subTitleStr=[' Cambodia', ' Cambodia'])

    import pandas as pd
    CAM_df = dataDict['df_CAM']
    CAM_df['Date_Received_format'] = pd.to_datetime(CAM_df['Date_Received'], format='%m/%d/%Y')
    CAM_df_2006 = CAM_df[CAM_df['Date_Received_format'] >= '2006-01-01']
    CAM_df_2003 = CAM_df[CAM_df['Date_Received_format'] < '2006-01-01']

    tbl_CAM_G1_2006 = CAM_df_2006[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    tbl_CAM_G1_2006 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in tbl_CAM_G1_2006]
    tbl_CAM_G2_2006 = CAM_df_2006[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    tbl_CAM_G2_2006 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in tbl_CAM_G2_2006]
    tbl_CAM_G3_2006 = CAM_df_2006[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    tbl_CAM_G3_2006 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in tbl_CAM_G3_2006]

    lgDict = util.testresultsfiletotable(tbl_CAM_G1_2006, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500, 'prior': methods.prior_normal(mu=priorMean),
                   'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Cambodia', ' Cambodia'])

    lgDict = util.testresultsfiletotable(tbl_CAM_G2_2006, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500, 'prior': methods.prior_normal(mu=priorMean),
                   'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Cambodia', ' Cambodia'])

    lgDict = util.testresultsfiletotable(tbl_CAM_G3_2006, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500, 'prior': methods.prior_normal(mu=priorMean),
                   'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Cambodia', ' Cambodia'])

    tbl_CAM_G1_2003 = CAM_df_2003[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    tbl_CAM_G1_2003 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in tbl_CAM_G1_2003]
    tbl_CAM_G2_2003 = CAM_df_2003[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    tbl_CAM_G2_2003 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in tbl_CAM_G2_2003]
    tbl_CAM_G3_2003 = CAM_df_2003[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    tbl_CAM_G3_2003 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in tbl_CAM_G3_2003]

    lgDict = util.testresultsfiletotable(tbl_CAM_G1_2003, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500, 'prior': methods.prior_normal(mu=priorMean),
                   'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Cambodia', ' Cambodia'])

    lgDict = util.testresultsfiletotable(tbl_CAM_G2_2003, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500, 'prior': methods.prior_normal(mu=priorMean),
                   'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Cambodia', ' Cambodia'])

    lgDict = util.testresultsfiletotable(tbl_CAM_G3_2003, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500, 'prior': methods.prior_normal(mu=priorMean),
                   'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Cambodia', ' Cambodia'])
    ##### END CAMBODIA #####

    ##### ETHIOPIA #####
    lgDict = util.testresultsfiletotable(dataDict['tbl_ETH_G1'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ethiopia', ' Ethiopia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_ETH_G2'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ethiopia', ' Ethiopia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_ETH_G3'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ethiopia', ' Ethiopia'])
    # Break up the TNs
    sampMedians = [np.median(lgDict['postSamples'][:, i]) for i in range(lgDict['importerNum'] + lgDict['outletNum'])]
    highTNInds = [i for i, x in enumerate(sampMedians[lgDict['importerNum']:]) if x > 0.1]
    util.plotPostSamples(lgDict, 'int90', outletIndsSubset=highTNInds, subTitleStr=[' Cambodia', ' Cambodia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_ETH_G1_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ethiopia', ' Ethiopia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_ETH_G2_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ethiopia', ' Ethiopia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_ETH_G3_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ethiopia', ' Ethiopia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_ETH_G1_antibiotic'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ethiopia', ' Ethiopia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_ETH_G2_antibiotic'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ethiopia', ' Ethiopia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_ETH_G3_antibiotic'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ethiopia', ' Ethiopia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_ETH_G1_antimalarial'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ethiopia', ' Ethiopia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_ETH_G2_antimalarial'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ethiopia', ' Ethiopia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_ETH_G3_antimalarial'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ethiopia', ' Ethiopia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_ETH_G1_antibiotic_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ethiopia', ' Ethiopia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_ETH_G2_antibiotic_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ethiopia', ' Ethiopia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_ETH_G3_antibiotic_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ethiopia', ' Ethiopia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_ETH_G1_antimalarial_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ethiopia', ' Ethiopia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_ETH_G2_antimalarial_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ethiopia', ' Ethiopia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_ETH_G3_antimalarial_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ethiopia', ' Ethiopia'])

    ##### END ETHIOPIA #####

    ##### GHANA #####
    lgDict = util.testresultsfiletotable(dataDict['tbl_GHA_G1'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean,var=1.1), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ghana', ' Ghana'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_GHA_G2'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ghana', ' Ghana'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_GHA_G3'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ghana', ' Ghana'])
    # Break up TNs
    sampMedians = [np.median(lgDict['postSamples'][:, i]) for i in range(lgDict['importerNum'] + lgDict['outletNum'])]
    highTNInds = [i for i, x in enumerate(sampMedians[lgDict['importerNum']:]) if x > 0.1]
    util.plotPostSamples(lgDict, 'int90', outletIndsSubset=highTNInds, subTitleStr=[' Cambodia', ' Cambodia'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_GHA_G1_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ghana', ' Ghana'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_GHA_G2_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ghana', ' Ghana'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_GHA_G3_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ghana', ' Ghana'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_GHA_G1_antimalarial'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ghana', ' Ghana'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_GHA_G2_antimalarial'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ghana', ' Ghana'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_GHA_G3_antimalarial'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ghana', ' Ghana'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_GHA_G1_antimalarial_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ghana', ' Ghana'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_GHA_G2_antimalarial_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ghana', ' Ghana'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_GHA_G3_antimalarial_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Ghana', ' Ghana'])

    ##### END GHANA #####

    ##### KENYA #####
    lgDict = util.testresultsfiletotable(dataDict['tbl_KEN_G1'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Kenya', ' Kenya'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_KEN_G2'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Kenya', ' Kenya'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_KEN_G3'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Kenya', ' Kenya'])
    # Break up TNs
    sampMedians = [np.median(lgDict['postSamples'][:, i]) for i in range(lgDict['importerNum'] + lgDict['outletNum'])]
    highTNInds = [i for i, x in enumerate(sampMedians[lgDict['importerNum']:]) if x > 0.1]
    util.plotPostSamples(lgDict, 'int90', outletIndsSubset=highTNInds, subTitleStr=[' Kenya', ' Kenya'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_KEN_G1_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Kenya', ' Kenya'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_KEN_G2_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Kenya', ' Kenya'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_KEN_G3_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Kenya', ' Kenya'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_KEN_G1_antimalarial'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Kenya', ' Kenya'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_KEN_G2_antimalarial'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Kenya', ' Kenya'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_KEN_G3_antimalarial'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Kenya', ' Kenya'])
    # Break up TNs
    sampMedians = [np.median(lgDict['postSamples'][:, i]) for i in range(lgDict['importerNum'] + lgDict['outletNum'])]
    highTNInds = [i for i, x in enumerate(sampMedians[lgDict['importerNum']:]) if x > 0.1]
    util.plotPostSamples(lgDict, 'int90', outletIndsSubset=highTNInds, subTitleStr=[' Kenya', ' Kenya'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_KEN_G1_antimalarial_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Kenya', ' Kenya'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_KEN_G2_antimalarial_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Kenya', ' Kenya'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_KEN_G3_antimalarial_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Kenya', ' Kenya'])

    ##### END KENYA #####

    ##### LAOS #####
    lgDict = util.testresultsfiletotable(dataDict['tbl_LAO_G1'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Laos', ' Laos'])
    # Break up SNs
    sampMedians = [np.median(lgDict['postSamples'][:, i]) for i in range(lgDict['importerNum'] + lgDict['outletNum'])]
    highSNInds = [i for i, x in enumerate(sampMedians[:lgDict['importerNum']]) if x > 0.1]
    util.plotPostSamples(lgDict, 'int90', importerIndsSubset=highSNInds, subTitleStr=[' Laos', ' Laos'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_LAO_G2'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Laos', ' Laos'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_LAO_G3'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Laos', ' Laos'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_LAO_G1_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Laos', ' Laos'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_LAO_G2_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Laos', ' Laos'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_LAO_G3_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Laos', ' Laos'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_LAO_G1_antibiotic'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Laos', ' Laos'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_LAO_G2_antibiotic'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Laos', ' Laos'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_LAO_G3_antibiotic'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Laos', ' Laos'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_LAO_G1_antimalarial'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Laos', ' Laos'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_LAO_G2_antimalarial'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Laos', ' Laos'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_LAO_G3_antimalarial'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Laos', ' Laos'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_LAO_G1_antibiotic_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Laos', ' Laos'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_LAO_G2_antibiotic_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Laos', ' Laos'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_LAO_G3_antibiotic_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Laos', ' Laos'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_LAO_G1_antimalarial_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Laos', ' Laos'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_LAO_G2_antimalarial_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Laos', ' Laos'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_LAO_G3_antimalarial_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Laos', ' Laos'])
    ##### END LAOS #####

    ##### MOZAMBIQUE #####
    lgDict = util.testresultsfiletotable(dataDict['tbl_MOZ_G1'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Mozambique', ' Mozambique'])
    # Break up SNs
    sampMedians = [np.median(lgDict['postSamples'][:, i]) for i in range(lgDict['importerNum'] + lgDict['outletNum'])]
    highSNInds = [i for i, x in enumerate(sampMedians[:lgDict['importerNum']]) if x > 0.1]
    util.plotPostSamples(lgDict, 'int90', importerIndsSubset=highSNInds, subTitleStr=[' Mozambique', ' Mozambique'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_MOZ_G2'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Mozambique', ' Mozambique'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_MOZ_G3'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Mozambique', ' Mozambique'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_MOZ_G1_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Mozambique', ' Mozambique'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_MOZ_G2_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Mozambique', ' Mozambique'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_MOZ_G3_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Mozambique', ' Mozambique'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_MOZ_G1_antibiotic'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Mozambique', ' Mozambique'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_MOZ_G2_antibiotic'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Mozambique', ' Mozambique'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_MOZ_G3_antibiotic'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Mozambique', ' Mozambique'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_MOZ_G1_antimalarial'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Mozambique', ' Mozambique'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_MOZ_G2_antimalarial'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Mozambique', ' Mozambique'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_MOZ_G3_antimalarial'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Mozambique', ' Mozambique'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_MOZ_G1_antibiotic_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Mozambique', ' Mozambique'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_MOZ_G2_antibiotic_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Mozambique', ' Mozambique'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_MOZ_G3_antibiotic_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Mozambique', ' Mozambique'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_MOZ_G1_antimalarial_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Mozambique', ' Mozambique'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_MOZ_G2_antimalarial_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Mozambique', ' Mozambique'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_MOZ_G3_antimalarial_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Mozambique', ' Mozambique'])
    ##### END MOZAMBIQUE #####

    ##### PERU #####
    lgDict = util.testresultsfiletotable(dataDict['tbl_PER_G1'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Peru', ' Peru'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_PER_G2'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Peru', ' Peru'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_PER_G3'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Peru', ' Peru'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_PER_G1_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Peru', ' Peru'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_PER_G2_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Peru', ' Peru'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_PER_G3_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Peru', ' Peru'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_PER_G1_antibiotic'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Peru', ' Peru'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_PER_G2_antibiotic'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Peru', ' Peru'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_PER_G3_antibiotic'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Peru', ' Peru'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_PER_G1_antibiotic_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Peru', ' Peru'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_PER_G2_antibiotic_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Peru', ' Peru'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_PER_G3_antibiotic_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[' Peru', ' Peru'])
    ##### END PERU #####

    ##### PHILIPPINES #####
    lgDict = util.testresultsfiletotable(dataDict['tbl_PHI_G1'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Philippines', ', Philippines'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_PHI_G2'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Philippines', ', Philippines'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_PHI_G3'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Philippines', ', Philippines'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_PHI_G1_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Philippines', ', Philippines'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_PHI_G2_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Philippines', ', Philippines'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_PHI_G3_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Philippines', ', Philippines'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_PHI_G1_antituberculosis'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Philippines', ', Philippines'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_PHI_G2_antituberculosis'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Philippines', ', Philippines'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_PHI_G3_antituberculosis'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Philippines', ', Philippines'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_PHI_G1_antituberculosis_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Philippines', ', Philippines'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_PHI_G2_antituberculosis_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Philippines', ', Philippines'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_PHI_G3_antituberculosis_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Philippines', ', Philippines'])

    ##### END PHILIPPINES #####

    ##### SENEGAL #####
    lgDict = util.testresultsfiletotable(dataDict['tbl_SEN_G1'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Senegal', ', Senegal'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_SEN_G2'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Senegal', ', Senegal'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_SEN_G3'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Senegal', ', Senegal'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_SEN_G1_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Senegal', ', Senegal'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_SEN_G2_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Senegal', ', Senegal'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_SEN_G3_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Senegal', ', Senegal'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_SEN_G1_antimalarial'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Senegal', ', Senegal'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_SEN_G2_antimalarial'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Senegal', ', Senegal'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_SEN_G3_antimalarial'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Senegal', ', Senegal'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_SEN_G1_antiretroviral'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Senegal', ', Senegal'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_SEN_G2_antiretroviral'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Senegal', ', Senegal'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_SEN_G3_antiretroviral'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Senegal', ', Senegal'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_SEN_G1_antimalarial_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Senegal', ', Senegal'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_SEN_G2_antimalarial_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Senegal', ', Senegal'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_SEN_G3_antimalarial_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Senegal', ', Senegal'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_SEN_G1_antiretroviral_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Senegal', ', Senegal'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_SEN_G2_antiretroviral_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Senegal', ', Senegal'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_SEN_G3_antiretroviral_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Senegal', ', Senegal'])

    #########################   #########################   #########################
    #####                           2010 SENEGAL DATA                           #####
    #########################   #########################   #########################
    # Breaking Senegal data into 2009 and 2010 data
    #import pandas as pd
    SEN_df = dataDict['df_SEN']
    SEN_df_2009 = SEN_df[SEN_df['Date_Received'] == '6/1/2009']
    SEN_df_2010 = SEN_df[SEN_df['Date_Received'] == '7/12/2010']

    tbl_SEN_G1_2009 = SEN_df_2009[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    tbl_SEN_G1_2009 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in tbl_SEN_G1_2009]
    tbl_SEN_G2_2009 = SEN_df_2009[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    tbl_SEN_G2_2009 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in tbl_SEN_G2_2009]
    tbl_SEN_G3_2009 = SEN_df_2009[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    tbl_SEN_G3_2009 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in tbl_SEN_G3_2009]

    tbl_SEN_G1_2010 = SEN_df_2010[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    tbl_SEN_G1_2010 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in tbl_SEN_G1_2010]
    tbl_SEN_G2_2010 = SEN_df_2010[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    tbl_SEN_G2_2010 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in tbl_SEN_G2_2010]
    tbl_SEN_G3_2010 = SEN_df_2010[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    tbl_SEN_G3_2010 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in tbl_SEN_G3_2010]

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2009, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Senegal', ', Senegal'])

    lgDict = util.testresultsfiletotable(tbl_SEN_G2_2009, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Senegal', ', Senegal'])

    lgDict = util.testresultsfiletotable(tbl_SEN_G3_2009, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Senegal', ', Senegal'])

    #########################   #########################   #########################
    #####                           2010 SENEGAL DATA                           #####
    #########################   #########################   #########################
    numPostSamps = 1000
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    meanSFPrate = dataDict['df_ALL'][dataDict['df_ALL']['Final_Test_Conclusion'] == 'Fail']['Sample_ID'].count() / \
                  dataDict['df_ALL']['Sample_ID'].count()
    priorMean = sps.logit(meanSFPrate)  # Mean SFP rate of the MQDB data

    priorVar = 1.416197468 # 5 times the variance of the Ozawa Africa rates

    # Use moments of distribution of Ozawa Africa rates
    #priorMean = -0.718837654
    #priorVar = 0.283239494
    # priorVar = 1.416197468 # 5 times the variance
    # Use SFPs from MQDB countries with at least 1 SFP
    # priorMean = -1.338762078
    # priorVar = 0.228190396
    # priorVar = 1.14095198; 5 times the variance

    SEN_df = dataDict['df_SEN']
    # 7 unique Province_Name_GROUPED; 23 unique Facility_Location_GROUPED; 66 unique Facility_Name_GROUPED
    # Remove 'Missing' and 'Unknown' labels
    SEN_df_2010 = SEN_df[(SEN_df['Date_Received'] == '7/12/2010') & (SEN_df['Manufacturer_GROUPED'] != 'Unknown') & (SEN_df['Facility_Location_GROUPED'] != 'Missing')].copy()

    '''
    SEN_df_2010.loc[(SEN_df_2010.Facility_Location_GROUPED == 'Médina')
                    | (SEN_df_2010.Facility_Location_GROUPED == 'Medina'),
                    'Facility_Location_GROUPED'] = 'Medina'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Location_GROUPED == 'Mbour-Thiès')
                    | (SEN_df_2010.Facility_Location_GROUPED == 'Mbour-Thies'),
                    'Facility_Location_GROUPED'] = 'Mbour-Thies'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Location_GROUPED == 'saint louis')
                    | (SEN_df_2010.Facility_Location_GROUPED == 'Sor Saint Louis')
                    | (SEN_df_2010.Facility_Location_GROUPED == 'pharmacie Mame Madia')
                    | (SEN_df_2010.Facility_Location_GROUPED == 'Saint Louis (Dept)'),
                    'Facility_Location_GROUPED'] = 'Saint Louis (Dept)'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Location_GROUPED == 'velingara TÃ©l : 33 997 - 11- 10')
                    | (SEN_df_2010.Facility_Location_GROUPED == 'Velingara'),
                    'Facility_Location_GROUPED'] = 'Velingara'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Location_GROUPED == 'koumpantoum')
                    | (SEN_df_2010.Facility_Location_GROUPED == 'Koumpantoum'),
                    'Facility_Location_GROUPED'] = 'Koumpantoum'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Location_GROUPED == 'Dr Dame SECK avenue J.F.KENNEDY B.P.157 tel/FAX941.17.11')
                    | (SEN_df_2010.Facility_Location_GROUPED == 'Kaolack (City)'),
                    'Facility_Location_GROUPED'] = 'Kaolack (City)'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Location_GROUPED == '150m hpt kaff')
                    | (SEN_df_2010.Facility_Location_GROUPED == 'Kaffrine (City)'),
                    'Facility_Location_GROUPED'] = 'Kaffrine (City)'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Location_GROUPED == 'kebemer')
                    | (SEN_df_2010.Facility_Location_GROUPED == 'Kebemer'),
                    'Facility_Location_GROUPED'] = 'Kebemer'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Location_GROUPED == 'Kaolack (City)')
                    | (SEN_df_2010.Facility_Location_GROUPED == 'Dr Assane TOURE tel 33.941.28.29 BP.53 Email boubakh@orange.sn'),
                    'Facility_Location_GROUPED'] = 'Kaolack (City)'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Location_GROUPED == 'Thiès')
                    | (SEN_df_2010.Facility_Location_GROUPED == 'Thies'),
                    'Facility_Location_GROUPED'] = 'Thies'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Location_GROUPED == 'BP 60 TAMBACOUNDA SENEGAL')
                    | (SEN_df_2010.Facility_Location_GROUPED == 'Quartier Abattoir Tambacounda')
                    | (SEN_df_2010.Facility_Location_GROUPED == 'Quartier Pout, Avenue Léopold S, Senghor Tambacounda')
                    | (SEN_df_2010.Facility_Location_GROUPED == 'Tambacounda'),
                    'Facility_Location_GROUPED'] = 'Tambacounda'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Location_GROUPED == 'Quartier: Sare Moussa Tél. 33 996 23 47')
                    | (SEN_df_2010.Facility_Location_GROUPED == 'Quartier Sikilo, route de Tripano, Kolda')
                    | (SEN_df_2010.Facility_Location_GROUPED == 'Quartier Centre II Tél : 33 997-11-58')
                    | (SEN_df_2010.Facility_Location_GROUPED == 'Kolda')
                    | (SEN_df_2010.Facility_Location_GROUPED == 'Kolda (City)'),
                    'Facility_Location_GROUPED'] = 'Kolda (City)'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Location_GROUPED == 'Quartier Caba Club Kedougou')
                    | (SEN_df_2010.Facility_Location_GROUPED == 'Quartier Gomba n° 630 Kedougou')
                    | (SEN_df_2010.Facility_Location_GROUPED == 'Kedougou')
                    | (SEN_df_2010.Facility_Location_GROUPED == 'Kedougou (City)'),
                    'Facility_Location_GROUPED'] = 'Kedougou (City)'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Location_GROUPED == 'Matam')
                    | (SEN_df_2010.Facility_Location_GROUPED == 'Matam (City)'),
                    'Facility_Location_GROUPED'] = 'Matam (City)'
    
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'hopital rÃƒÂ©gional de saint louis')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Hopital Regional de Saint Louis'),
                    'Facility_Name_GROUPED'] = 'Hopital Regional de Saint Louis'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'Centre de Santé Diourbel')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Centre de Sante Diourbel'),
                    'Facility_Name_GROUPED'] = 'Centre de Sante Diourbel'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'Hopital de Dioum')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Hopital de DIOUM'),
                    'Facility_Name_GROUPED'] = 'Hopital de Dioum'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == "Pharmacie RÃƒÂ©gionale d' Approvisionnement de Saint Louis")
                    | (SEN_df_2010.Facility_Name_GROUPED == "Pharmacie Regionale d' Approvisionnement de Saint Louis"),
                    'Facility_Name_GROUPED'] = "Pharmacie Regionale d' Approvisionnement de Saint Louis"
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'centre de sante de kolda')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Centre de Sante de Kolda'),
                    'Facility_Name_GROUPED'] = 'Centre de Sante de Kolda'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'centre de sante de velingara')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Centre de Sante de Velingara'),
                    'Facility_Name_GROUPED'] = 'Centre de Sante de Velingara'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'Hopitale regionale de Tambacounda')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Hopitale Regionale de Tambacounda'),
                    'Facility_Name_GROUPED'] = 'Hopitale Regionale de Tambacounda'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'centre de santÃ© de koumpantoum')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Centre de Sante de Koumpantoum'),
                    'Facility_Name_GROUPED'] = 'Centre de Sante de Koumpantoum'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'PHARMACIE MAME MADIA')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Pharmacie Mame Madia'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Mame Madia'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'pharmacie')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Pharmacie'),
                    'Facility_Name_GROUPED'] = 'Pharmacie'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'Centre de Santé Mbacké')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Centre de Sante Mbacke'),
                    'Facility_Name_GROUPED'] = 'Centre de Sante Mbacke'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'Pharmacie Ndamatou  Dr Omar Niasse tel : 33978-17-68Touba')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Pharmacie Ndamatou Dr O.N.'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Ndamatou Dr O.N.'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'Pharmacie centrale  Dr A. Camara tel : 33971-11-20 Diourbel')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Pharmacie Centrale Dr A.C.'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Centrale Dr A.C.'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'District Sanitaire Touba tel: 33-978-13-70')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'District Sanitaire Touba'),
                    'Facility_Name_GROUPED'] = 'District Sanitaire Touba'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'pharmacie ousmane')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Pharmacie Ousmane'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Ousmane'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'Centre de santé Ousmane Ngom')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Centre de Sante Ousmane Ngom'),
                    'Facility_Name_GROUPED'] = 'Centre de Sante Ousmane Ngom'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'centre de santé de matam')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Centre de Sante de Matam'),
                    'Facility_Name_GROUPED'] = 'Centre de Sante de Matam'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'pharmacie du Fleuve')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Pharmacie du Fleuve'),
                    'Facility_Name_GROUPED'] = 'Pharmacie du Fleuve'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'Centre de Santé de Richard Toll')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Centre de Sante de Richard Toll'),
                    'Facility_Name_GROUPED'] = 'Centre de Sante de Richard Toll'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'pharmacie boubakh')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Pharmacie Boubakh'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Boubakh'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'centre de santÃ© de dioum')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Centre de Sante de Dioum'),
                    'Facility_Name_GROUPED'] = 'Centre de Sante de Dioum'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'centre de santÃ© de kanel')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Centre de Sante de Kanel'),
                    'Facility_Name_GROUPED'] = 'Centre de Sante de Kanel'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'hôpital régionale de ouro-sogui')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Hopital Regionale de Ouro-Sogui'),
                    'Facility_Name_GROUPED'] = 'Hopital Regionale de Ouro-Sogui'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'Centre de traitement de la tuberculose d eTouba  tel : 33978-13-71')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Centre de Traitement de la Tuberculose de Touba'),
                    'Facility_Name_GROUPED'] = 'Centre de Traitement de la Tuberculose de Touba'
    SEN_df_2010.loc[
        (SEN_df_2010.Facility_Name_GROUPED == 'Pharmacie Mame  Ibrahima Ndour Dr Alassane Ndour Tél: 339760097 Mbacké')
        | (SEN_df_2010.Facility_Name_GROUPED == 'Pharmacie Mame Ibrahima Ndour Dr A.N.'),
        'Facility_Name_GROUPED'] = 'Pharmacie Mame Ibrahima Ndour Dr A.N.'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'hopitale regionale de koda')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Hopitale Regionale de Koda'),
                    'Facility_Name_GROUPED'] = 'Hopitale Regionale de Koda'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'Pharmacie Touba MosquÃ©e  Dr Amadou Malick Kane Tel : 33974-89-74')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Pharmacie Touba Mosque Dr A.M.K.'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Touba Mosque Dr A.M.K.'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == "pharmacie Château d'Eau")
                    | (SEN_df_2010.Facility_Name_GROUPED == "Pharmacie Chateau d'Eau"),
                    'Facility_Name_GROUPED'] = "Pharmacie Chateau d'Eau"
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'pharmacie Babacar sy')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Pharmacie Babacar Sy'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Babacar Sy'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'pharmacie Ceikh Ousmane Mbacké')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Pharmacie Ceikh Ousmane Mbacke'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Ceikh Ousmane Mbacke'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'Pharmacie du Baool Dr EL Badou Cissé tel :  33971-10-58   Diourbel')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Pharmacie du Baool Dr El-B.C.'),
                    'Facility_Name_GROUPED'] = 'Pharmacie du Baool Dr El-B.C.'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'Centre de Santé Roi Baudouin')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Centre de Sante Roi Baudouin'),
                    'Facility_Name_GROUPED'] = 'Centre de Sante Roi Baudouin'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'centre Hospitalier Régional de Thiès')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Centre Hospitalier Regional de Thies'),
                    'Facility_Name_GROUPED'] = 'Centre Hospitalier Regional de Thies'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'PRA Thiès')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'PRA Thies'),
                    'Facility_Name_GROUPED'] = 'PRA Thies'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'centre de sante de kedougou')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Centre de Sante de Kedougou'),
                    'Facility_Name_GROUPED'] = 'Centre de Sante de Kedougou'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'pharmacie sogui')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Pharmacie Sogui'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Sogui'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'Hopital Touba tel: 33-978-13-70')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Hopital Touba'),
                    'Facility_Name_GROUPED'] = 'Hopital Touba'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'centre de sante de Tambacounda')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Centre de Sante de Tambacounda'),
                    'Facility_Name_GROUPED'] = 'Centre de Sante de Tambacounda'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'pharmacie cheikh tidiane')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Pharmacie Cheikh Tidiane'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Cheikh Tidiane'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'Pharmacie Mame Diarra Bousso Dr Yéro Diouma Dian  tel: 33-971-34-35 Diourbel')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Pharmacie Mame Diarra Bousso Dr Y.D.D.'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Mame Diarra Bousso Dr Y.D.D.'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'pharmacie awa barry')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Pharmacie Awa Barry'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Awa Barry'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'pharmacie FOULADOU')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Pharmacie Fouladou'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Fouladou'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'pharmacie El hadj omar Tall')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Pharmacie El Hadj Omar Tall'),
                    'Facility_Name_GROUPED'] = 'Pharmacie El Hadj Omar Tall'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'pharmacie KANCISSE')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Pharmacie Kancisse'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Kancisse'
    SEN_df_2010.loc[(SEN_df_2010.Facility_Name_GROUPED == 'pharmacie KOLDA')
                    | (SEN_df_2010.Facility_Name_GROUPED == 'Pharmacie Kolda'),
                    'Facility_Name_GROUPED'] = 'Pharmacie Kolda'
    # END OF STRING CLEANING
    '''

    tbl_SEN_G1_2010 = SEN_df_2010[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    tbl_SEN_G1_2010 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in tbl_SEN_G1_2010]
    tbl_SEN_G2_2010 = SEN_df_2010[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    tbl_SEN_G2_2010 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in tbl_SEN_G2_2010]
    tbl_SEN_G3_2010 = SEN_df_2010[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    tbl_SEN_G3_2010 = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in tbl_SEN_G3_2010]
    '''
    tbl_SEN_G1_2010_nomissing = SEN_df_2010_nomissing[
        ['Province_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    tbl_SEN_G1_2010_nomissing = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in tbl_SEN_G1_2010_nomissing]
    tbl_SEN_G2_2010_nomissing = SEN_df_2010_nomissing[
        ['Facility_Location_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    tbl_SEN_G2_2010_nomissing = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in tbl_SEN_G2_2010_nomissing]
    tbl_SEN_G3_2010_nomissing = SEN_df_2010_nomissing[
        ['Facility_Name_GROUPED', 'Manufacturer_GROUPED', 'Final_Test_Conclusion']].values.tolist()
    tbl_SEN_G3_2010_nomissing = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in tbl_SEN_G3_2010_nomissing]
    '''
    # Print some overall summaries of the data
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
    SEN_df_2010[SEN_df_2010['Province_Name_GROUPED'].isin(['Dakar', 'Kaffrine', 'Kedougou', 'Kaolack']) & SEN_df_2010['Final_Test_Conclusion'].isin(['Fail'])].pivot_table(
        index=['Manufacturer_GROUPED'], columns=['Province_Name_GROUPED','Final_Test_Conclusion'],
        aggfunc='size', fill_value=0)
    SEN_df_2010[SEN_df_2010['Province_Name_GROUPED'].isin(['Matam', 'Kolda', 'Saint Louis']) & SEN_df_2010['Final_Test_Conclusion'].isin(['Fail'])].pivot_table(
        index=['Manufacturer_GROUPED'], columns=['Province_Name_GROUPED','Final_Test_Conclusion'],
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

    # DEIDENTIFICATION
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
        newName = 'Mnfr. ' + str(i+1)
        for ind, item in enumerate(tbl_SEN_G1_2010):
            if item[1] == currName:
                tbl_SEN_G1_2010[ind][1] = newName
        for ind, item in enumerate(tbl_SEN_G2_2010):
            if item[1] == currName:
                tbl_SEN_G2_2010[ind][1] = newName
        for ind, item in enumerate(tbl_SEN_G3_2010):
            if item[1] == currName:
                tbl_SEN_G3_2010[ind][1] = newName
    # Replace Province
    orig_PROV_lst = ['Dakar', 'Kaffrine', 'Kaolack', 'Kedougou', 'Kolda', 'Matam', 'Saint Louis']
    shuf_PROV_lst = orig_PROV_lst.copy()
    random.seed(333)
    random.shuffle(shuf_PROV_lst)
    # print(shuf_PROV_lst)
    for i in range(len(shuf_PROV_lst)):
        currName = shuf_PROV_lst[i]
        newName = 'Province ' + str(i+1)
        for ind, item in enumerate(tbl_SEN_G1_2010):
            if item[0] == currName:
                tbl_SEN_G1_2010[ind][0] = newName
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
        newName = 'District ' + str(i+1)
        for ind, item in enumerate(tbl_SEN_G2_2010):
            if item[0] == currName:
                tbl_SEN_G2_2010[ind][0] = newName
    # Swap Districts 7 & 8
    for ind, item in enumerate(tbl_SEN_G2_2010):
        if item[0] == 'District 7':
            tbl_SEN_G2_2010[ind][0] = 'District 8'
        elif item[0] == 'District 8':
            tbl_SEN_G2_2010[ind][0] = 'District 7'

    # Replace Facility Name
    orig_NAME_lst = ['CHR', 'CTA-Fann', 'Centre Hospitalier Regional de Thies', 'Centre de Sante Diourbel',
                     'Centre de Sante Mbacke', 'Centre de Sante Ousmane Ngom', 'Centre de Sante Roi Baudouin',
                     'Centre de Sante de Dioum', 'Centre de Sante de Kanel', 'Centre de Sante de Kedougou',
                     'Centre de Sante de Kolda', 'Centre de Sante de Koumpantoum', 'Centre de Sante de Matam',
                     'Centre de Sante de Richard Toll', 'Centre de Sante de Tambacounda',
                     'Centre de Sante de Velingara',
                     'Centre de Traitement de la Tuberculose de Touba', 'District Sanitaire Touba',
                     'District Sanitaire de Mbour',
                     'District Sanitaire de Rufisque', 'District Sanitaire de Tivaoune', 'District Sud',
                     'Hopital Diourbel',
                     'Hopital Regional de Saint Louis', 'Hopital Regionale de Ouro-Sogui', 'Hopital Touba',
                     'Hopital de Dioum',
                     'Hopitale Regionale de Koda', 'Hopitale Regionale de Tambacounda', 'PNA', 'PRA', 'PRA Diourbel',
                     'PRA Thies',
                     'Pharmacie', 'Pharmacie Awa Barry', 'Pharmacie Babacar Sy', 'Pharmacie Boubakh',
                     'Pharmacie Ceikh Ousmane Mbacke', 'Pharmacie Centrale Dr A.C.', "Pharmacie Chateau d'Eau",
                     'Pharmacie Cheikh Tidiane', 'Pharmacie El Hadj Omar Tall', 'Pharmacie Fouladou',
                     'Pharmacie Kancisse',
                     'Pharmacie Keneya', 'Pharmacie Kolda', 'Pharmacie Koldoise',
                     'Pharmacie Mame Diarra Bousso Dr Y.D.D.',
                     'Pharmacie Mame Fatou Diop Yoro', 'Pharmacie Mame Ibrahima Ndour Dr A.N.', 'Pharmacie Mame Madia',
                     'Pharmacie Ndamatou Dr O.N.', 'Pharmacie Oriantale', 'Pharmacie Oumou Khairy Ndiaye',
                     'Pharmacie Ousmane',
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
        newName = 'Facility ' + str(i+1)
        for ind, item in enumerate(tbl_SEN_G3_2010):
            if item[0] == currName:
                tbl_SEN_G3_2010[ind][0] = newName


    # RUN 1: s=1.0, r=1.0, prior is laplace(-2.5,3.5)
    priorMean = -2.5
    priorVar = 3.5

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    #util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
                       :5] + ',' + str(np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.95))[:5] + ')')

    lgDict = util.testresultsfiletotable(tbl_SEN_G2_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    #util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
    TNinds = lgDict['outletNames'].index('Facil. Location 7')
    print('Facility Location 7: (' + str(np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.05))[
                          :5] + ',' + str(np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.95))[:5] + ')')
    TNinds = lgDict['outletNames'].index('Facil. Location 8')
    print('Facility Location 8: (' + str(np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.05))[
                           :5] + ',' + str(np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.95))[:5] + ')')

    lgDict = util.testresultsfiletotable(tbl_SEN_G3_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Name', '\nSenegal - Facility Name'])

    # RUN 1b: s=1.0, r=1.0, prior is normal(-2.5,3.5)
    priorMean = -2.5
    priorVar = 3.5

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
                   'prior':  methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
    SNinds = lgDict['importerNames'].index('Mnfr. 5')
    print('Manufacturer 5: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    SNinds = lgDict['importerNames'].index('Mnfr. 8')
    print('Manufacturer 8: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    SNinds = lgDict['importerNames'].index('Mnfr. 10')
    print('Manufacturer 10: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    TNinds = lgDict['outletNames'].index('District 7')
    print('District 7: (' + str(
        np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.05))[
                                     :5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.95))[:5] + ')')
    TNinds = lgDict['outletNames'].index('District 8')
    print('District 8: (' + str(
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

    # RUN 1c: s=0.8, r=1.0, prior is normal(-2.5,3.5)
    priorMean = -2.5
    priorVar = 3.5
    s, r = 0.8, 1.0
    lgDict = util.testresultsfiletotable(tbl_SEN_G2_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar/2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
    SNinds = lgDict['importerNames'].index('Mnfr. 5')
    print('Manufacturer 5: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    SNinds = lgDict['importerNames'].index('Mnfr. 8')
    print('Manufacturer 8: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    SNinds = lgDict['importerNames'].index('Mnfr. 10')
    print('Manufacturer 10: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    TNinds = lgDict['outletNames'].index('District 7')
    print('District 7: (' + str(
        np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.05))[
                            :5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.95))[:5] + ')')
    TNinds = lgDict['outletNames'].index('District 8')
    print('District 8: (' + str(
        np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.05))[
                            :5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.95))[:5] + ')')

    s, r = 1.0, 0.95
    lgDict = util.testresultsfiletotable(tbl_SEN_G2_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
    SNinds = lgDict['importerNames'].index('Mnfr. 5')
    print('Manufacturer 5: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    SNinds = lgDict['importerNames'].index('Mnfr. 8')
    print('Manufacturer 8: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    SNinds = lgDict['importerNames'].index('Mnfr. 10')
    print('Manufacturer 10: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    TNinds = lgDict['outletNames'].index('District 7')
    print('District 7: (' + str(
        np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.05))[
                            :5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.95))[:5] + ')')
    TNinds = lgDict['outletNames'].index('District 8')
    print('District 8: (' + str(
        np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.05))[
                            :5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.95))[:5] + ')')


    # RUN 2: s=1.0, r=1.0, prior is laplace(-2.5,1.5)
    priorMean = -2.5
    priorVar = 1.5

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    #util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    #util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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

    ##### USE THIS RUN TO GENERATE PLOTS #####
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as sps
    import scipy.special as spsp
    priorMean = -2.5
    priorVar = 3.5
    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    numSN, numTN = lgDict['importerNum'], lgDict['outletNum']

    lowerQuant, upperQuant = 0.05, 0.95
    priorSamps = lgDict['prior'].expitrand(5000)
    #priorLower, priorUpper = np.quantile(priorSamps, lowerQuant), np.quantile(priorSamps, upperQuant)
    priorLower = spsp.expit(sps.laplace.ppf(lowerQuant, loc=priorMean, scale=np.sqrt(priorVar / 2)))
    priorUpper = spsp.expit(sps.laplace.ppf(upperQuant, loc=priorMean, scale=np.sqrt(priorVar / 2)))

    SNindsSubset = range(numSN)
    SNnames = [lgDict['importerNames'][i] for i in SNindsSubset]
    SNlowers = [np.quantile(lgDict['postSamples'][:, l], lowerQuant) for l in SNindsSubset]
    SNuppers = [np.quantile(lgDict['postSamples'][:, l], upperQuant) for l in SNindsSubset]
    floorVal = 0.05
    ceilVal = 0.3
    # First group
    SNlowers1 = [i for i in SNlowers if i > floorVal]
    SNuppers1 = [SNuppers[ind] for ind,i in enumerate(SNlowers) if i > floorVal]
    SNnames1 = [SNnames[ind] for ind,i in enumerate(SNlowers) if i > floorVal]
    midpoints1 = [SNuppers1[i] - (SNuppers1[i] - SNlowers1[i]) / 2 for i in range(len(SNuppers1))]
    zippedList1 = zip(midpoints1, SNuppers1, SNlowers1, SNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    SNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    SNuppers2 = [i for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers2 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNnames2 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    midpoints2 = [SNuppers2[i] - (SNuppers2[i] - SNlowers2[i]) / 2 for i in range(len(SNuppers2))]
    zippedList2 = zip(midpoints2, SNuppers2, SNlowers2, SNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    SNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    SNuppers3 = [i for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers3 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNnames3 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    midpoints3 = [SNuppers3[i] - (SNuppers3[i] - SNlowers3[i]) / 2 for i in range(len(SNuppers3))]
    zippedList3 = zip(midpoints3, SNuppers3, SNlowers3, SNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    SNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    SNnamesSorted = SNnamesSorted1.copy()
    #sorted_pairs = sorted_pairs1.copy()
    SNnamesSorted.append(' ')
    #sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted = SNnamesSorted + SNnamesSorted2
    #sorted_pairs = sorted_pairs + sorted_pairs2
    SNnamesSorted.append(' ')
    #sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted = SNnamesSorted + SNnamesSorted3
    #sorted_pairs = sorted_pairs + sorted_pairs3
    #sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted.append(' ')
    SNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(10, 10), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o-', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o-', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o-', color='red')
    plt.plot((SNnamesSorted[-1], SNnamesSorted[-1]), (priorLower, priorUpper), 'o--', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(SNnamesSorted)), SNnamesSorted, rotation=90)
    plt.title('Supply Node 90% Intervals\nManufacturer-Province Analysis',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Supply Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    fig.tight_layout()
    plt.show()
    plt.close()

    TNindsSubset = range(numTN)
    TNnames = [lgDict['outletNames'][i] for i in TNindsSubset]
    TNlowers = [np.quantile(lgDict['postSamples'][:, numSN + l], lowerQuant) for l in TNindsSubset]
    TNuppers = [np.quantile(lgDict['postSamples'][:, numSN + l], upperQuant) for l in TNindsSubset]
    floorVal = 0.05
    ceilVal = 0.3
    # First group
    TNlowers1 = [i for i in TNlowers if i > floorVal]
    TNuppers1 = [TNuppers[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    TNnames1 = [TNnames[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    midpoints1 = [TNuppers1[i] - (TNuppers1[i] - TNlowers1[i]) / 2 for i in range(len(TNuppers1))]
    zippedList1 = zip(midpoints1, TNuppers1, TNlowers1, TNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    TNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    TNuppers2 = [i for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers2 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNnames2 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    midpoints2 = [TNuppers2[i] - (TNuppers2[i] - TNlowers2[i]) / 2 for i in range(len(TNuppers2))]
    zippedList2 = zip(midpoints2, TNuppers2, TNlowers2, TNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    TNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    TNuppers3 = [i for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers3 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNnames3 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    midpoints3 = [TNuppers3[i] - (TNuppers3[i] - TNlowers3[i]) / 2 for i in range(len(TNuppers3))]
    zippedList3 = zip(midpoints3, TNuppers3, TNlowers3, TNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    TNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    TNnamesSorted = TNnamesSorted1.copy()
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted2
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted3
    TNnamesSorted.append(' ')
    TNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(10, 10), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o-', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o-', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o-', color='red')
    plt.plot((TNnamesSorted[-1], TNnamesSorted[-1]), (priorLower, priorUpper), 'o--', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(TNnamesSorted)), TNnamesSorted, rotation=90)
    plt.title('Test Node 90% Intervals\nManufacturer-Province Analysis',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Test Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    fig.tight_layout()
    plt.show()
    plt.close()

    # District as TNs; TRACKED
    priorMean = -2.5
    priorVar = 3.5
    lowerQuant, upperQuant = 0.05, 0.95
    priorLower = spsp.expit(sps.laplace.ppf(lowerQuant, loc=priorMean, scale=np.sqrt(priorVar / 2)))
    priorUpper = spsp.expit(sps.laplace.ppf(upperQuant, loc=priorMean, scale=np.sqrt(priorVar / 2)))

    lgDict = util.testresultsfiletotable(tbl_SEN_G2_2010, csvName=False)
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    numSN, numTN = lgDict['importerNum'], lgDict['outletNum']

    SNindsSubset = range(numSN)
    SNnames = [lgDict['importerNames'][i] for i in SNindsSubset]
    SNlowers = [np.quantile(lgDict['postSamples'][:, l], lowerQuant) for l in SNindsSubset]
    SNuppers = [np.quantile(lgDict['postSamples'][:, l], upperQuant) for l in SNindsSubset]
    floorVal = 0.05
    ceilVal = 0.3
    # First group
    SNlowers1 = [i for i in SNlowers if i > floorVal]
    SNuppers1 = [SNuppers[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    SNnames1 = [SNnames[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    midpoints1 = [SNuppers1[i] - (SNuppers1[i] - SNlowers1[i]) / 2 for i in range(len(SNuppers1))]
    zippedList1 = zip(midpoints1, SNuppers1, SNlowers1, SNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    SNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    SNuppers2 = [i for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers2 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNnames2 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    midpoints2 = [SNuppers2[i] - (SNuppers2[i] - SNlowers2[i]) / 2 for i in range(len(SNuppers2))]
    zippedList2 = zip(midpoints2, SNuppers2, SNlowers2, SNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    SNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    SNuppers3 = [i for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers3 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNnames3 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    midpoints3 = [SNuppers3[i] - (SNuppers3[i] - SNlowers3[i]) / 2 for i in range(len(SNuppers3))]
    zippedList3 = zip(midpoints3, SNuppers3, SNlowers3, SNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    SNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    SNnamesSorted = SNnamesSorted1.copy()
    # sorted_pairs = sorted_pairs1.copy()
    SNnamesSorted.append(' ')
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted = SNnamesSorted + SNnamesSorted2
    # sorted_pairs = sorted_pairs + sorted_pairs2
    SNnamesSorted.append(' ')
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted = SNnamesSorted + SNnamesSorted3
    # sorted_pairs = sorted_pairs + sorted_pairs3
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted.append(' ')
    SNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(10, 6), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o--', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o--', color='orange')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o:', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o:', color='green')
    plt.plot((SNnamesSorted[-1], SNnamesSorted[-1]), (priorLower, priorUpper), 'o-', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(SNnamesSorted)), SNnamesSorted, rotation=90)
    plt.title('Supply Node 90% Intervals\nManufacturer-District Analysis, Tracked Setting',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Supply Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    plt.axhline(y=floorVal, color='r', linestyle='-', alpha=0.1)  # line for 'l'
    plt.axhline(y=ceilVal, color='blue', linestyle='-', alpha=0.1)  # line for 'u'
    plt.text(26.3, ceilVal + .015, 'u=0.30', color='blue', alpha=0.5, size=9)
    plt.text(26.3, floorVal+.015, 'l=0.05', color='r', alpha=0.5, size=9)
    fig.tight_layout()
    plt.show()
    plt.close()

    TNindsSubset = range(numTN)
    TNnames = [lgDict['outletNames'][i] for i in TNindsSubset]
    TNlowers = [np.quantile(lgDict['postSamples'][:, numSN + l], lowerQuant) for l in TNindsSubset]
    TNuppers = [np.quantile(lgDict['postSamples'][:, numSN + l], upperQuant) for l in TNindsSubset]
    floorVal = 0.05
    ceilVal = 0.3
    # First group
    TNlowers1 = [i for i in TNlowers if i > floorVal]
    TNuppers1 = [TNuppers[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    TNnames1 = [TNnames[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    midpoints1 = [TNuppers1[i] - (TNuppers1[i] - TNlowers1[i]) / 2 for i in range(len(TNuppers1))]
    zippedList1 = zip(midpoints1, TNuppers1, TNlowers1, TNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    TNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    TNuppers2 = [i for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers2 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNnames2 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    midpoints2 = [TNuppers2[i] - (TNuppers2[i] - TNlowers2[i]) / 2 for i in range(len(TNuppers2))]
    zippedList2 = zip(midpoints2, TNuppers2, TNlowers2, TNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    TNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    TNuppers3 = [i for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers3 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNnames3 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    midpoints3 = [TNuppers3[i] - (TNuppers3[i] - TNlowers3[i]) / 2 for i in range(len(TNuppers3))]
    zippedList3 = zip(midpoints3, TNuppers3, TNlowers3, TNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    TNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    TNnamesSorted = TNnamesSorted1.copy()
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted2
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted3
    TNnamesSorted.append(' ')
    TNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(10, 6), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o--', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o--', color='orange')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o:', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o:', color='green')
    plt.plot((TNnamesSorted[-1], TNnamesSorted[-1]), (priorLower, priorUpper), 'o-', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(TNnamesSorted)), TNnamesSorted, rotation=90)
    plt.title('Test Node 90% Intervals\nManufacturer-District Analysis, Tracked Setting',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Test Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    plt.axhline(y=floorVal, color='r', linestyle='-', alpha=0.1)  # line for 'l'
    plt.axhline(y=ceilVal, color='blue', linestyle='-', alpha=0.1)  # line for 'u'
    plt.text(24.4, ceilVal + .015, 'u=0.30', color='blue', alpha=0.5, size=9)
    plt.text(24.4, floorVal + .015, 'l=0.05', color='r', alpha=0.5, size=9)
    fig.tight_layout()
    plt.show()
    plt.close()

    # District as TNs; UNTRACKED
    lgDict = util.testresultsfiletotable(tbl_SEN_G2_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    numSN, numTN = lgDict['importerNum'], lgDict['outletNum']

    SNindsSubset = range(numSN)
    SNnames = [lgDict['importerNames'][i] for i in SNindsSubset]
    SNlowers = [np.quantile(lgDict['postSamples'][:, l], lowerQuant) for l in SNindsSubset]
    SNuppers = [np.quantile(lgDict['postSamples'][:, l], upperQuant) for l in SNindsSubset]
    floorVal = 0.05
    ceilVal = 0.3
    # First group
    SNlowers1 = [i for i in SNlowers if i > floorVal]
    SNuppers1 = [SNuppers[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    SNnames1 = [SNnames[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    midpoints1 = [SNuppers1[i] - (SNuppers1[i] - SNlowers1[i]) / 2 for i in range(len(SNuppers1))]
    zippedList1 = zip(midpoints1, SNuppers1, SNlowers1, SNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    SNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    SNuppers2 = [i for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers2 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNnames2 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    midpoints2 = [SNuppers2[i] - (SNuppers2[i] - SNlowers2[i]) / 2 for i in range(len(SNuppers2))]
    zippedList2 = zip(midpoints2, SNuppers2, SNlowers2, SNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    SNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    SNuppers3 = [i for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers3 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNnames3 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    midpoints3 = [SNuppers3[i] - (SNuppers3[i] - SNlowers3[i]) / 2 for i in range(len(SNuppers3))]
    zippedList3 = zip(midpoints3, SNuppers3, SNlowers3, SNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    SNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    SNnamesSorted = SNnamesSorted1.copy()
    # sorted_pairs = sorted_pairs1.copy()
    SNnamesSorted.append(' ')
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted = SNnamesSorted + SNnamesSorted2
    # sorted_pairs = sorted_pairs + sorted_pairs2
    SNnamesSorted.append(' ')
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted = SNnamesSorted + SNnamesSorted3
    # sorted_pairs = sorted_pairs + sorted_pairs3
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted.append(' ')
    SNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(10, 6), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o--', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o--', color='orange')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o:', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o:', color='green')
    plt.plot((SNnamesSorted[-1], SNnamesSorted[-1]), (priorLower, priorUpper), 'o-', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(SNnamesSorted)), SNnamesSorted, rotation=90)
    plt.title('Supply Node 90% Intervals\nManufacturer-District Analysis, Untracked Setting',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Supply Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    plt.axhline(y=floorVal, color='r', linestyle='-', alpha=0.1)  # line for 'l'
    plt.axhline(y=ceilVal, color='blue', linestyle='-', alpha=0.1)  # line for 'u'
    plt.text(26.3, ceilVal + .015, 'u=0.30', color='blue', alpha=0.5, size=9)
    plt.text(26.3, floorVal + .015, 'l=0.05', color='r', alpha=0.5, size=9)
    fig.tight_layout()
    plt.show()
    plt.close()

    TNindsSubset = range(numTN)
    TNnames = [lgDict['outletNames'][i] for i in TNindsSubset]
    TNlowers = [np.quantile(lgDict['postSamples'][:, numSN + l], lowerQuant) for l in TNindsSubset]
    TNuppers = [np.quantile(lgDict['postSamples'][:, numSN + l], upperQuant) for l in TNindsSubset]
    floorVal = 0.05
    ceilVal = 0.3
    # First group
    TNlowers1 = [i for i in TNlowers if i > floorVal]
    TNuppers1 = [TNuppers[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    TNnames1 = [TNnames[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    midpoints1 = [TNuppers1[i] - (TNuppers1[i] - TNlowers1[i]) / 2 for i in range(len(TNuppers1))]
    zippedList1 = zip(midpoints1, TNuppers1, TNlowers1, TNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    TNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    TNuppers2 = [i for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers2 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNnames2 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    midpoints2 = [TNuppers2[i] - (TNuppers2[i] - TNlowers2[i]) / 2 for i in range(len(TNuppers2))]
    zippedList2 = zip(midpoints2, TNuppers2, TNlowers2, TNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    TNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    TNuppers3 = [i for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers3 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNnames3 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    midpoints3 = [TNuppers3[i] - (TNuppers3[i] - TNlowers3[i]) / 2 for i in range(len(TNuppers3))]
    zippedList3 = zip(midpoints3, TNuppers3, TNlowers3, TNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    TNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    TNnamesSorted = TNnamesSorted1.copy()
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted2
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted3
    TNnamesSorted.append(' ')
    TNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(10, 6), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o--', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o--', color='orange')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o:', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o:', color='green')
    plt.plot((TNnamesSorted[-1], TNnamesSorted[-1]), (priorLower, priorUpper), 'o-', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(TNnamesSorted)), TNnamesSorted, rotation=90)
    plt.title('Test Node 90% Intervals\nManufacturer-District Analysis, Untracked Setting',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Test Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    plt.axhline(y=floorVal, color='r', linestyle='-', alpha=0.1)  # line for 'l'
    plt.axhline(y=ceilVal, color='blue', linestyle='-', alpha=0.1)  # line for 'u'
    plt.text(24.4, ceilVal + .015, 'u=0.30', color='blue', alpha=0.5, size=9)
    plt.text(24.4, floorVal + .015, 'l=0.05', color='r', alpha=0.5, size=9)
    fig.tight_layout()
    plt.show()
    plt.close()

    # District as TNs; UNTRACKED; what if Q looked different?
    lgDict = util.testresultsfiletotable(tbl_SEN_G2_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    random.seed(31)
    for i, Nrow in enumerate(lgDict['N']):
        tempRow = Nrow / np.sum(Nrow)
        random.shuffle(tempRow)
        Q[i] = tempRow
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    numSN, numTN = lgDict['importerNum'], lgDict['outletNum']

    SNindsSubset = range(numSN)
    SNnames = [lgDict['importerNames'][i] for i in SNindsSubset]
    SNlowers = [np.quantile(lgDict['postSamples'][:, l], lowerQuant) for l in SNindsSubset]
    SNuppers = [np.quantile(lgDict['postSamples'][:, l], upperQuant) for l in SNindsSubset]
    floorVal = 0.05
    ceilVal = 0.3
    # First group
    SNlowers1 = [i for i in SNlowers if i > floorVal]
    SNuppers1 = [SNuppers[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    SNnames1 = [SNnames[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    midpoints1 = [SNuppers1[i] - (SNuppers1[i] - SNlowers1[i]) / 2 for i in range(len(SNuppers1))]
    zippedList1 = zip(midpoints1, SNuppers1, SNlowers1, SNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    SNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    SNuppers2 = [i for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers2 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNnames2 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    midpoints2 = [SNuppers2[i] - (SNuppers2[i] - SNlowers2[i]) / 2 for i in range(len(SNuppers2))]
    zippedList2 = zip(midpoints2, SNuppers2, SNlowers2, SNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    SNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    SNuppers3 = [i for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers3 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNnames3 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    midpoints3 = [SNuppers3[i] - (SNuppers3[i] - SNlowers3[i]) / 2 for i in range(len(SNuppers3))]
    zippedList3 = zip(midpoints3, SNuppers3, SNlowers3, SNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    SNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    SNnamesSorted = SNnamesSorted1.copy()
    # sorted_pairs = sorted_pairs1.copy()
    SNnamesSorted.append(' ')
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted = SNnamesSorted + SNnamesSorted2
    # sorted_pairs = sorted_pairs + sorted_pairs2
    SNnamesSorted.append(' ')
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted = SNnamesSorted + SNnamesSorted3
    # sorted_pairs = sorted_pairs + sorted_pairs3
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted.append(' ')
    SNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(10, 6), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o--', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o--', color='orange')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o:', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o:', color='green')
    plt.plot((SNnamesSorted[-1], SNnamesSorted[-1]), (priorLower, priorUpper), 'o-', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(SNnamesSorted)), SNnamesSorted, rotation=90)
    plt.title('Supply Node 90% Intervals\nManufacturer-District Analysis, Untracked Setting',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Supply Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    plt.axhline(y=floorVal, color='r', linestyle='-', alpha=0.1)  # line for 'l'
    plt.axhline(y=ceilVal, color='blue', linestyle='-', alpha=0.1)  # line for 'u'
    plt.text(26.3, ceilVal + .015, 'u=0.30', color='blue', alpha=0.5, size=9)
    plt.text(26.3, floorVal + .015, 'l=0.05', color='r', alpha=0.5, size=9)
    fig.tight_layout()
    plt.show()
    plt.close()

    TNindsSubset = range(numTN)
    TNnames = [lgDict['outletNames'][i] for i in TNindsSubset]
    TNlowers = [np.quantile(lgDict['postSamples'][:, numSN + l], lowerQuant) for l in TNindsSubset]
    TNuppers = [np.quantile(lgDict['postSamples'][:, numSN + l], upperQuant) for l in TNindsSubset]
    floorVal = 0.05
    ceilVal = 0.3
    # First group
    TNlowers1 = [i for i in TNlowers if i > floorVal]
    TNuppers1 = [TNuppers[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    TNnames1 = [TNnames[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    midpoints1 = [TNuppers1[i] - (TNuppers1[i] - TNlowers1[i]) / 2 for i in range(len(TNuppers1))]
    zippedList1 = zip(midpoints1, TNuppers1, TNlowers1, TNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    TNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    TNuppers2 = [i for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers2 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNnames2 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    midpoints2 = [TNuppers2[i] - (TNuppers2[i] - TNlowers2[i]) / 2 for i in range(len(TNuppers2))]
    zippedList2 = zip(midpoints2, TNuppers2, TNlowers2, TNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    TNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    TNuppers3 = [i for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers3 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNnames3 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    midpoints3 = [TNuppers3[i] - (TNuppers3[i] - TNlowers3[i]) / 2 for i in range(len(TNuppers3))]
    zippedList3 = zip(midpoints3, TNuppers3, TNlowers3, TNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    TNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    TNnamesSorted = TNnamesSorted1.copy()
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted2
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted3
    TNnamesSorted.append(' ')
    TNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(10, 6), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o--', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o--', color='orange')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o:', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o:', color='green')
    plt.plot((TNnamesSorted[-1], TNnamesSorted[-1]), (priorLower, priorUpper), 'o-', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(TNnamesSorted)), TNnamesSorted, rotation=90)
    plt.title('Test Node 90% Intervals\nManufacturer-District Analysis, Untracked Setting',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Test Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    plt.axhline(y=floorVal, color='r', linestyle='-', alpha=0.1)  # line for 'l'
    plt.axhline(y=ceilVal, color='blue', linestyle='-', alpha=0.1)  # line for 'u'
    plt.text(24.4, ceilVal + .015, 'u=0.30', color='blue', alpha=0.5, size=9)
    plt.text(24.4, floorVal + .015, 'l=0.05', color='r', alpha=0.5, size=9)
    fig.tight_layout()
    plt.show()
    plt.close()

    # Facility Location as TNs
    lgDict = util.testresultsfiletotable(tbl_SEN_G3_2010, csvName=False)
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    numSN, numTN = lgDict['importerNum'], lgDict['outletNum']

    SNindsSubset = range(numSN)
    SNnames = [lgDict['importerNames'][i] for i in SNindsSubset]
    SNlowers = [np.quantile(lgDict['postSamples'][:, l], lowerQuant) for l in SNindsSubset]
    SNuppers = [np.quantile(lgDict['postSamples'][:, l], upperQuant) for l in SNindsSubset]
    floorVal = 0.05
    ceilVal = 0.3
    # First group
    SNlowers1 = [i for i in SNlowers if i > floorVal]
    SNuppers1 = [SNuppers[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    SNnames1 = [SNnames[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    midpoints1 = [SNuppers1[i] - (SNuppers1[i] - SNlowers1[i]) / 2 for i in range(len(SNuppers1))]
    zippedList1 = zip(midpoints1, SNuppers1, SNlowers1, SNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    SNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    SNuppers2 = [i for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers2 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNnames2 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    midpoints2 = [SNuppers2[i] - (SNuppers2[i] - SNlowers2[i]) / 2 for i in range(len(SNuppers2))]
    zippedList2 = zip(midpoints2, SNuppers2, SNlowers2, SNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    SNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    SNuppers3 = [i for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers3 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNnames3 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    midpoints3 = [SNuppers3[i] - (SNuppers3[i] - SNlowers3[i]) / 2 for i in range(len(SNuppers3))]
    zippedList3 = zip(midpoints3, SNuppers3, SNlowers3, SNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    SNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    SNnamesSorted = SNnamesSorted1.copy()
    # sorted_pairs = sorted_pairs1.copy()
    SNnamesSorted.append(' ')
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted = SNnamesSorted + SNnamesSorted2
    # sorted_pairs = sorted_pairs + sorted_pairs2
    SNnamesSorted.append(' ')
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted = SNnamesSorted + SNnamesSorted3
    # sorted_pairs = sorted_pairs + sorted_pairs3
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted.append(' ')
    SNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(10, 10), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o-', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o-', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o-', color='red')
    plt.plot((SNnamesSorted[-1], SNnamesSorted[-1]), (priorLower, priorUpper), 'o--', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(SNnamesSorted)), SNnamesSorted, rotation=90)
    plt.title('Supply Node 90% Intervals\nManufacturer-Facility Analysis',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Supply Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    fig.tight_layout()
    plt.show()
    plt.close()

    TNindsSubset = range(numTN)
    TNnames = [lgDict['outletNames'][i] for i in TNindsSubset]
    TNlowers = [np.quantile(lgDict['postSamples'][:, numSN + l], lowerQuant) for l in TNindsSubset]
    TNuppers = [np.quantile(lgDict['postSamples'][:, numSN + l], upperQuant) for l in TNindsSubset]
    floorVal = 0.05
    ceilVal = 0.3
    # First group
    TNlowers1 = [i for i in TNlowers if i > floorVal]
    TNuppers1 = [TNuppers[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    TNnames1 = [TNnames[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    midpoints1 = [TNuppers1[i] - (TNuppers1[i] - TNlowers1[i]) / 2 for i in range(len(TNuppers1))]
    zippedList1 = zip(midpoints1, TNuppers1, TNlowers1, TNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    TNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    TNuppers2 = [i for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers2 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNnames2 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    midpoints2 = [TNuppers2[i] - (TNuppers2[i] - TNlowers2[i]) / 2 for i in range(len(TNuppers2))]
    zippedList2 = zip(midpoints2, TNuppers2, TNlowers2, TNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    TNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    TNuppers3 = [i for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers3 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNnames3 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    midpoints3 = [TNuppers3[i] - (TNuppers3[i] - TNlowers3[i]) / 2 for i in range(len(TNuppers3))]
    zippedList3 = zip(midpoints3, TNuppers3, TNlowers3, TNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    TNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    TNnamesSorted = TNnamesSorted1.copy()
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted2
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted3
    TNnamesSorted.append(' ')
    TNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(10, 10), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o-', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o-', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o-', color='red')
    plt.plot((TNnamesSorted[-1], TNnamesSorted[-1]), (priorLower, priorUpper), 'o--', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(TNnamesSorted)), TNnamesSorted, rotation=90)
    plt.title('Test Node 90% Intervals\nManufacturer-Facility Analysis',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Test Node Name', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 16, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(9)
    fig.tight_layout()
    plt.show()
    plt.close()

    # What does a good prior look like?
    mean = -2.5
    var = 1.5
    s = np.random.laplace(mean, np.sqrt(var/2), 10000)
    t = np.exp(s) / (1 + np.exp(s))
    print(np.mean(t))
    import matplotlib.pyplot as plt
    plt.hist(s, density=True, bins=30)
    plt.show()
    plt.hist(t, density=True, bins=30)
    plt.show()

    mean = -2.5
    var = 1.5
    s = np.random.normal(mean, np.sqrt(var), 10000)
    t = np.exp(s) / (1 + np.exp(s))
    print(np.mean(t))
    plt.hist(s, density=True, bins=30)
    plt.show()
    plt.hist(t, density=True, bins=30)
    plt.show()


    import scipy.stats as sps
    import scipy.special as spsp
    int50 = sps.laplace.ppf(0.50, loc=mean, scale=np.sqrt(var / 2))
    int05 = sps.laplace.ppf(0.05, loc=mean, scale=np.sqrt(var / 2))
    int95 = sps.laplace.ppf(0.95, loc=mean, scale=np.sqrt(var / 2))
    int70 = sps.laplace.ppf(0.70, loc=mean, scale=np.sqrt(var / 2))
    print(spsp.expit(int05), spsp.expit(int50), spsp.expit(int70), spsp.expit(int95))
    print(spsp.expit(int05), spsp.expit(int95))

    # Generate samples for paper example in Section 3, to be used in Section 5
    lgDict = {}
    priorMean, priorVar = -2, 1
    numPostSamps = 1000
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    int50 = sps.norm.ppf(0.50, loc=priorMean, scale=np.sqrt(priorVar))
    int05 = sps.norm.ppf(0.05, loc=priorMean, scale=np.sqrt(priorVar))
    int95 = sps.norm.ppf(0.95, loc=priorMean, scale=np.sqrt(priorVar))
    int70 = sps.norm.ppf(0.70, loc=priorMean, scale=np.sqrt(priorVar))
    print(spsp.expit(int05), spsp.expit(int50), spsp.expit(int70), spsp.expit(int95))
    Ntoy = np.array([[6, 11], [12, 6], [2, 13]])
    Ytoy = np.array([[3, 0], [6, 0], [0, 0]])
    TNnames, SNnames = ['Test Node 1', 'Test Node 2', 'Test Node 3'], ['Supply Node 1', 'Supply Node 2']
    lgDict.update({'type':'Tracked', 'outletNum':3, 'importerNum':2, 'diagSens':1.0, 'diagSpec':1.0,
                   'N':Ntoy, 'Y':Ytoy, 'numPostSamples': numPostSamps, 'MCMCdict': MCMCdict,
                   'outletNames':TNnames, 'importerNames':SNnames,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar)})
    lgDict = methods.GeneratePostSamples(lgDict)
    numSN, numTN = lgDict['importerNum'], lgDict['outletNum']
    lowerQuant, upperQuant = 0.05, 0.95
    priorLower = spsp.expit(sps.norm.ppf(lowerQuant, loc=priorMean, scale=np.sqrt(priorVar)))
    priorUpper = spsp.expit(sps.norm.ppf(upperQuant, loc=priorMean, scale=np.sqrt(priorVar)))
    SNindsSubset = range(numSN)

    SNnames = [lgDict['importerNames'][i] for i in SNindsSubset]
    SNlowers = [np.quantile(lgDict['postSamples'][:, l], lowerQuant) for l in SNindsSubset]
    SNuppers = [np.quantile(lgDict['postSamples'][:, l], upperQuant) for l in SNindsSubset]
    floorVal = 0.05
    ceilVal = 0.2
    # First group
    SNlowers1 = [i for i in SNlowers if i > floorVal]
    SNuppers1 = [SNuppers[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    SNnames1 = [SNnames[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    midpoints1 = [SNuppers1[i] - (SNuppers1[i] - SNlowers1[i]) / 2 for i in range(len(SNuppers1))]
    zippedList1 = zip(midpoints1, SNuppers1, SNlowers1, SNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    SNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    SNuppers2 = [i for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers2 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    SNnames2 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i > ceilVal and SNlowers[ind] <= floorVal)]
    midpoints2 = [SNuppers2[i] - (SNuppers2[i] - SNlowers2[i]) / 2 for i in range(len(SNuppers2))]
    zippedList2 = zip(midpoints2, SNuppers2, SNlowers2, SNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    SNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    SNuppers3 = [i for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers3 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNnames3 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    midpoints3 = [SNuppers3[i] - (SNuppers3[i] - SNlowers3[i]) / 2 for i in range(len(SNuppers3))]
    zippedList3 = zip(midpoints3, SNuppers3, SNlowers3, SNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    SNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    SNnamesSorted = SNnamesSorted1.copy()
    # sorted_pairs = sorted_pairs1.copy()
    SNnamesSorted.append(' ')
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted = SNnamesSorted + SNnamesSorted2
    # sorted_pairs = sorted_pairs + sorted_pairs2
    SNnamesSorted.append(' ')
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted = SNnamesSorted + SNnamesSorted3
    # sorted_pairs = sorted_pairs + sorted_pairs3
    # sorted_pairs.append((np.nan, np.nan, np.nan, ' '))
    SNnamesSorted.append(' ')
    SNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(5, 5), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o--', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o--', color='orange')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o:', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o:', color='green')
    plt.plot((SNnamesSorted[-1], SNnamesSorted[-1]), (priorLower, priorUpper), 'o-', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(SNnamesSorted)), SNnamesSorted, rotation=90)
    plt.title('Supply Node 90% Intervals\nExample',
              fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Supply Node Name', fontdict={'fontsize': 12, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 12, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(11)
    plt.axhline(y=floorVal, color='r', linestyle='-', alpha=0.3) # line for 'l'
    plt.axhline(y=ceilVal, color='blue', linestyle='-', alpha=0.3) # line for 'u'
    fig.tight_layout()
    plt.show()
    plt.close()

    TNindsSubset = range(numTN)
    TNnames = [lgDict['outletNames'][i] for i in TNindsSubset]
    TNlowers = [np.quantile(lgDict['postSamples'][:, numSN + l], lowerQuant) for l in TNindsSubset]
    TNuppers = [np.quantile(lgDict['postSamples'][:, numSN + l], upperQuant) for l in TNindsSubset]
    floorVal = 0.05
    ceilVal = 0.2
    # First group
    TNlowers1 = [i for i in TNlowers if i > floorVal]
    TNuppers1 = [TNuppers[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    TNnames1 = [TNnames[ind] for ind, i in enumerate(TNlowers) if i > floorVal]
    midpoints1 = [TNuppers1[i] - (TNuppers1[i] - TNlowers1[i]) / 2 for i in range(len(TNuppers1))]
    zippedList1 = zip(midpoints1, TNuppers1, TNlowers1, TNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    TNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    TNuppers2 = [i for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers2 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNnames2 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    midpoints2 = [TNuppers2[i] - (TNuppers2[i] - TNlowers2[i]) / 2 for i in range(len(TNuppers2))]
    zippedList2 = zip(midpoints2, TNuppers2, TNlowers2, TNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    TNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    TNuppers3 = [i for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers3 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNnames3 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    midpoints3 = [TNuppers3[i] - (TNuppers3[i] - TNlowers3[i]) / 2 for i in range(len(TNuppers3))]
    zippedList3 = zip(midpoints3, TNuppers3, TNlowers3, TNnames3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)
    TNnamesSorted3 = [tup[-1] for tup in sorted_pairs3]
    # Combine groups
    TNnamesSorted = TNnamesSorted1.copy()
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted2
    TNnamesSorted.append(' ')
    TNnamesSorted = TNnamesSorted + TNnamesSorted3
    TNnamesSorted.append(' ')
    TNnamesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(5, 5), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        plt.plot((name, name), (lower, upper), 'o--', color='orange')
    plt.plot((' ', ' '), (np.nan, np.nan), 'o--', color='orange')
    for _, upper, lower, name in sorted_pairs3:
        plt.plot((name, name), (lower, upper), 'o:', color='green')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o:', color='green')
    plt.plot((TNnamesSorted[-1], TNnamesSorted[-1]), (priorLower, priorUpper), 'o-', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(TNnamesSorted)), TNnamesSorted, rotation=90)
    plt.title('Test Node 90% Intervals\nExample',
              fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Test Node Name', fontdict={'fontsize': 12, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 12, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(11)
    plt.axhline(y=floorVal, color='r', linestyle='-', alpha=0.3)  # line for 'l'
    plt.axhline(y=ceilVal, color='blue', linestyle='-', alpha=0.3)  # line for 'u'
    fig.tight_layout()
    plt.show()
    plt.close()

    # COMBINED INTO ONE PLOT; FORMATTED FOR VERY PARTICULAR DATA SET, E.G., SKIPS HIGH RISK INTERVALS FOR TEST NODES
    floorVal = 0.05
    ceilVal = 0.2
    # First group
    SNlowers1 = [i for i in SNlowers if i > floorVal]
    SNuppers1 = [SNuppers[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    SNnames1 = [SNnames[ind] for ind, i in enumerate(SNlowers) if i > floorVal]
    midpoints1 = [SNuppers1[i] - (SNuppers1[i] - SNlowers1[i]) / 2 for i in range(len(SNuppers1))]
    zippedList1 = zip(midpoints1, SNuppers1, SNlowers1, SNnames1)
    sorted_pairs1 = sorted(zippedList1, reverse=True)
    SNnamesSorted1 = [tup[-1] for tup in sorted_pairs1]
    # Second group
    TNuppers2 = [i for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers2 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    TNnames2 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i > ceilVal and TNlowers[ind] <= floorVal)]
    midpoints2 = [TNuppers2[i] - (TNuppers2[i] - TNlowers2[i]) / 2 for i in range(len(TNuppers2))]
    zippedList2 = zip(midpoints2, TNuppers2, TNlowers2, TNnames2)
    sorted_pairs2 = sorted(zippedList2, reverse=True)
    TNnamesSorted2 = [tup[-1] for tup in sorted_pairs2]
    # Third group
    TNuppers3 = [i for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNlowers3 = [TNlowers[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNnames3 = [TNnames[ind] for ind, i in enumerate(TNuppers) if (i <= ceilVal and TNlowers[ind] <= floorVal)]
    TNmidpoints3 = [TNuppers3[i] - (TNuppers3[i] - TNlowers3[i]) / 2 for i in range(len(TNuppers3))]
    TNzippedList3 = zip(TNmidpoints3, TNuppers3, TNlowers3, TNnames3)
    TNsorted_pairs3 = sorted(TNzippedList3, reverse=True)
    TNnamesSorted3 = [tup[-1] for tup in TNsorted_pairs3]

    SNuppers3 = [i for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNlowers3 = [SNlowers[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNnames3 = [SNnames[ind] for ind, i in enumerate(SNuppers) if (i <= ceilVal and SNlowers[ind] <= floorVal)]
    SNmidpoints3 = [SNuppers3[i] - (SNuppers3[i] - SNlowers3[i]) / 2 for i in range(len(SNuppers3))]
    SNzippedList3 = zip(SNmidpoints3, SNuppers3, SNlowers3, SNnames3)
    SNsorted_pairs3 = sorted(SNzippedList3, reverse=True)
    SNnamesSorted3 = [tup[-1] for tup in SNsorted_pairs3]

    midpoints3 = TNmidpoints3 + SNmidpoints3
    uppers3 = TNuppers3 + SNuppers3
    lowers3 = TNlowers3 + SNlowers3
    names3 = TNnames3 + SNnames3
    zippedList3 = zip(midpoints3, uppers3, lowers3, names3)
    sorted_pairs3 = sorted(zippedList3, reverse=True)

    # Combine groups
    namesSorted = SNnamesSorted1.copy()
    #namesSorted.append(' ')
    namesSorted = namesSorted + TNnamesSorted2
    #namesSorted.append(' ')
    namesSorted = namesSorted + TNnamesSorted3 + SNnamesSorted3
    namesSorted.append(' ')
    namesSorted.append('(Prior)')
    fig, (ax) = plt.subplots(figsize=(5, 5), ncols=1)
    for _, upper, lower, name in sorted_pairs1:
        #plt.plot((name, name), (lower, upper), 'o-', color='red')
        plt.plot((name, name), (lower, upper), 'o-', color='blue')
 #   plt.plot(('', ''), (np.nan, np.nan), 'o-', color='red')
    for _, upper, lower, name in sorted_pairs2:
        #plt.plot((name, name), (lower, upper), 'o--', color='orange')
        plt.plot((name, name), (lower, upper), 'o-', color='blue')
    #plt.plot((' ', ' '), (np.nan, np.nan), 'o--', color='orange')
    for _, upper, lower, name in sorted_pairs3:
        #plt.plot((name, name), (lower, upper), 'o:', color='green')
        plt.plot((name, name), (lower, upper), 'o-', color='blue')
    plt.plot(('  ', '  '), (np.nan, np.nan), 'o:', color='green')
    plt.plot((namesSorted[-1], namesSorted[-1]), (priorLower, priorUpper), 'o-', color='gray')
    plt.ylim([0, 0.6])
    plt.xticks(range(len(namesSorted)), namesSorted, rotation=90)
    plt.title('Node 90% Intervals',
              fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Node Name', fontdict={'fontsize': 12, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 12, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(11)
    #plt.axhline(y=floorVal, color='r', linestyle='-', alpha=0.1)  # line for 'l'
    #plt.axhline(y=ceilVal, color='blue', linestyle='-', alpha=0.1)  # line for 'u'
    #plt.text(6.7, 0.215, 'u=0.20', color='blue', alpha=0.5)
    #plt.text(6.7, 0.065, 'l=0.05', color='r', alpha=0.5)
    fig.tight_layout()
    plt.show()
    plt.close()

    # 90% CI VALUES USING NEWTON, 2009
    for i in range(numSN):  # sum across TNs to see totals for SNs
        currTotal = np.sum(lgDict['N'], axis=0)[i]
        currPos = np.sum(lgDict['Y'], axis=0)[i]
        pHat = currPos / currTotal
        lowerBd = pHat - (1.645 * np.sqrt(pHat * (1 - pHat) / currTotal))
        upperBd = pHat + (1.645 * np.sqrt(pHat * (1 - pHat) / currTotal))
        print(lgDict['importerNames'][i] + ': (' + str(lowerBd)[:5] + ', ' + str(upperBd)[:5] + '), ' + str(currPos) + '/'+str(currTotal))
    # Test nodes
    for i in range(numTN):  # sum across SNs to see totals for TNs
        currTotal = np.sum(lgDict['N'], axis=1)[i]
        currPos = np.sum(lgDict['Y'], axis=1)[i]
        pHat = currPos / currTotal
        lowerBd = pHat - (1.645 * np.sqrt(pHat * (1 - pHat) / currTotal))
        upperBd = pHat + (1.645 * np.sqrt(pHat * (1 - pHat) / currTotal))
        print(lgDict['outletNames'][i] + ': (' + str(lowerBd)[:5] + ', ' + str(upperBd)[:5] + '), ' + str(currPos) + '/'+ str(currTotal))


    # TIMING ANALYSIS
    import time

    numPostSamps = 1000
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}

    # Look at difference in runtimes for HMC and LMC
    times1 = []
    times2 = []
    for runs in range(20):
        testSysDict = util.generateRandDataDict(numImp=50, numOut=50, diagSens=0.90,
                             diagSpec=0.99, numSamples=50 * 20,
                             dataType='Tracked', transMatLambda=1.1,
                             randSeed=-1,trueRates=[])
        testSysDict = util.GetVectorForms(testSysDict)
        print(np.count_nonzero(testSysDict['N']))
        priorMean, priorVar = -2.4, 1
        MCMCdict.update({'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4})
        testSysDict.update({'numPostSamples': numPostSamps, 'MCMCdict': MCMCdict,
                            'prior': methods.prior_normal(mu=priorMean, var=priorVar)})
        startTime = time.time()
        testSysDict = methods.GeneratePostSamples(testSysDict)
        endTime = time.time()
        print(endTime-startTime)
        times1.append(endTime-startTime)
        MCMCdict.update({'MCMCtype': 'Langevin'})
        testSysDict.update({'MCMCdict': MCMCdict})
        startTime = time.time()
        testSysDict = methods.GeneratePostSamples(testSysDict)
        endTime = time.time()
        print(endTime - startTime)
        times2.append(endTime-startTime)
    print(np.max(times1), np.min(times1), np.mean(times1))
    print(np.max(times2), np.min(times2), np.mean(times2))
    # Look at effect of more supply-chain traces
    baseN = [346,318,332,331,361,348,351,321,334,341,322,328,315,307,341,333,331,344,334,323]
    print(np.mean(baseN)/(50*50))
    MCMCdict.update({'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4})
    times3 = [] # Less supply-chain traces
    lowerN = []
    for runs in range(20):
        testSysDict = util.generateRandDataDict(numImp=50, numOut=50, diagSens=0.90,
                                                diagSpec=0.99, numSamples=50 * 20,
                                                dataType='Tracked', transMatLambda=0.5,
                                                randSeed=-1, trueRates=[])
        testSysDict = util.GetVectorForms(testSysDict)
        lowerN.append(np.count_nonzero(testSysDict['N']))
        priorMean, priorVar = -2.4, 1
        testSysDict.update({'numPostSamples': numPostSamps, 'MCMCdict': MCMCdict,
                            'prior': methods.prior_normal(mu=priorMean, var=priorVar)})
        startTime = time.time()
        testSysDict = methods.GeneratePostSamples(testSysDict)
        endTime = time.time()
        print(endTime - startTime)
        times3.append(endTime - startTime)
    print(np.max(times3), np.min(times3), np.mean(times3))
    print(np.average(lowerN)/(50*50))
    times4 = [] # More supply-chain traces
    upperN = []
    for runs in range(20):
        testSysDict = util.generateRandDataDict(numImp=50, numOut=50, diagSens=0.90,
                                                diagSpec=0.99, numSamples=50 * 20,
                                                dataType='Tracked', transMatLambda=4.5,
                                                randSeed=-1, trueRates=[])
        testSysDict = util.GetVectorForms(testSysDict)
        upperN.append(np.count_nonzero(testSysDict['N']))
        priorMean, priorVar = -2.4, 1
        testSysDict.update({'numPostSamples': numPostSamps, 'MCMCdict': MCMCdict,
                            'prior': methods.prior_normal(mu=priorMean, var=priorVar)})
        startTime = time.time()
        testSysDict = methods.GeneratePostSamples(testSysDict)
        endTime = time.time()
        print(endTime - startTime)
        times4.append(endTime - startTime)
    print(np.max(times4), np.min(times4), np.mean(times4))
    print(np.average(upperN)/(50*50))
    # Look at effect of less or more nodes
    times5 = [] # Less nodes
    upperN = []
    for runs in range(20):
        testSysDict = util.generateRandDataDict(numImp=25, numOut=25, diagSens=0.90,
                                                diagSpec=0.99, numSamples= 50 * 20,
                                                dataType='Tracked', transMatLambda=1.1,
                                                randSeed=-1, trueRates=[])
        testSysDict = util.GetVectorForms(testSysDict)
        upperN.append(np.count_nonzero(testSysDict['N']))
        priorMean, priorVar = -2.4, 1
        testSysDict.update({'numPostSamples': numPostSamps, 'MCMCdict': MCMCdict,
                            'prior': methods.prior_normal(mu=priorMean, var=priorVar)})
        startTime = time.time()
        testSysDict = methods.GeneratePostSamples(testSysDict)
        endTime = time.time()
        print(endTime - startTime)
        times5.append(endTime - startTime)
    print(np.max(times5), np.min(times5), np.mean(times5))
    print(np.average(upperN)/(25*25))
    times6 = []  # More nodes
    upperN = []
    for runs in range(20):
        testSysDict = util.generateRandDataDict(numImp=100, numOut=100, diagSens=0.90,
                                                diagSpec=0.99, numSamples=50 * 20,
                                                dataType='Tracked', transMatLambda=1.1,
                                                randSeed=-1, trueRates=[])
        testSysDict = util.GetVectorForms(testSysDict)
        upperN.append(np.count_nonzero(testSysDict['N']))
        priorMean, priorVar = -2.4, 1
        testSysDict.update({'numPostSamples': numPostSamps, 'MCMCdict': MCMCdict,
                            'prior': methods.prior_normal(mu=priorMean, var=priorVar)})
        startTime = time.time()
        testSysDict = methods.GeneratePostSamples(testSysDict)
        endTime = time.time()
        print(endTime - startTime)
        times6.append(endTime - startTime)
    print(np.max(times6), np.min(times6), np.mean(times6))
    print(np.average(upperN) / (100 * 100))



    ##### END OF MANUAL PLOT GENERATION #####

    # RUN 3: s=1.0, r=1.0, prior is laplace(-3.5, 3.5)
    priorMean = -3.5
    priorVar = 3.5

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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

    # RUN 4: s=1.0, r=1.0, prior is laplace(-3.5, 1.5)
    priorMean = -3.5
    priorVar = 1.5

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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

    import numpy as np

    # RUN 5: s=1.0, r=1.0, prior is laplace(-2.5, 3.5 ) ; UNTRACKED
    priorMean = -2.5
    priorVar = 3.5

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    #util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    #util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Name', '\nSenegal - Facility Name'])

    # RUN 6: s=1.0, r=1.0, prior is laplace(-2.5, 1.5 ) ; UNTRACKED
    priorMean = -2.5
    priorVar = 1.5

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Name', '\nSenegal - Facility Name'])

    # RUN 7: s=1.0, r=1.0, prior is laplace(-2.5, 1.5 ) ; UNTRACKED
    priorMean = -3.5
    priorVar = 3.5

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Name', '\nSenegal - Facility Name'])

    # RUN 8: s=1.0, r=1.0, prior is laplace(-2.5, 1.5 ) ; UNTRACKED
    priorMean = -3.5
    priorVar = 1.5

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Name', '\nSenegal - Facility Name'])

    # RUN 9: s=0.8, r=0.95, prior is MQDB countries with at least 1 SFP
    priorMean = -2.5
    priorVar = 3.5
    s, r = 0.8, 0.95

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Name', '\nSenegal - Facility Name'])

    # RUN 10: s=0.8, r=0.95, prior is MQDB countries with at least 1 SFP, with 5 times the variance
    priorMean = -2.5
    priorVar = 1.5
    s, r = 0.8, 0.95

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Name', '\nSenegal - Facility Name'])

    # RUN 11: s=0.8, r=0.95, prior is Ozawa Africa countries with n>=150
    priorMean = -3.5
    priorVar = 3.5
    s, r = 0.8, 0.95

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Name', '\nSenegal - Facility Name'])

    # RUN 12: s=0.8, r=0.95, prior is Ozawa Africa countries with n>=150, with 5 times the variance
    priorMean = -3.5
    priorVar = 1.5
    s, r = 0.8, 0.95

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Name', '\nSenegal - Facility Name'])

    import numpy as np

    # RUN 13: s=0.8, r=0.95, prior is MQDB countries with at least 1 SFP; UNTRACKED
    priorMean = -2.5
    priorVar = 3.5
    s, r = 0.8, 0.95

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Name', '\nSenegal - Facility Name'])

    # RUN 14: s=0.8, r=0.95, prior is MQDB countries with at least 1 SFP, with 5 times the variance; UNTRACKED
    priorMean = -2.5
    priorVar = 1.5
    s, r = 0.8, 0.95

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Name', '\nSenegal - Facility Name'])

    # RUN 15: s=0.8, r=0.95, prior is Ozawa Africa studies w n>=150; UNTRACKED
    priorMean = -3.5
    priorVar = 3.5
    s, r = 0.8, 0.95

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Name', '\nSenegal - Facility Name'])

    # RUN 16: s=0.8, r=0.95, prior is MQDB countries with at least 1 SFP; UNTRACKED
    priorMean = -3.5
    priorVar = 1.5
    s, r = 0.8, 0.95

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Province', '\nSenegal - Province'])
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
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_laplace(mu=priorMean, scale=np.sqrt(priorVar / 2)), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    # util.plotPostSamples(lgDict, 'int90',
    #                     subTitleStr=['\nSenegal - Facility Location', '\nSenegal - Facility Location'])
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
    lgDict.update({'diagSens': s, 'diagSpec': r, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal - Facility Name', '\nSenegal - Facility Name'])












    # Rerun the 2010 data using untracked data; use N to estimate a sourcing probability matrix, Q
    import numpy as np
    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    Q = lgDict['N'].copy() # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal [Untracked] - Province', '\nSenegal [Untracked] - Province'])
    SNinds = lgDict['importerNames'].index('Aurobindo Pharmaceuticals Ltd')
    print('Aurobindo Pharmaceuticals Ltd: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    SNinds = lgDict['importerNames'].index('Macleods Pharmaceuticals Ltd')
    print('Macleods Pharmaceuticals Ltd: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    SNinds = lgDict['importerNames'].index('Lupin Limited')
    print('Lupin Limited: (' + str(np.quantile(lgDict['postSamples'][:, SNinds], 0.05))[:5] + ',' + str(
        np.quantile(lgDict['postSamples'][:, SNinds], 0.95))[:5] + ')')
    TNinds = lgDict['outletNames'].index('Dakar')
    print('Dakar: (' + str(np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.05))[
                       :5] + ',' + str(np.quantile(lgDict['postSamples'][:, len(lgDict['importerNames']) + TNinds], 0.95))[:5] + ')')

    lgDict = util.testresultsfiletotable(tbl_SEN_G2_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # np.count_nonzero(Q)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal [Untracked] - Facility Location', '\nSenegal [Untracked] - Facility Location'])

    lgDict = util.testresultsfiletotable(tbl_SEN_G3_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal [Untracked] - Facility Name',
                                                       '\nSenegal [Untracked] - Facility Name'])

    # Rerun the 2010 data using untracked data; use N to estimate a sourcing probability matrix, Q, and add a
    # "flattening" parameter to Q to make the sourcing probabilities less sharp
    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Add a constant across Q
    flatParam = 0.05
    Q = Q + flatParam
    for i, Qrow in enumerate(Q):
        Q[i] = Qrow / np.sum(Qrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    util.plotPostSamples(lgDict, 'int90',
                         subTitleStr=['\nSenegal [Untracked] - Province', '\nSenegal [Untracked] - Province'])

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Add a constant across Q
    flatParam = 0.01
    Q = Q + flatParam
    for i, Qrow in enumerate(Q):
        Q[i] = Qrow / np.sum(Qrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    util.plotPostSamples(lgDict, 'int90',
                         subTitleStr=['\nSenegal [Untracked] - Province', '\nSenegal [Untracked] - Province'])

    lgDict = util.testresultsfiletotable(tbl_SEN_G2_2010, csvName=False)
    Q = lgDict['N'].copy()  # Generate Q
    for i, Nrow in enumerate(lgDict['N']):
        Q[i] = Nrow / np.sum(Nrow)
    # Add a constant across Q
    flatParam = 0.02
    Q = Q + flatParam
    for i, Qrow in enumerate(Q):
        Q[i] = Qrow / np.sum(Qrow)
    # Update N and Y
    lgDict.update({'N': np.sum(lgDict['N'], axis=1), 'Y': np.sum(lgDict['Y'], axis=1)})
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'type': 'Untracked', 'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict,
                   'transMat': Q, 'importerNum': Q.shape[1], 'outletNum': Q.shape[0]})
    lgDict = methods.GeneratePostSamples(lgDict)
    util.plotPostSamples(lgDict, 'int90',
                         subTitleStr=['\nSenegal [Untracked] - Province', '\nSenegal [Untracked] - Province'])

    # Rerun 2010 data using different testing tool accuracy
    newSens, newSpec = 0.8, 0.95
    #newSens, newSpec = 0.6, 0.9

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': newSens, 'diagSpec': newSpec, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    subTitle = '\nSenegal [s=' + str(newSens) + ',r=' + str(newSpec) + '] - Province'
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[subTitle, subTitle])

    lgDict = util.testresultsfiletotable(tbl_SEN_G2_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': newSens, 'diagSpec': newSpec, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    subTitle = '\nSenegal [s=' + str(newSens) + ',r=' + str(newSpec) + '] - Facility Location'
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[subTitle, subTitle])

    lgDict = util.testresultsfiletotable(tbl_SEN_G3_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': newSens, 'diagSpec': newSpec, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    subTitle = '\nSenegal [s=' + str(newSens) + ',r=' + str(newSpec) + '] - Facility Name'
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[subTitle, subTitle])

    # Look at sensitivity to prior selection
    priorMean, priorVar = -0.7, 2 # 0.7,2 is from Ozawa Africa SFP studies with n>149 samples

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean,var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=['\nSenegal [mu=' + str(priorMean) + ',var=' + str(priorVar) + '] - Province',
                                      '\nSenegal [mu=-2,var=1] - Province'])
    macleodInd = lgDict['importerNames'].index('Macleods Pharmaceuticals Ltd')
    np.quantile(lgDict['postSamples'][:, macleodInd], 0.05)
    np.quantile(lgDict['postSamples'][:, macleodInd], 0.95)
    np.quantile(lgDict['postSamples'][:, macleodInd], 0.05)
    np.quantile(lgDict['postSamples'][:, macleodInd], 0.95)



    priorMean, priorVar = -2., 5.

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90',
                         subTitleStr=['\nSenegal [mu=' + str(priorMean) + ',var=' + str(priorVar) + '] - Province',
                                      '\nSenegal [mu=-2,var=1] - Province'])

    priorMean, priorVar = -1., 5.

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90',
                         subTitleStr=['\nSenegal [mu=' + str(priorMean) + ',var=' + str(priorVar) + '] - Province',
                                      '\nSenegal [mu=-2,var=1] - Province'])

    priorMean, priorVar = -1., -1.

    lgDict = util.testresultsfiletotable(tbl_SEN_G1_2010, csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(
        lgDict['Y'].sum() / lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': numPostSamps,
                   'prior': methods.prior_normal(mu=priorMean, var=priorVar), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90',
                         subTitleStr=['\nSenegal [mu=' + str(priorMean) + ',var=' + str(priorVar) + '] - Province',
                                      '\nSenegal [mu=-2,var=1] - Province'])


    ##### END SENEGAL #####

    ##### THAILAND #####
    lgDict = util.testresultsfiletotable(dataDict['tbl_THA_G1'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Thailand', ', Thailand'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_THA_G2'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Thailand', ', Thailand'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_THA_G3'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Thailand', ', Thailand'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_THA_G1_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Thailand', ', Thailand'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_THA_G2_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Thailand', ', Thailand'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_THA_G3_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Thailand', ', Thailand'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_THA_G1_antibiotic'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Thailand', ', Thailand'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_THA_G2_antibiotic'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Thailand', ', Thailand'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_THA_G3_antibiotic'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Thailand', ', Thailand'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_THA_G1_antimalarial'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Thailand', ', Thailand'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_THA_G2_antimalarial'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Thailand', ', Thailand'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_THA_G3_antimalarial'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Thailand', ', Thailand'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_THA_G1_antibiotic_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Thailand', ', Thailand'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_THA_G2_antibiotic_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Thailand', ', Thailand'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_THA_G3_antibiotic_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Thailand', ', Thailand'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_THA_G1_antimalarial_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Thailand', ', Thailand'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_THA_G2_antimalarial_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Thailand', ', Thailand'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_THA_G3_antimalarial_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Thailand', ', Thailand'])
    ##### END THAILAND #####

    ##### VIETNAM #####
    lgDict = util.testresultsfiletotable(dataDict['tbl_VIE_G1'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Vietnam', ', Vietnam'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_VIE_G2'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Vietnam', ', Vietnam'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_VIE_G3'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Vietnam', ', Vietnam'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_VIE_G1_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Vietnam', ', Vietnam'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_VIE_G2_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Vietnam', ', Vietnam'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_VIE_G3_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Vietnam', ', Vietnam'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_VIE_G1_antibiotic'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Vietnam', ', Vietnam'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_VIE_G2_antibiotic'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Vietnam', ', Vietnam'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_VIE_G3_antibiotic'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Vietnam', ', Vietnam'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_VIE_G1_antimalarial'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Vietnam', ', Vietnam'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_VIE_G2_antimalarial'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Vietnam', ', Vietnam'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_VIE_G3_antimalarial'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Vietnam', ', Vietnam'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_VIE_G1_antibiotic_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Vietnam', ', Vietnam'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_VIE_G2_antibiotic_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Vietnam', ', Vietnam'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_VIE_G3_antibiotic_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Vietnam', ', Vietnam'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_VIE_G1_antimalarial_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Vietnam', ', Vietnam'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_VIE_G2_antimalarial_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Vietnam', ', Vietnam'])

    lgDict = util.testresultsfiletotable(dataDict['tbl_VIE_G3_antimalarial_ff'], csvName=False)
    print('size: ' + str(lgDict['N'].shape) + ', obsvns: ' + str(lgDict['N'].sum()) + ', propor pos: ' + str(lgDict['Y'].sum()/lgDict['N'].sum()))
    lgDict.update({'diagSens': 1.0, 'diagSpec': 1.0, 'numPostSamples': 500,
                   'prior': methods.prior_normal(mu=priorMean), 'MCMCdict': MCMCdict})
    lgDict = lg.runlogistigate(lgDict)
    util.plotPostSamples(lgDict, 'int90', subTitleStr=[', Vietnam', ', Vietnam'])

    ##### END VIETNAM #####

    return