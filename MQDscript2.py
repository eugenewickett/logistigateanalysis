# -*- coding: utf-8 -*-
'''
Script that analyzes a more complete version of the MQDB data, where more supply-chain features are associated with
each PMS data point.

'''

from logistigate.logistigate import utilities as util # Pull from the submodule "develop" branch
from logistigate.logistigate import methods
from logistigate.logistigate import lg


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
    MQD_df = pd.read_csv(os.path.join(filesPath,'MQDB_Master_Expanded.csv'),low_memory=False) # Main raw database file

    #Change 'pass' to 'Pass'
    MQD_df.loc[MQD_df.Final_Test_Conclusion=='pass','Final_Test_Conclusion'] = 'Pass'

    #Drop 'Province', which has NULL values for some reason; keep 'Province_Name'
    MQD_df = MQD_df.drop('Province', axis=1)

    # Remove 'Guatemala', 'Bolivia', 'Colombia', 'Ecuador', 'Guyana', and 'Yunnan China' due to low sample size
    dropCountries = ['Bolivia', 'Colombia', 'Ecuador', 'Guatemala', 'Guyana', 'Yunnan China']
    MQD_df = MQD_df[~MQD_df['Country_Name'].isin(dropCountries)]

    # By collection year; need to add a year column
    MQD_df['Year_Sample_Collected'] = pd.DatetimeIndex(MQD_df['Date_Sample_Collected']).year
    MQD_df.pivot_table(index=['Country_Name','Year_Sample_Collected'], columns=['Final_Test_Conclusion'], aggfunc='size', fill_value=0)

    # Retitle 'Manufacturer_Name' to 'Manufacturer' to synthesize with old code
    MQD_df= MQD_df.rename(columns={'Manufacturer_Name':'Manufacturer'})

    #import matplotlib.pyplot as plt
    #plt.plot(piv.values,piv.index.tolist())

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
        'Facility_Location_EDIT'] = 'Anlong Veng District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Bakan District')
        | (MQD_df_CAM.Facility_Location == ''),
        'Facility_Location_EDIT'] = 'Bakan District'
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
        'Facility_Location_EDIT'] = 'Banlung District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Battambang City')
        | (MQD_df_CAM.Facility_Location == 'Battambang city')
        | (MQD_df_CAM.Facility_Location == 'Maung Reussy Dist. Battambang province'),
        'Facility_Location_EDIT'] = 'Battambang City'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Borkeo District')
        | (MQD_df_CAM.Facility_Location == 'Borkeo district')
        | (MQD_df_CAM.Facility_Location == 'Cabinet-Keo Akara, near Borkeo Market')
        | (MQD_df_CAM.Facility_Location == 'Midwife- Saren, near Borkeo Market'),
        'Facility_Location_EDIT'] = 'Borkeo District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Cham Knan District')
        | (MQD_df_CAM.Facility_Location == ''),
        'Facility_Location_EDIT'] = 'Cham Knan District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Chamroeun District')
        | (MQD_df_CAM.Facility_Location == 'Cham Roeun District'),
        'Facility_Location_EDIT'] = 'Chamroeun District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'National Road No.4, Chbamon district')
        | (MQD_df_CAM.Facility_Location == 'Chbamon District')
        | (MQD_df_CAM.Facility_Location == 'No. 3B, Peanich Kam village, Roka Thom commune, Chbamon district')
        | (MQD_df_CAM.Facility_Location == 'Peanichkam village, Roka Thom commune, Chbamon District')
        | (MQD_df_CAM.Facility_Location == 'Roka Thom Commune, Chbamon District')
        | (MQD_df_CAM.Facility_Location == '#01D, Psar Kampong Speu, Chbamon district'),
        'Facility_Location_EDIT'] = 'Chbamon District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Chheb District')
        | (MQD_df_CAM.Facility_Location == ''),
        'Facility_Location_EDIT'] = 'Chheb District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Chom Ksan District')
        | (MQD_df_CAM.Facility_Location == 'O Chhounh Village, Chom Ksan district')
        | (MQD_df_CAM.Facility_Location == 'Sra Em village, Chom Ksan District'),
        'Facility_Location_EDIT'] = 'Chom Ksan District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Choran Ksan District')
        | (MQD_df_CAM.Facility_Location == ''),
        'Facility_Location_EDIT'] = 'Choran Ksan District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Dongtong District')
        | (MQD_df_CAM.Facility_Location == 'Dongtong Market')
        | (MQD_df_CAM.Facility_Location == 'No. 50, South of Dongtong Market'),
        'Facility_Location_EDIT'] = 'Dongtong District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Kampong Bay District')
        | (MQD_df_CAM.Facility_Location == 'No. 93, St. 3, Kampong Bay district')
        | (MQD_df_CAM.Facility_Location == 'No. 5, St. 3, Kampong Bay district')
        | (MQD_df_CAM.Facility_Location == 'Kampong Bay district')
        | (MQD_df_CAM.Facility_Location == '#79, Kampong Bay district')
        | (MQD_df_CAM.Facility_Location == '#16, St. 7 Makara, Kandal village, Kampong Bay district'),
        'Facility_Location_EDIT'] = 'Kampong Bay District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Kampong Cham City')
        | (MQD_df_CAM.Facility_Location == 'Kampong Cham District')
        | (MQD_df_CAM.Facility_Location == 'Memot Market, Kampong Cham')
        | (MQD_df_CAM.Facility_Location == 'Steung Market, Kampong Cham')
        | (MQD_df_CAM.Facility_Location == 'Street Preah Bath Ang Duong (East Phsar Thom), Kampong Cham')
        | (MQD_df_CAM.Facility_Location == 'Street Preah Bath Ang Duong (Near Kosona Bridge), Kampong Cham'),
        'Facility_Location_EDIT'] = 'Kampong Cham District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Kampong Siam District')
        | (MQD_df_CAM.Facility_Location == 'Kampong Siam district'),
        'Facility_Location_EDIT'] = 'Kampong Siam District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Kampong Thmor Market')
        | (MQD_df_CAM.Facility_Location == '#66, Tral village, Kompong Thmor market')
        | (MQD_df_CAM.Facility_Location == '66, Tral village, Kompong Thmor market')
        | (MQD_df_CAM.Facility_Location == 'No. 3, Rd 6A, Kampong Thmor'),
        'Facility_Location_EDIT'] = 'Kampong Thmor Market'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Kampong Thom Capital')
        | (MQD_df_CAM.Facility_Location == 'Kampong Thom Market, Kampong Thom capital')
        | (MQD_df_CAM.Facility_Location == 'No. 15Eo, Kampong Thom market, Kampong Thom capital')
        | (MQD_df_CAM.Facility_Location == 'No. 43, Rd No. 6, Kampong Thom capital')
        | (MQD_df_CAM.Facility_Location == 'No. 9 Eo, Kampong Thom Market, Kampong Thom capital')
        | (MQD_df_CAM.Facility_Location == 'No.45, Rd No. 6, Kampong Thom capital'),
        'Facility_Location_EDIT'] = 'Kampong Thom Capital'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Kampong Trach District')
        | (MQD_df_CAM.Facility_Location == 'Kampong Trach Village, Kampong Trach district'),
        'Facility_Location_EDIT'] = 'Kampong Trach District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Keo Seima District')
        | (MQD_df_CAM.Facility_Location == 'Keoseima District')
        | (MQD_df_CAM.Facility_Location == 'Keoseima district')
        | (MQD_df_CAM.Facility_Location == 'Keosema District')
        | (MQD_df_CAM.Facility_Location == "Khum Sre Kh'tob, Keo Seima district"),
        'Facility_Location_EDIT'] = 'Keo Seima District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Koh Kong District')
        | (MQD_df_CAM.Facility_Location == 'Koh Kong Province')
        | (MQD_df_CAM.Facility_Location == 'Kohk Kong Capital')
        | (MQD_df_CAM.Facility_Location == "Pum trorpeagh , Sre'ambel  ,  koh kong  province."),
        'Facility_Location_EDIT'] = 'Koh Kong District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Kolen District')
        | (MQD_df_CAM.Facility_Location == 'Sro Yang Village, Sro Yang Commune, Kolen District'),
        'Facility_Location_EDIT'] = 'Kolen District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Krakor District')
        | (MQD_df_CAM.Facility_Location == 'Chheutom Commune, Krakor District'),
        'Facility_Location_EDIT'] = 'Krakor District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Kratie District')
        | (MQD_df_CAM.Facility_Location == 'Kratie commune, Kratie Distrist, Kratie'),
        'Facility_Location_EDIT'] = 'Kratie District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Maung Russei District')
        | (MQD_df_CAM.Facility_Location == ''),
        'Facility_Location_EDIT'] = 'Maung Russei District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Memot District')
        | (MQD_df_CAM.Facility_Location == 'Khum Dar, Memot District')
        | (MQD_df_CAM.Facility_Location == 'OD Memut'),
        'Facility_Location_EDIT'] = 'Memot District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'O Tavao District')
        | (MQD_df_CAM.Facility_Location == "Krachab, O'Tavao District")
        | (MQD_df_CAM.Facility_Location == "Krachab, O'Tavao, Pailin"),
        'Facility_Location_EDIT'] = 'O Tavao District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Oyadav District')
        | (MQD_df_CAM.Facility_Location == ''),
        'Facility_Location_EDIT'] = 'Oyadav District'
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
        'Facility_Location_EDIT'] = 'Pailin City'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Peamror District')
        | (MQD_df_CAM.Facility_Location == '#309, Prek Khsay commune, Peamror district')
        | (MQD_df_CAM.Facility_Location == 'National Road N0.1, Prek Khsay commune, Peamror district'),
        'Facility_Location_EDIT'] = 'Peamror District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Pearing District')
        | (MQD_df_CAM.Facility_Location == 'St. 8A. Roka commune, Pearing district'),
        'Facility_Location_EDIT'] = 'Pearing District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Phnom Kravanh District')
        | (MQD_df_CAM.Facility_Location == 'Leach village, Phnom Kravanh district')
        | (MQD_df_CAM.Facility_Location == 'Phnom Kravanh District')
        | (MQD_df_CAM.Facility_Location == 'Phnom Krovanh District'),
        'Facility_Location_EDIT'] = 'Phnom Kravanh District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Phnom Preal District')
        | (MQD_df_CAM.Facility_Location == 'Kondamrey - Phnom Preal')
        | (MQD_df_CAM.Facility_Location == 'Koun Domrey, Phnom Preal District')
        | (MQD_df_CAM.Facility_Location == 'O dontaleu Phnom Preal District')
        | (MQD_df_CAM.Facility_Location == 'Phnom Preal')
        | (MQD_df_CAM.Facility_Location == 'Phnom Preal District'),
        'Facility_Location_EDIT'] = 'Phnom Preal District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Ponhea Krek District')
        | (MQD_df_CAM.Facility_Location == 'Ponhea Krek District'),
        'Facility_Location_EDIT'] = 'Ponhea Krek District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Posat City')
        | (MQD_df_CAM.Facility_Location == 'Peal Nhek 2, Posat City'),
        'Facility_Location_EDIT'] = 'Posat City'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Preah Vihear Town')
        | (MQD_df_CAM.Facility_Location == 'Preah Vihear Province')
        | (MQD_df_CAM.Facility_Location == 'Preah Vihear Town')
        | (MQD_df_CAM.Facility_Location == 'Preah Vihear Town')
        | (MQD_df_CAM.Facility_Location == 'Preah Vihear Town'),
        'Facility_Location_EDIT'] = 'Preah Vihear Town'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Prey Chhor District')
        | (MQD_df_CAM.Facility_Location == 'OD Prey Chhor')
        | (MQD_df_CAM.Facility_Location == 'Phsar Prey Toteng, Prey Chhor District'),
        'Facility_Location_EDIT'] = 'Prey Chhor District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Prey Veng District')
        | (MQD_df_CAM.Facility_Location == '#26A, St.15, Kampong Leav commune, Prey Veng district')
        | (MQD_df_CAM.Facility_Location == '#36, St. 15, Kampong Leav commune, Prey Veng district'),
        'Facility_Location_EDIT'] = 'Prey Veng District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Pursat City')
        | (MQD_df_CAM.Facility_Location == 'Peal Nhek 2, Pursat City')
        | (MQD_df_CAM.Facility_Location == 'Phum Piel Nhek, Pursat')
        | (MQD_df_CAM.Facility_Location == 'Village Peal Nhek 2, Pursat City'),
        'Facility_Location_EDIT'] = 'Pursat City'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Roveahek District')
        | (MQD_df_CAM.Facility_Location == 'Angkor Prosre commune, Roveahek district')
        | (MQD_df_CAM.Facility_Location == 'Kampong Trach commune, Roveahek district'),
        'Facility_Location_EDIT'] = 'Roveahek District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Rovieng District')
        | (MQD_df_CAM.Facility_Location == 'Roveing District')
        | (MQD_df_CAM.Facility_Location == 'Ro Vieng District, Tel.: 012 24 82 65')
        | (MQD_df_CAM.Facility_Location == 'Ro Vieng District')
        | (MQD_df_CAM.Facility_Location == 'Ro Veing District')
        | (MQD_df_CAM.Facility_Location == 'Rovieng District')
        | (MQD_df_CAM.Facility_Location == 'Rovieng District'),
        'Facility_Location_EDIT'] = 'Rovieng District'
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
        'Facility_Location_EDIT'] = 'Sala Krau District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Sampov Meas District')
        | (MQD_df_CAM.Facility_Location == ''),
        'Facility_Location_EDIT'] = 'Sampov Meas District'
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
        'Facility_Location_EDIT'] = 'Samraong District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Sangkum Thmei District')
        | (MQD_df_CAM.Facility_Location == 'Sangkomthmey District')
        | (MQD_df_CAM.Facility_Location == 'Sangkum Thmei District')
        | (MQD_df_CAM.Facility_Location == 'Sangkomthmey District, Tel: 011 56 99 26')
        | (MQD_df_CAM.Facility_Location == 'Sangkom Thmei District, Tel.: 011 56 99 26'),
        'Facility_Location_EDIT'] = 'Sangkum Thmei District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Senmonorom City')
        | (MQD_df_CAM.Facility_Location == 'Sangkat Speanmeanchey, Senmonorom City')
        | (MQD_df_CAM.Facility_Location == 'Senmonorom District')
        | (MQD_df_CAM.Facility_Location == 'Senmonorom district'),
        'Facility_Location_EDIT'] = 'Senmonorom City'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Smach Mean Chey District')
        | (MQD_df_CAM.Facility_Location == ''),
        'Facility_Location_EDIT'] = 'Smach Mean Chey District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Sre Ambel District')
        | (MQD_df_CAM.Facility_Location == 'Sre Ambel'),
        'Facility_Location_EDIT'] = 'Sre Ambel District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Village Prek Por Krom, Prek Por Commune, Srey Santhor District')
        | (MQD_df_CAM.Facility_Location == 'Srey Santhor District, Kampong Cham')
        | (MQD_df_CAM.Facility_Location == 'Srey Santhor District')
        | (MQD_df_CAM.Facility_Location == 'Srey Santhor District')
        | (MQD_df_CAM.Facility_Location == 'Prek Por Commune, Srey Santhor District, ')
        | (MQD_df_CAM.Facility_Location == 'Rokar Village, Prek Por Commune, Srey Santhor District'),
        'Facility_Location_EDIT'] = 'Srey Santhor District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Steung Treng Downtown, Tel: 092 251125')
        | (MQD_df_CAM.Facility_Location == 'Steung Treng Downtown, Tel: 092 958707')
        | (MQD_df_CAM.Facility_Location == 'Steung Treng Downtown, Tel: 097 9822096')
        | (MQD_df_CAM.Facility_Location == 'Steung Treng Downtown, Tel:011252525')
        | (MQD_df_CAM.Facility_Location == 'Steung Treng down town')
        | (MQD_df_CAM.Facility_Location == 'Steung Treng downtown, Tel: 017 808287')
        | (MQD_df_CAM.Facility_Location == 'Steung Treng downtown, tel.: 099906174')
        | (MQD_df_CAM.Facility_Location == 'Steung Treng downtown,Tel: 097 90 43 071'),
        'Facility_Location_EDIT'] = 'Steung Treng Downtown'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Suong City')
        | (MQD_df_CAM.Facility_Location == '#65, National Road 7, Soung Commune, Suong City'),
        'Facility_Location_EDIT'] = 'Suong City'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Svay Antor District')
        | (MQD_df_CAM.Facility_Location == 'Pich Chenda commune,Svay Antor district')
        | (MQD_df_CAM.Facility_Location == 'Svay Antor commune, Svay Antor district'),
        'Facility_Location_EDIT'] = 'Svay Antor District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Svay Chrom District')
        | (MQD_df_CAM.Facility_Location == 'National Road No. 1, Crol Kor commune, Svay Chrom district'),
        'Facility_Location_EDIT'] = 'Svay Chrom District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Svay Rieng District')
        | (MQD_df_CAM.Facility_Location == '# 111, Svay Rieng capital')
        | (MQD_df_CAM.Facility_Location == '#1, Rd. 6, Svay Rieng Commune, Svay Rieng district')
        | (MQD_df_CAM.Facility_Location == '#5, Veal Yun Market, Svay Rieng capital')
        | (MQD_df_CAM.Facility_Location == 'Svay Rieng Province')
        | (MQD_df_CAM.Facility_Location == 'Veal Yun market, Svay Rieng capital')
        | (MQD_df_CAM.Facility_Location == 'Svay Rieng District'),
        'Facility_Location_EDIT'] = 'Svay Rieng District'
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
        'Facility_Location_EDIT'] = 'Takeo Capital'
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
        'Facility_Location_EDIT'] = 'Trapeang Prasat District'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Facility_Location == 'Van Sai District')
        | (MQD_df_CAM.Facility_Location == 'Srok Vern sai'),
        'Facility_Location_EDIT'] = 'Van Sai District'

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

    # Facility_Name
    MQD_df_GHA = assignlabels(MQD_df_GHA, 'Facility_Name', thresh=90)
    # Manual adjustments
    # todo: MANUAL ADJUSTMENTS REQUIRED

    # Manufacturer_Name





    MQD_df_GHA.loc[MQD_df_GHA.Manufacturer == 'nan']
    MQD_df_GHA[(MQD_df_GHA.Manufacturer_GROUPED == 'Epharm/Emucure' )].count()
    a = MQD_df_GHA['Facility_Name_GROUPED'].astype('str').unique()
    print(len(a))
    for item in sorted(a):
        print(item)
    MQD_df_GHA.pivot_table(index=['Province_Name_GROUPED'], columns=['Final_Test_Conclusion'], aggfunc='size', fill_value=0)











    # Ghana
    # Province
    MQD_df_GHA.loc[
        (MQD_df_GHA.Province == 'Northern') | (MQD_df_GHA.Province == 'Northern Region')
        | (MQD_df_GHA.Province == 'Northern Region, Northern Region'),
        'Province'] = 'Northern'
    MQD_df_GHA.loc[
        (MQD_df_GHA.Province == 'Western (Ghana)'),
        'Province'] = 'Western'
    # Manufacturer
    MQD_df_GHA.loc[
        (MQD_df_GHA.Manufacturer == 'Ajanta Pharma Ltd') | (MQD_df_GHA.Manufacturer == 'Ajanta Pharma Ltd.'),
        'Manufacturer'] = 'Ajanta Pharma Ltd.'
    MQD_df_GHA.loc[
        (MQD_df_GHA.Manufacturer == 'Ally Pharma Options Pvt Ltd.') | (MQD_df_GHA.Manufacturer == 'Ally Pharma Options Pvt. Ltd'),
        'Manufacturer'] = 'Ally Pharma Options Pvt. Ltd'
    MQD_df_GHA.loc[
        (MQD_df_GHA.Manufacturer == 'Bliss GVS Pharma Ltd') | (MQD_df_GHA.Manufacturer == 'Bliss GVS Pharmaceuticals Ltd.'),
        'Manufacturer'] = 'Bliss GVS Pharma Ltd'
    MQD_df_GHA.loc[
        (MQD_df_GHA.Manufacturer == 'Cipla Ltd. India') | (MQD_df_GHA.Manufacturer == 'Cipla Ltd'),
        'Manufacturer'] = 'Cipla Ltd'
    MQD_df_GHA.loc[
        (MQD_df_GHA.Manufacturer == 'Danadams Pharmaceutical Industry Limited')
        | (MQD_df_GHA.Manufacturer == 'Danadams Pharmaceutical Industry, Ltd.')
        | (MQD_df_GHA.Manufacturer == 'Danadams Pharmaceuticals Industry Limited'),
        'Manufacturer'] = 'Danadams'
    MQD_df_GHA.loc[
        (MQD_df_GHA.Manufacturer == 'Guilin Pharmaceutical Company Ltd.')
        | (MQD_df_GHA.Manufacturer == 'Guilin Pharmaceutical Co. Ltd')
        | (MQD_df_GHA.Manufacturer == 'Guilin  Pharmaceutical Co., Ltd'),
        'Manufacturer'] = 'Guilin'
    MQD_df_GHA.loc[
        (MQD_df_GHA.Manufacturer == 'Kinapharma Limited') | (MQD_df_GHA.Manufacturer == 'Kinapharma Ltd'),
        'Manufacturer'] = 'Kinapharma'
    MQD_df_GHA.loc[
        (MQD_df_GHA.Manufacturer == 'Maphar Laboratories') | (MQD_df_GHA.Manufacturer == 'Maphar'),
        'Manufacturer'] = 'Maphar'
    MQD_df_GHA.loc[
        (MQD_df_GHA.Manufacturer == 'Novartis Pharmaceutical Corporation')
        | (MQD_df_GHA.Manufacturer == 'Novartis Pharmaceuticals Corporation'),
        'Manufacturer'] = 'Novartis'
    MQD_df_GHA.loc[
        (MQD_df_GHA.Manufacturer == 'Pharmanova Limited')
        | (MQD_df_GHA.Manufacturer == 'Pharmanova Ltd'),
        'Manufacturer'] = 'Pharmanova'
    MQD_df_GHA.loc[
        (MQD_df_GHA.Manufacturer == 'Phyto-Riker (Gihoc) Pharmaceuticals Ltd')
        | (MQD_df_GHA.Manufacturer == 'Phyto-Riker (Gihoc) Pharmaceuticals, Ltd.'),
        'Manufacturer'] = 'Phyto-Riker'
    MQD_df_GHA.loc[
        (MQD_df_GHA.Manufacturer == 'Ronak Exim PVT. Ltd')
        | (MQD_df_GHA.Manufacturer == 'Ronak Exim Pvt Ltd'),
        'Manufacturer'] = 'Ronak Exim'

    # Philippines
    # Province
    MQD_df_PHI.loc[(MQD_df_PHI.Province == 'CALABARZON'), 'Province'] = 'Calabarzon'
    MQD_df_PHI.loc[(MQD_df_PHI.Province == 'region 1 '), 'Province'] = 'Region 1'
    MQD_df_PHI.loc[(MQD_df_PHI.Province == 'region7'), 'Province'] = 'Region 7'
    MQD_df_PHI.loc[(MQD_df_PHI.Province == 'region9'), 'Province'] = 'Region 9'
    # Manufacturer
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer == 'AM-Europharma')
                   | (MQD_df_PHI.Manufacturer == 'Am-Euro Pharma Corporation'),
                   'Manufacturer'] = 'AM-Europharma'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer == 'Amherst Laboratories Inc')
                   | (MQD_df_PHI.Manufacturer == 'Amherst Laboratories Inc.')
                   | (MQD_df_PHI.Manufacturer == 'Amherst Laboratories, Inc.'),
                   'Manufacturer'] = 'Amherst'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer == 'Biotech Research Lab Inc.')
                   | (MQD_df_PHI.Manufacturer == 'BRLI'),
                   'Manufacturer'] = 'BRLI'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer == 'Compact Pharmaceutical Corp')
                   | (MQD_df_PHI.Manufacturer == 'Compact Pharmaceutical Corp.')
                   | (MQD_df_PHI.Manufacturer == 'Compact Pharmaceutical Corporation'),
                   'Manufacturer'] = 'Compact'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer == 'Diamond Laboratorie, Inc. ')
                   | (MQD_df_PHI.Manufacturer == 'Diamond Laboratories, Inc.'),
                   'Manufacturer'] = 'Diamond'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer == 'Drugmakers Biotech Research Laboratories, Inc.')
                   | (MQD_df_PHI.Manufacturer == 'Drugmakers Laboratories Inc')
                   | (MQD_df_PHI.Manufacturer == 'Drugmakers Laboratories Inc.')
                   | (MQD_df_PHI.Manufacturer == 'Drugmakers Laboratories, Inc.'),
                   'Manufacturer'] = 'Drugmakers'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer == 'Flamingo Pharmaceuticals Ltd')
                   | (MQD_df_PHI.Manufacturer == 'Flamingo Pharmaceuticals Ltd.')
                   | (MQD_df_PHI.Manufacturer == 'Flamingo Pharmaceuticals, Ltd.'),
                   'Manufacturer'] = 'Flamingo'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer == 'Interphil Laboratories')
                   | (MQD_df_PHI.Manufacturer == 'Interphil Laboratories, Inc.')
                   | (MQD_df_PHI.Manufacturer == 'Interphil Laboratories,Inc'),
                   'Manufacturer'] = 'Interphil'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer == 'J.M. Tolman Laboratories, Inc.')
                   | (MQD_df_PHI.Manufacturer == 'J.M. Tolmann Lab. Inc.')
                   | (MQD_df_PHI.Manufacturer == 'J.M. Tolmann Laboratories, Inc.')
                   | (MQD_df_PHI.Manufacturer == 'J.M.Tollman Laboratories Inc.')
                   | (MQD_df_PHI.Manufacturer == 'J.M.Tolmann Laboratories Inc')
                   | (MQD_df_PHI.Manufacturer == 'J.M.Tolmann Laboratories Inc.')
                   | (MQD_df_PHI.Manufacturer == 'Tolmann'),
                   'Manufacturer'] = 'J.M. Tolmann'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer == 'Lloyd Laboratories Inc')
                   | (MQD_df_PHI.Manufacturer == 'Lloyd Laboratories Inc.')
                   | (MQD_df_PHI.Manufacturer == 'Lloyd Laboratories, Inc.'),
                   'Manufacturer'] = 'Lloyd'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer == 'Lumar Pharmaceutical Lab')
                   | (MQD_df_PHI.Manufacturer == 'Lumar Pharmaceutical Lab. ')
                   | (MQD_df_PHI.Manufacturer == 'Lumar Pharmaceutical Laboratory'),
                   'Manufacturer'] = 'Lumar'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer == 'Lupin Limited') | (MQD_df_PHI.Manufacturer == 'Lupin Ltd')
                   | (MQD_df_PHI.Manufacturer == 'Lupin Ltd.'),
                   'Manufacturer'] = 'Lupin'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer == 'Missing') | (MQD_df_PHI.Manufacturer == 'No Information Available')
                   | (MQD_df_PHI.Manufacturer == 'No information'),
                   'Manufacturer'] = 'Unknown'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer == 'Natrapharm') | (MQD_df_PHI.Manufacturer == 'Natrapharm Inc.')
                   | (MQD_df_PHI.Manufacturer == 'Natrapharm, Inc.'),
                   'Manufacturer'] = 'Natrapharm'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer == 'New Myrex Lab., Inc.') | (MQD_df_PHI.Manufacturer == 'New Myrex Laboratories Inc')
                   | (MQD_df_PHI.Manufacturer == 'New Myrex Laboratories Inc.')
                   | (MQD_df_PHI.Manufacturer == 'New Myrex Laboratories, Inc.'),
                   'Manufacturer'] = 'New Myrex'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer == 'Novartis (Bangladesh)') | (MQD_df_PHI.Manufacturer == 'Novartis (Bangladesh) Ltd.')
                   | (MQD_df_PHI.Manufacturer == 'Novartis Bangladesh Ltd')
                   | (MQD_df_PHI.Manufacturer == 'Novartis Bangladesh Ltd.')
                   | (MQD_df_PHI.Manufacturer == 'Novartis'),
                   'Manufacturer'] = 'Novartis'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer == 'Pascual Lab. Inc.')
                   | (MQD_df_PHI.Manufacturer == 'Pascual Laboratories, Inc.'),
                   'Manufacturer'] = 'Pascual'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer == 'Pharex Health Corp.')
                   | (MQD_df_PHI.Manufacturer == 'Pharex'),
                   'Manufacturer'] = 'Pharex'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer == 'Plethico Pharmaceutical Ltd.')
                   | (MQD_df_PHI.Manufacturer == 'Plethico Pharmaceuticals, Ltd.'),
                   'Manufacturer'] = 'Plethico'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer == 'San Marino Lab., Corp.')
                   | (MQD_df_PHI.Manufacturer == 'San Marino Laboratories Corp'),
                   'Manufacturer'] = 'San Marino'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer == 'Sandoz South Africa Ltd.')
                   | (MQD_df_PHI.Manufacturer == 'Sandoz Private Ltd.')
                   | (MQD_df_PHI.Manufacturer == 'Sandoz Philippines Corp.')
                   | (MQD_df_PHI.Manufacturer == 'Sandoz GmbH')
                   | (MQD_df_PHI.Manufacturer == 'Sandoz'),
                   'Manufacturer'] = 'Sandoz'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer == 'Scheele Laboratories Phil., Inc.')
                   | (MQD_df_PHI.Manufacturer == 'Scheele Laboratories Phils, Inc.')
                   | (MQD_df_PHI.Manufacturer == 'Scheele Laboratories Phis., Inc.')
                   | (MQD_df_PHI.Manufacturer == 'Scheele Laboratories Phils, Inc.'),
                   'Manufacturer'] = 'Scheele'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer == 'The Generics Pharmacy')
                   | (MQD_df_PHI.Manufacturer == 'The Generics Pharmacy Inc.')
                   | (MQD_df_PHI.Manufacturer == 'TGP'),
                   'Manufacturer'] = 'TGP'
    MQD_df_PHI.loc[(MQD_df_PHI.Manufacturer == 'Wyeth Pakistan Limited')
                   | (MQD_df_PHI.Manufacturer == 'Wyeth Pakistan Ltd')
                   | (MQD_df_PHI.Manufacturer == 'Wyeth Pakistan Ltd.'),
                   'Manufacturer'] = 'Wyeth'

    # Make smaller data frames filtered for facility type and therapeutic indication
    # Filter for facility type
    MQD_df_CAM_filt = MQD_df_CAM[MQD_df_CAM['Facility Type'].isin(
        ['Depot of Pharmacy', 'Health Clinic', 'Pharmacy', 'Pharmacy Depot', 'Private Clinic',
         'Retail-drug Outlet', 'Retail drug outlet', 'Clinic'])].copy()
    MQD_df_GHA_filt = MQD_df_GHA[MQD_df_GHA['Facility Type'].isin(
        ['Health Clinic', 'Hospital', 'Pharmacy', 'Retail Shop', 'Retail-drug Outlet'])].copy()
    MQD_df_PHI_filt = MQD_df_PHI[MQD_df_PHI['Facility Type'].isin(
        ['Health Center', 'Health Clinic', 'Hospital', 'Hospital Pharmacy', 'Pharmacy',
         'Retail-drug Outlet', 'health office'])].copy()
    # Now filter by chosen drug types
    MQD_df_CAM_antimalarial = MQD_df_CAM_filt[MQD_df_CAM_filt['Therapeutic Indications'].isin(['Antimalarial'])].copy()
    MQD_df_GHA_antimalarial = MQD_df_GHA_filt[MQD_df_GHA_filt['Therapeutic Indications'].isin(['Antimalarial',
                                                                                               'Antimalarials'])].copy()
    MQD_df_PHI_antituberculosis = MQD_df_PHI_filt[MQD_df_PHI_filt['Therapeutic Indications'].isin(['Anti-tuberculosis',
                                                                                               'Antituberculosis'])].copy()
    # For each desired data set, generate lists suitable for use with logistigate
    # Overall data
    dataTbl_CAM = MQD_df_CAM[['Province', 'Manufacturer', 'Final Test Result']].values.tolist()
    dataTbl_CAM = [[i[0],i[1],1] if i[2]=='Fail' else [i[0],i[1],0] for i in dataTbl_CAM]
    dataTbl_GHA = MQD_df_GHA[['Province', 'Manufacturer', 'Final Test Result']].values.tolist()
    dataTbl_GHA = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_GHA]
    dataTbl_PHI = MQD_df_PHI[['Province', 'Manufacturer', 'Final Test Result']].values.tolist()
    dataTbl_PHI = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_PHI]
    # Filtered data
    dataTbl_CAM_filt = MQD_df_CAM_filt[['Province', 'Manufacturer', 'Final Test Result']].values.tolist()
    dataTbl_CAM_filt = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_CAM_filt]
    dataTbl_GHA_filt = MQD_df_GHA_filt[['Province', 'Manufacturer', 'Final Test Result']].values.tolist()
    dataTbl_GHA_filt = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_GHA_filt]
    dataTbl_PHI_filt = MQD_df_PHI_filt[['Province', 'Manufacturer', 'Final Test Result']].values.tolist()
    dataTbl_PHI_filt = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_PHI_filt]
    # Therapeutics data
    dataTbl_CAM_antimalarial = MQD_df_CAM_antimalarial[['Province', 'Manufacturer', 'Final Test Result']].values.tolist()
    dataTbl_CAM_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_CAM_antimalarial]
    dataTbl_GHA_antimalarial = MQD_df_GHA_antimalarial[['Province', 'Manufacturer', 'Final Test Result']].values.tolist()
    dataTbl_GHA_antimalarial = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_GHA_antimalarial]
    dataTbl_PHI_antituberculosis = MQD_df_PHI_antituberculosis[['Province', 'Manufacturer', 'Final Test Result']].values.tolist()
    dataTbl_PHI_antituberculosis = [[i[0], i[1], 1] if i[2] == 'Fail' else [i[0], i[1], 0] for i in dataTbl_PHI_antituberculosis]
    # Put the databases and lists into a dictionary
    outputDict = {}
    outputDict.update({'df_ALL':MQD_df,
                       'df_CAM':MQD_df_CAM, 'df_GHA':MQD_df_GHA, 'df_PHI':MQD_df_PHI,
                       'df_CAM_filt':MQD_df_CAM_filt, 'df_GHA_filt':MQD_df_GHA_filt, 'df_PHI_filt':MQD_df_PHI_filt,
                       'df_CAM_antimalarial':MQD_df_CAM_antimalarial, 'df_GHA_antimalarial':MQD_df_GHA_antimalarial,
                       'df_PHI_antituberculosis':MQD_df_PHI_antituberculosis,
                       'dataTbl_CAM':dataTbl_CAM, 'dataTbl_GHA':dataTbl_GHA, 'dataTbl_PHI':dataTbl_PHI,
                       'dataTbl_CAM_filt':dataTbl_CAM_filt, 'dataTbl_GHA_filt':dataTbl_GHA_filt,
                       'dataTbl_PHI_filt':dataTbl_PHI_filt, 'dataTbl_CAM_antimalarial':dataTbl_CAM_antimalarial,
                       'dataTbl_GHA_antimalarial':dataTbl_GHA_antimalarial,
                       'dataTbl_PHI_antituberculosis':dataTbl_PHI_antituberculosis})

    return outputDict

def MQDdataScript():
    '''Script looking at the MQD data'''
    import scipy.special as sps
    import numpy as np
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, 'logistigate', 'exmples','data')))

    # Grab processed data tables
    dataDict = cleanMQD()

    # Run with Country as outlets
    dataTblDict = util.testresultsfiletotable('MQDfiles/MQD_TRIMMED1')
    dataTblDict.update({'diagSens': 1.0,
                        'diagSpec': 1.0,
                        'numPostSamples': 500,
                        'prior': methods.prior_normal(mu=sps.logit(0.038)),
                        'MCMCdict': MCMCdict})
    logistigateDict = lg.runlogistigate(dataTblDict)

    util.plotPostSamples(logistigateDict)
    util.printEstimates(logistigateDict)

    # Run with Country-Province as outlets
    dataTblDict2 = util.testresultsfiletotable('MQDfiles/MQD_TRIMMED2.csv')
    dataTblDict2.update({'diagSens': 1.0,
                        'diagSpec': 1.0,
                        'numPostSamples': 500,
                        'prior': methods.prior_normal(mu=sps.logit(0.038)),
                        'MCMCdict': MCMCdict})
    logistigateDict2 = lg.runlogistigate(dataTblDict2)

    util.plotPostSamples(logistigateDict2)
    util.printEstimates(logistigateDict2)

    # Run with Cambodia provinces
    dataTblDict_CAM = util.testresultsfiletotable(dataDict['dataTbl_CAM'], csvName=False)
    countryMean = np.sum(dataTblDict_CAM['Y']) / np.sum(dataTblDict_CAM['N'])
    dataTblDict_CAM.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 1000,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_CAM = lg.runlogistigate(dataTblDict_CAM)
    numCamImps_fourth = int(np.floor(logistigateDict_CAM['importerNum'] / 4))
    util.plotPostSamples(logistigateDict_CAM, plotType='int90',
                         importerIndsSubset=np.arange(numCamImps_fourth).tolist(),
                         subTitleStr=['\nCambodia - 1st Quarter', '\nCambodia'])
    util.plotPostSamples(logistigateDict_CAM, plotType='int90',
                         importerIndsSubset=np.arange(numCamImps_fourth,numCamImps_fourth*2).tolist(),
                         subTitleStr=['\nCambodia - 2nd Quarter', '\nCambodia'])
    util.plotPostSamples(logistigateDict_CAM, plotType='int90',
                         importerIndsSubset=np.arange(numCamImps_fourth * 2, numCamImps_fourth * 3).tolist(),
                         subTitleStr=['\nCambodia - 3rd Quarter', '\nCambodia'])
    util.plotPostSamples(logistigateDict_CAM, plotType='int90',
                         importerIndsSubset=np.arange(numCamImps_fourth * 3, numCamImps_fourth * 4).tolist(),
                         subTitleStr=['\nCambodia - 4th Quarter', '\nCambodia'])
    util.printEstimates(logistigateDict_CAM)
    # Plot importers subset where median sample is above 0.4
    totalEntities = logistigateDict_CAM['importerNum'] + logistigateDict_CAM['outletNum']
    sampMedians = [np.median(logistigateDict_CAM['postSamples'][:,i]) for i in range(totalEntities)]
    highImporterInds = [i for i, x in enumerate(sampMedians[:logistigateDict_CAM['importerNum']]) if x > 0.4]
    util.plotPostSamples(logistigateDict_CAM, importerIndsSubset=highImporterInds,subTitleStr=['\nCambodia - Subset','\nCambodia'])
    util.printEstimates(logistigateDict_CAM, importerIndsSubset=highImporterInds)
    # Run with Cambodia provinces filtered for outlet-type samples
    dataTblDict_CAM_filt = util.testresultsfiletotable(dataDict['dataTbl_CAM_filt'], csvName=False)
    #dataTblDict_CAM_filt = util.testresultsfiletotable('MQDfiles/MQD_CAMBODIA_FACILITYFILTER.csv')
    countryMean = np.sum(dataTblDict_CAM_filt['Y']) / np.sum(dataTblDict_CAM_filt['N'])
    dataTblDict_CAM_filt.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 1000,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_CAM_filt = lg.runlogistigate(dataTblDict_CAM_filt)
    numCamImps_fourth = int(np.floor(logistigateDict_CAM_filt['importerNum'] / 4))
    util.plotPostSamples(logistigateDict_CAM_filt, plotType='int90',
                         importerIndsSubset=np.arange(numCamImps_fourth).tolist(),
                         subTitleStr=['\nCambodia (filtered) - 1st Quarter', '\nCambodia (filtered)'])
    util.plotPostSamples(logistigateDict_CAM_filt, plotType='int90',
                         importerIndsSubset=np.arange(numCamImps_fourth, numCamImps_fourth * 2).tolist(),
                         subTitleStr=['\nCambodia (filtered) - 2nd Quarter', '\nCambodia (filtered)'])
    util.plotPostSamples(logistigateDict_CAM_filt, plotType='int90',
                         importerIndsSubset=np.arange(numCamImps_fourth * 2, numCamImps_fourth * 3).tolist(),
                         subTitleStr=['\nCambodia (filtered) - 3rd Quarter', '\nCambodia (filtered)'])
    util.plotPostSamples(logistigateDict_CAM_filt, plotType='int90',
                         importerIndsSubset=np.arange(numCamImps_fourth * 3, logistigateDict_CAM_filt['importerNum']).tolist(),
                         subTitleStr=['\nCambodia (filtered) - 4th Quarter', '\nCambodia (filtered)'])
    # Run with Cambodia provinces filtered for antibiotics
    dataTblDict_CAM_antibiotic = util.testresultsfiletotable('MQDfiles/MQD_CAMBODIA_ANTIBIOTIC.csv')
    countryMean = np.sum(dataTblDict_CAM_antibiotic['Y']) / np.sum(dataTblDict_CAM_antibiotic['N'])
    dataTblDict_CAM_antibiotic.update({'diagSens': 1.0,
                                 'diagSpec': 1.0,
                                 'numPostSamples': 1000,
                                 'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                                 'MCMCdict': MCMCdict})
    logistigateDict_CAM_antibiotic = lg.runlogistigate(dataTblDict_CAM_antibiotic)
    numCamImps_third = int(np.floor(logistigateDict_CAM_antibiotic['importerNum'] / 3))
    util.plotPostSamples(logistigateDict_CAM_antibiotic, plotType='int90',
                         importerIndsSubset=np.arange(numCamImps_third).tolist(),
                         subTitleStr=['\nCambodia - 1st Third (Antibiotics)', '\nCambodia (Antibiotics)'])
    util.plotPostSamples(logistigateDict_CAM_antibiotic, plotType='int90',
                         importerIndsSubset=np.arange(numCamImps_third, numCamImps_third * 2).tolist(),
                         subTitleStr=['\nCambodia - 2nd Third (Antibiotics)', '\nCambodia (Antibiotics)'])
    util.plotPostSamples(logistigateDict_CAM_antibiotic, plotType='int90',
                         importerIndsSubset=np.arange(numCamImps_third * 2, logistigateDict_CAM_antibiotic['importerNum']).tolist(),
                         subTitleStr=['\nCambodia - 3rd Third (Antibiotics)', '\nCambodia (Antibiotics)'])
    util.printEstimates(logistigateDict_CAM_antibiotic)
    # Run with Cambodia provinces filtered for antimalarials
    dataTblDict_CAM_antimalarial = util.testresultsfiletotable(dataDict['dataTbl_CAM_antimalarial'], csvName=False)
    countryMean = np.sum(dataTblDict_CAM_antimalarial['Y']) / np.sum(dataTblDict_CAM_antimalarial['N'])
    dataTblDict_CAM_antimalarial.update({'diagSens': 1.0,
                                       'diagSpec': 1.0,
                                       'numPostSamples': 1000,
                                       'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                                       'MCMCdict': MCMCdict})
    logistigateDict_CAM_antimalarial = lg.runlogistigate(dataTblDict_CAM_antimalarial)

    #numCamImps_half = int(np.floor(logistigateDict_CAM_antimalarial['importerNum'] / 2))
    #util.plotPostSamples(logistigateDict_CAM_antimalarial, plotType='int90',
    #                     importerIndsSubset=np.arange(numCamImps_half).tolist(),
    #                     subTitleStr=['\nCambodia - 1st Half (Antimalarials)', '\nCambodia (Antimalarials)'])
    #util.plotPostSamples(logistigateDict_CAM_antimalarial, plotType='int90',
    #                     importerIndsSubset=np.arange(numCamImps_half,
    #                                                  logistigateDict_CAM_antimalarial['importerNum']).tolist(),
    #                     subTitleStr=['\nCambodia - 2nd Half (Antimalarials)', '\nCambodia (Antimalarials)'])

    # Special plotting for these data sets
    numImp, numOut = logistigateDict_CAM_antimalarial['importerNum'], logistigateDict_CAM_antimalarial['outletNum']
    lowerQuant, upperQuant = 0.05, 0.95
    intStr = '90'
    priorSamps = logistigateDict_CAM_antimalarial['prior'].expitrand(5000)
    priorLower, priorUpper = np.quantile(priorSamps, lowerQuant), np.quantile(priorSamps, upperQuant)
    importerIndsSubset = range(numImp)
    impNames = [logistigateDict_CAM_antimalarial['importerNames'][i] for i in importerIndsSubset]
    impLowers = [np.quantile(logistigateDict_CAM_antimalarial['postSamples'][:, l], lowerQuant) for l in importerIndsSubset]
    impUppers = [np.quantile(logistigateDict_CAM_antimalarial['postSamples'][:, l], upperQuant) for l in importerIndsSubset]
    midpoints = [impUppers[i] - (impUppers[i] - impLowers[i]) / 2 for i in range(len(impUppers))]
    zippedList = zip(midpoints, impUppers, impLowers, impNames)
    sorted_pairs = sorted(zippedList, reverse=True)
    impNamesSorted = [tup[3] for tup in sorted_pairs]
    impNamesSorted.append('')
    impNamesSorted.append('(Prior)')
    # Plot
    import matplotlib.pyplot as plt
    fig, (ax) = plt.subplots(figsize=(10, 10), ncols=1)
    sorted_pairs.append((np.nan, np.nan, np.nan, ' '))  # for spacing
    for _, upper, lower, name in sorted_pairs:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot((impNamesSorted[-1], impNamesSorted[-1]), (priorLower, priorUpper), 'o--', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(impNamesSorted)), impNamesSorted, rotation=90)
    plt.title('Manufacturers - ' + intStr + '% Intervals' + '\nCambodia Antimalarials',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Manufacturer Name', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(9)
    fig.tight_layout()
    plt.show()
    plt.close()

    outletIndsSubset = range(numOut)
    outNames = [logistigateDict_CAM_antimalarial['outletNames'][i] for i in outletIndsSubset]
    outLowers = [np.quantile(logistigateDict_CAM_antimalarial['postSamples'][:, numImp + l], lowerQuant) for l in outletIndsSubset]
    outUppers = [np.quantile(logistigateDict_CAM_antimalarial['postSamples'][:, numImp + l], upperQuant) for l in outletIndsSubset]
    midpoints = [outUppers[i] - (outUppers[i] - outLowers[i]) / 2 for i in range(len(outUppers))]
    zippedList = zip(midpoints, outUppers, outLowers, outNames)
    sorted_pairs = sorted(zippedList, reverse=True)
    outNamesSorted = [tup[3] for tup in sorted_pairs]
    outNamesSorted.append('')
    outNamesSorted.append('(Prior)')
    # Plot
    fig, (ax) = plt.subplots(figsize=(8, 10), ncols=1)
    sorted_pairs.append((np.nan, np.nan, np.nan, ' '))  # for spacing
    for _, upper, lower, name in sorted_pairs:
        plt.plot((name, name), (lower, upper), 'o-', color='purple')
    plt.plot((outNamesSorted[-1], outNamesSorted[-1]), (priorLower, priorUpper), 'o--', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(outNamesSorted)), outNamesSorted, rotation=90)
    plt.title('Regional Aggregates - ' + intStr + '% Intervals' + '\nCambodia Antimalarials',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Regional Aggregate', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(11)
    fig.tight_layout()
    plt.show()
    plt.close()

    util.Summarize(logistigateDict_CAM_antimalarial)

    # Run with Ethiopia provinces
    dataTblDict_ETH = util.testresultsfiletotable('MQDfiles/MQD_ETHIOPIA.csv')
    countryMean = np.sum(dataTblDict_ETH['Y']) / np.sum(dataTblDict_ETH['N'])
    dataTblDict_ETH.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 500,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_ETH = lg.runlogistigate(dataTblDict_ETH)
    util.plotPostSamples(logistigateDict_ETH)
    util.printEstimates(logistigateDict_ETH)

    # Run with Ghana provinces
    dataTblDict_GHA = util.testresultsfiletotable(dataDict['dataTbl_GHA'], csvName=False)
    #dataTblDict_GHA = util.testresultsfiletotable('MQDfiles/MQD_GHANA.csv')
    countryMean = np.sum(dataTblDict_GHA['Y']) / np.sum(dataTblDict_GHA['N'])
    dataTblDict_GHA.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 1000,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_GHA = lg.runlogistigate(dataTblDict_GHA)
    util.plotPostSamples(logistigateDict_GHA, plotType='int90',
                         subTitleStr=['\nGhana', '\nGhana'])
    util.printEstimates(logistigateDict_GHA)
    # Plot importers subset where median sample is above 0.4
    totalEntities = logistigateDict_GHA['importerNum'] + logistigateDict_GHA['outletNum']
    sampMedians = [np.median(logistigateDict_GHA['postSamples'][:, i]) for i in range(totalEntities)]
    highImporterInds = [i for i, x in enumerate(sampMedians[:logistigateDict_GHA['importerNum']]) if x > 0.4]
    highOutletInds = [i for i, x in enumerate(sampMedians[logistigateDict_GHA['importerNum']:]) if x > 0.15]
    util.plotPostSamples(logistigateDict_GHA, importerIndsSubset=highImporterInds,
                         outletIndsSubset=highOutletInds,
                         subTitleStr=['\nGhana - Subset', '\nGhana - Subset'])
    util.printEstimates(logistigateDict_GHA, importerIndsSubset=highImporterInds,outletIndsSubset=highOutletInds)
    # Run with Ghana provinces filtered for outlet-type samples
    dataTblDict_GHA_filt = util.testresultsfiletotable(dataDict['dataTbl_GHA_filt'], csvName=False)
    #dataTblDict_GHA_filt = util.testresultsfiletotable('MQDfiles/MQD_GHANA_FACILITYFILTER.csv')
    countryMean = np.sum(dataTblDict_GHA_filt['Y']) / np.sum(dataTblDict_GHA_filt['N'])
    dataTblDict_GHA_filt.update({'diagSens': 1.0,
                                 'diagSpec': 1.0,
                                 'numPostSamples': 1000,
                                 'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                                 'MCMCdict': MCMCdict})
    logistigateDict_GHA_filt = lg.runlogistigate(dataTblDict_GHA_filt)
    util.plotPostSamples(logistigateDict_GHA_filt, plotType='int90',
                         subTitleStr=['\nGhana (filtered)', '\nGhana (filtered)'])
    util.printEstimates(logistigateDict_GHA_filt)
    # Run with Ghana provinces filtered for antimalarials
    dataTblDict_GHA_antimalarial = util.testresultsfiletotable(dataDict['dataTbl_GHA_antimalarial'], csvName=False)
    #dataTblDict_GHA_antimalarial = util.testresultsfiletotable('MQDfiles/MQD_GHANA_ANTIMALARIAL.csv')
    countryMean = np.sum(dataTblDict_GHA_antimalarial['Y']) / np.sum(dataTblDict_GHA_antimalarial['N'])
    dataTblDict_GHA_antimalarial.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 1000,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_GHA_antimalarial = lg.runlogistigate(dataTblDict_GHA_antimalarial)
    #util.plotPostSamples(logistigateDict_GHA_antimalarial, plotType='int90',
    #                     subTitleStr=['\nGhana (Antimalarials)', '\nGhana (Antimalarials)'])
    #util.printEstimates(logistigateDict_GHA_antimalarial)

    # Special plotting for these data sets
    numImp, numOut = logistigateDict_GHA_antimalarial['importerNum'], logistigateDict_GHA_antimalarial['outletNum']
    lowerQuant, upperQuant = 0.05, 0.95
    intStr = '90'
    priorSamps = logistigateDict_GHA_antimalarial['prior'].expitrand(5000)
    priorLower, priorUpper = np.quantile(priorSamps, lowerQuant), np.quantile(priorSamps, upperQuant)
    importerIndsSubset = range(numImp)
    impNames = [logistigateDict_GHA_antimalarial['importerNames'][i] for i in importerIndsSubset]
    impLowers = [np.quantile(logistigateDict_GHA_antimalarial['postSamples'][:, l], lowerQuant) for l in
                 importerIndsSubset]
    impUppers = [np.quantile(logistigateDict_GHA_antimalarial['postSamples'][:, l], upperQuant) for l in
                 importerIndsSubset]
    midpoints = [impUppers[i] - (impUppers[i] - impLowers[i]) / 2 for i in range(len(impUppers))]
    zippedList = zip(midpoints, impUppers, impLowers, impNames)
    sorted_pairs = sorted(zippedList, reverse=True)
    impNamesSorted = [tup[3] for tup in sorted_pairs]
    impNamesSorted.append('')
    impNamesSorted.append('(Prior)')
    # Plot
    import matplotlib.pyplot as plt
    fig, (ax) = plt.subplots(figsize=(10, 10), ncols=1)
    sorted_pairs.append((np.nan, np.nan, np.nan, ' '))  # for spacing
    for _, upper, lower, name in sorted_pairs:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot((impNamesSorted[-1], impNamesSorted[-1]), (priorLower, priorUpper), 'o--', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(impNamesSorted)), impNamesSorted, rotation=90)
    plt.title('Manufacturers - ' + intStr + '% Intervals' + '\nGhana Antimalarials',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Manufacturer Name', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(9)
    fig.tight_layout()
    plt.show()
    plt.close()

    outletIndsSubset = range(numOut)
    outNames = [logistigateDict_GHA_antimalarial['outletNames'][i][6:] for i in outletIndsSubset]
    outNames[7] = 'Western'
    outLowers = [np.quantile(logistigateDict_GHA_antimalarial['postSamples'][:, numImp + l], lowerQuant) for l in
                 outletIndsSubset]
    outUppers = [np.quantile(logistigateDict_GHA_antimalarial['postSamples'][:, numImp + l], upperQuant) for l in
                 outletIndsSubset]
    midpoints = [outUppers[i] - (outUppers[i] - outLowers[i]) / 2 for i in range(len(outUppers))]
    zippedList = zip(midpoints, outUppers, outLowers, outNames)
    sorted_pairs = sorted(zippedList, reverse=True)
    outNamesSorted = [tup[3] for tup in sorted_pairs]
    outNamesSorted.append('')
    outNamesSorted.append('(Prior)')
    # Plot
    fig, (ax) = plt.subplots(figsize=(8, 10), ncols=1)
    sorted_pairs.append((np.nan, np.nan, np.nan, ' '))  # for spacing
    for _, upper, lower, name in sorted_pairs:
        plt.plot((name, name), (lower, upper), 'o-', color='purple')
    plt.plot((outNamesSorted[-1], outNamesSorted[-1]), (priorLower, priorUpper), 'o--', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(outNamesSorted)), outNamesSorted, rotation=90)
    plt.title('Regional Aggregates - ' + intStr + '% Intervals' + '\nGhana Antimalarials',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Regional Aggregate', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(11)
    fig.tight_layout()
    plt.show()
    plt.close()

    util.Summarize(logistigateDict_GHA_antimalarial)

    # Run with Kenya provinces
    dataTblDict_KEN = util.testresultsfiletotable('MQDfiles/MQD_KENYA.csv')
    countryMean = np.sum(dataTblDict_KEN['Y']) / np.sum(dataTblDict_KEN['N'])
    dataTblDict_KEN.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 500,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_KEN = lg.runlogistigate(dataTblDict_KEN)
    util.plotPostSamples(logistigateDict_KEN)
    util.printEstimates(logistigateDict_KEN)


    # Run with Laos provinces
    dataTblDict_LAO = util.testresultsfiletotable('MQDfiles/MQD_LAOS.csv')
    countryMean = np.sum(dataTblDict_LAO['Y']) / np.sum(dataTblDict_LAO['N'])
    dataTblDict_LAO.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 500,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_LAO = lg.runlogistigate(dataTblDict_LAO)
    util.plotPostSamples(logistigateDict_LAO)
    util.printEstimates(logistigateDict_LAO)


    # Run with Mozambique provinces
    dataTblDict_MOZ = util.testresultsfiletotable('MQDfiles/MQD_MOZAMBIQUE.csv')
    countryMean = np.sum(dataTblDict_MOZ['Y']) / np.sum(dataTblDict_MOZ['N'])
    dataTblDict_MOZ.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 500,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_MOZ = lg.runlogistigate(dataTblDict_MOZ)
    util.plotPostSamples(logistigateDict_MOZ)
    util.printEstimates(logistigateDict_MOZ)

    # Run with Nigeria provinces
    dataTblDict_NIG = util.testresultsfiletotable('MQDfiles/MQD_NIGERIA.csv')
    countryMean = np.sum(dataTblDict_NIG['Y']) / np.sum(dataTblDict_NIG['N'])
    dataTblDict_NIG.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 500,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_NIG = lg.runlogistigate(dataTblDict_NIG)
    util.plotPostSamples(logistigateDict_NIG)
    util.printEstimates(logistigateDict_NIG)

    # Run with Peru provinces
    dataTblDict_PER = util.testresultsfiletotable('MQDfiles/MQD_PERU.csv')
    countryMean = np.sum(dataTblDict_PER['Y']) / np.sum(dataTblDict_PER['N'])
    dataTblDict_PER.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 1000,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_PER = lg.runlogistigate(dataTblDict_PER)
    numPeruImps_half = int(np.floor(logistigateDict_PER['importerNum']/2))
    util.plotPostSamples(logistigateDict_PER, plotType='int90',
                         importerIndsSubset=np.arange(0,numPeruImps_half).tolist(), subTitleStr=['\nPeru - 1st Half', '\nPeru'])
    util.plotPostSamples(logistigateDict_PER, plotType='int90',
                         importerIndsSubset=np.arange(numPeruImps_half,logistigateDict_PER['importerNum']).tolist(),
                         subTitleStr=['\nPeru - 2nd Half', '\nPeru'])
    util.printEstimates(logistigateDict_PER)
    # Plot importers subset where median sample is above 0.4
    totalEntities = logistigateDict_PER['importerNum'] + logistigateDict_PER['outletNum']
    sampMedians = [np.median(logistigateDict_PER['postSamples'][:, i]) for i in range(totalEntities)]
    highImporterInds = [i for i, x in enumerate(sampMedians[:logistigateDict_PER['importerNum']]) if x > 0.4]
    highImporterInds = [highImporterInds[i] for i in [3,6,7,8,9,12,13,16]] # Only manufacturers with more than 1 sample
    highOutletInds = [i for i, x in enumerate(sampMedians[logistigateDict_PER['importerNum']:]) if x > 0.12]
    util.plotPostSamples(logistigateDict_PER, importerIndsSubset=highImporterInds,
                         outletIndsSubset=highOutletInds,
                         subTitleStr=['\nPeru - Subset', '\nPeru - Subset'])
    util.printEstimates(logistigateDict_PER, importerIndsSubset=highImporterInds, outletIndsSubset=highOutletInds)
    # Run with Peru provinces filtered for outlet-type samples
    dataTblDict_PER_filt = util.testresultsfiletotable('MQDfiles/MQD_PERU_FACILITYFILTER.csv')
    countryMean = np.sum(dataTblDict_PER_filt['Y']) / np.sum(dataTblDict_PER_filt['N'])
    dataTblDict_PER_filt.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 1000,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_PER_filt = lg.runlogistigate(dataTblDict_PER_filt)
    numPeruImps_half = int(np.floor(logistigateDict_PER_filt['importerNum'] / 2))
    util.plotPostSamples(logistigateDict_PER_filt, plotType='int90',
                         importerIndsSubset=np.arange(0, numPeruImps_half).tolist(),
                         subTitleStr=['\nPeru - 1st Half (filtered)', '\nPeru (filtered)'])
    util.plotPostSamples(logistigateDict_PER_filt, plotType='int90',
                         importerIndsSubset=np.arange(numPeruImps_half, logistigateDict_PER_filt['importerNum']).tolist(),
                         subTitleStr=['\nPeru - 2nd Half (filtered)', '\nPeru (filtered)'])
    util.printEstimates(logistigateDict_PER_filt)
    # Run with Peru provinces filtered for antibiotics
    dataTblDict_PER_antibiotics = util.testresultsfiletotable('MQDfiles/MQD_PERU_ANTIBIOTIC.csv')
    countryMean = np.sum(dataTblDict_PER_antibiotics['Y']) / np.sum(dataTblDict_PER_antibiotics['N'])
    dataTblDict_PER_antibiotics.update({'diagSens': 1.0,
                                 'diagSpec': 1.0,
                                 'numPostSamples': 1000,
                                 'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                                 'MCMCdict': MCMCdict})
    logistigateDict_PER_antibiotics = lg.runlogistigate(dataTblDict_PER_antibiotics)
    numPeruImps_half = int(np.floor(logistigateDict_PER_antibiotics['importerNum'] / 2))
    util.plotPostSamples(logistigateDict_PER_antibiotics, plotType='int90',
                         importerIndsSubset=np.arange(numPeruImps_half).tolist(),
                         subTitleStr=['\nPeru - 1st Half (Antibiotics)', '\nPeru (Antibiotics)'])
    util.plotPostSamples(logistigateDict_PER_antibiotics, plotType='int90',
                         importerIndsSubset=np.arange(numPeruImps_half, logistigateDict_PER_antibiotics['importerNum']).tolist(),
                         subTitleStr=['\nPeru - 2nd Half (Antibiotics)', '\nPeru (Antibiotics)'])
    util.printEstimates(logistigateDict_PER_antibiotics)

    # Run with Philippines provinces
    dataTblDict_PHI = util.testresultsfiletotable(dataDict['dataTbl_PHI'], csvName=False)
    #dataTblDict_PHI = util.testresultsfiletotable('MQDfiles/MQD_PHILIPPINES.csv')
    countryMean = np.sum(dataTblDict_PHI['Y']) / np.sum(dataTblDict_PHI['N'])
    dataTblDict_PHI.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 1000,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_PHI = lg.runlogistigate(dataTblDict_PHI)
    util.plotPostSamples(logistigateDict_PHI,plotType='int90',subTitleStr=['\nPhilippines','\nPhilippines'])
    util.printEstimates(logistigateDict_PHI)
    # Plot importers subset where median sample is above 0.1
    totalEntities = logistigateDict_PHI['importerNum'] + logistigateDict_PHI['outletNum']
    sampMedians = [np.median(logistigateDict_PHI['postSamples'][:, i]) for i in range(totalEntities)]
    highImporterInds = [i for i, x in enumerate(sampMedians[:logistigateDict_PHI['importerNum']]) if x > 0.1]
    #highImporterInds = [highImporterInds[i] for i in
    #                    [3, 6, 7, 8, 9, 12, 13, 16]]  # Only manufacturers with more than 1 sample
    highOutletInds = [i for i, x in enumerate(sampMedians[logistigateDict_PHI['importerNum']:]) if x > 0.1]
    #util.plotPostSamples(logistigateDict_PHI, importerIndsSubset=highImporterInds,
    #                     outletIndsSubset=highOutletInds,
    #                     subTitleStr=['\nPhilippines - Subset', '\nPhilippines - Subset'])

    # Special plotting for these data sets
    numImp, numOut = logistigateDict_PHI['importerNum'], logistigateDict_PHI['outletNum']
    lowerQuant, upperQuant = 0.05, 0.95
    intStr = '90'
    priorSamps = logistigateDict_PHI['prior'].expitrand(5000)
    priorLower, priorUpper = np.quantile(priorSamps, lowerQuant), np.quantile(priorSamps, upperQuant)
    importerIndsSubset = range(numImp)
    impNames = [logistigateDict_PHI['importerNames'][i] for i in importerIndsSubset]
    impLowers = [np.quantile(logistigateDict_PHI['postSamples'][:, l], lowerQuant) for l in
                 importerIndsSubset]
    impUppers = [np.quantile(logistigateDict_PHI['postSamples'][:, l], upperQuant) for l in
                 importerIndsSubset]
    midpoints = [impUppers[i] - (impUppers[i] - impLowers[i]) / 2 for i in range(len(impUppers))]
    zippedList = zip(midpoints, impUppers, impLowers, impNames)
    sorted_pairs = sorted(zippedList, reverse=True)
    impNamesSorted = [tup[3] for tup in sorted_pairs]
    impNamesSorted.append('')
    impNamesSorted.append('(Prior)')
    # Plot
    import matplotlib.pyplot as plt
    fig, (ax) = plt.subplots(figsize=(10, 10), ncols=1)
    sorted_pairs.append((np.nan, np.nan, np.nan, ' '))  # for spacing
    for _, upper, lower, name in sorted_pairs:
        plt.plot((name, name), (lower, upper), 'o-', color='red')
    plt.plot((impNamesSorted[-1], impNamesSorted[-1]), (priorLower, priorUpper), 'o--', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(impNamesSorted)), impNamesSorted, rotation=90)
    plt.title('Manufacturers - ' + intStr + '% Intervals' + '\nPhilippines Anti-tuberculosis Medicines',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Manufacturer Name', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(9)
    fig.tight_layout()
    plt.show()
    plt.close()

    outletIndsSubset = range(numOut)
    outNames = [logistigateDict_PHI['outletNames'][i] for i in outletIndsSubset]
    outLowers = [np.quantile(logistigateDict_PHI['postSamples'][:, numImp + l], lowerQuant) for l in
                 outletIndsSubset]
    outUppers = [np.quantile(logistigateDict_PHI['postSamples'][:, numImp + l], upperQuant) for l in
                 outletIndsSubset]
    midpoints = [outUppers[i] - (outUppers[i] - outLowers[i]) / 2 for i in range(len(outUppers))]
    zippedList = zip(midpoints, outUppers, outLowers, outNames)
    sorted_pairs = sorted(zippedList, reverse=True)
    outNamesSorted = [tup[3] for tup in sorted_pairs]
    outNamesSorted.append('')
    outNamesSorted.append('(Prior)')
    # Plot
    fig, (ax) = plt.subplots(figsize=(8, 10), ncols=1)
    sorted_pairs.append((np.nan, np.nan, np.nan, ' '))  # for spacing
    for _, upper, lower, name in sorted_pairs:
        plt.plot((name, name), (lower, upper), 'o-', color='purple')
    plt.plot((outNamesSorted[-1], outNamesSorted[-1]), (priorLower, priorUpper), 'o--', color='gray')
    plt.ylim([0, 1])
    plt.xticks(range(len(outNamesSorted)), outNamesSorted, rotation=90)
    plt.title('Regional Aggregates - ' + intStr + '% Intervals' + '\nPhilippines Anti-tuberculosis Medicines',
              fontdict={'fontsize': 18, 'fontname': 'Trebuchet MS'})
    plt.xlabel('Regional Aggregate', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    plt.ylabel('Interval value', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(11)
    fig.tight_layout()
    plt.show()
    plt.close()

    util.Summarize(logistigateDict_PHI)
    util.printEstimates(logistigateDict_PHI, importerIndsSubset=highImporterInds, outletIndsSubset=highOutletInds)

    # Run with Philippines provinces filtered for outlet-type samples
    dataTblDict_PHI_filt = util.testresultsfiletotable('MQDfiles/MQD_PHILIPPINES_FACILITYFILTER.csv')
    countryMean = np.sum(dataTblDict_PHI_filt['Y']) / np.sum(dataTblDict_PHI_filt['N'])
    dataTblDict_PHI_filt.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 1000,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_PHI_filt = lg.runlogistigate(dataTblDict_PHI_filt)
    util.plotPostSamples(logistigateDict_PHI_filt, plotType='int90', subTitleStr=['\nPhilippines (filtered)', '\nPhilippines (filtered)'])
    util.printEstimates(logistigateDict_PHI_filt)

    # Run with Thailand provinces
    dataTblDict_THA = util.testresultsfiletotable('MQDfiles/MQD_THAILAND.csv')
    countryMean = np.sum(dataTblDict_THA['Y']) / np.sum(dataTblDict_THA['N'])
    dataTblDict_THA.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 500,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_THA = lg.runlogistigate(dataTblDict_THA)
    util.plotPostSamples(logistigateDict_THA)
    util.printEstimates(logistigateDict_THA)

    # Run with Viet Nam provinces
    dataTblDict_VIE = util.testresultsfiletotable('MQDfiles/MQD_VIETNAM.csv')
    countryMean = np.sum(dataTblDict_VIE['Y']) / np.sum(dataTblDict_VIE['N'])
    dataTblDict_VIE.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 500,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_VIE = lg.runlogistigate(dataTblDict_VIE)
    util.plotPostSamples(logistigateDict_VIE)
    util.printEstimates(logistigateDict_VIE)

    return


