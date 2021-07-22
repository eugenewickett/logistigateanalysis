# Workaround for the 'methods' file not being able to locate the 'mcmcsamplers' folder for importing
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, 'logistigate','logistigate')))

import logistigate.logistigate.utilities as util # Pull from the submodule "develop" branch
import logistigate.logistigate.methods as methods # Pull from the submodule "develop" branch
import logistigate.logistigate.lg as lg # Pull from the submodule "develop" branch

def cleanMQD():
    '''
    Script that cleans up raw Medicines Quality Database data for use in logistigate.
    It reads in a CSV file with columns 'Country,' 'Province,' 'Therapeutic Indication',
    'Manufacturer,' 'Facility Type', 'Date Sample Collected', 'Final Test Result,' and
    'Type of Test', and returns a dictionary of objects to be formatted for use with logistigate.
    '''
    # Read in the raw database file
    import pandas as pd
    SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
    filesPath = os.path.join(SCRIPT_DIR, 'MQDfiles')
    MQD_df = pd.read_csv(os.path.join(filesPath,'MQD_ALL_CSV.csv')) # Main raw database file
    # Get data particular to each country of interest
    MQD_df_CAM = MQD_df[MQD_df['Country'] == 'Cambodia'].copy()
    MQD_df_GHA = MQD_df[MQD_df['Country'] == 'Ghana'].copy()
    MQD_df_PHI = MQD_df[MQD_df['Country'] == 'Philippines'].copy()

    # Consolidate typos or seemingly identical entries in significant categories
    # Cambodia
    # Province
    MQD_df_CAM.loc[
        (MQD_df_CAM.Province == 'Ratanakiri') | (MQD_df_CAM.Province == 'Rattanakiri'), 'Province'] = 'Ratanakiri'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Province == 'Steung Treng') | (MQD_df_CAM.Province == 'Stung Treng'), 'Province'] = 'Stung Treng'
    # Manufacturer
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Acdhon Co., Ltd') | (MQD_df_CAM.Manufacturer == 'Acdhon Company Ltd'),
        'Manufacturer'] = 'Acdhon Co., Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Alembic Limited') | (MQD_df_CAM.Manufacturer == 'Alembic Pharmaceuticals Ltd'),
        'Manufacturer'] = 'Alembic Limited'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'ALICE PHARMA PVT LTD') | (MQD_df_CAM.Manufacturer == 'Alice Pharma Pvt.Ltd')
        | (MQD_df_CAM.Manufacturer == 'Alice Pharmaceuticals'), 'Manufacturer'] = 'Alice Pharmaceuticals'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Atoz Pharmaceutical Pvt.Ltd') | (MQD_df_CAM.Manufacturer == 'Atoz Pharmaceuticals Ltd'),
        'Manufacturer'] = 'Atoz Pharmaceuticals Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Aurobindo Pharma LTD') | (MQD_df_CAM.Manufacturer == 'Aurobindo Pharma Ltd.')
        | (MQD_df_CAM.Manufacturer == 'Aurobindo Pharmaceuticals Ltd'), 'Manufacturer'] = 'Aurobindo'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Aventis') | (MQD_df_CAM.Manufacturer == 'Aventis Pharma Specialite'),
        'Manufacturer'] = 'Aventis'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Bright Future Laboratories') | (MQD_df_CAM.Manufacturer == 'Bright Future Pharma'),
        'Manufacturer'] = 'Bright Future Laboratories'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Burapha') | (MQD_df_CAM.Manufacturer == 'Burapha Dispensary Co, Ltd'),
        'Manufacturer'] = 'Burapha'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'CHANKIT') | (MQD_df_CAM.Manufacturer == 'Chankit Trading Ltd')
        | (MQD_df_CAM.Manufacturer == 'Chankit trading Ltd, Part'),
        'Manufacturer'] = 'Chankit Trading Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Chea Chamnan Laboratoire Co., LTD') | (MQD_df_CAM.Manufacturer == 'Chea Chamnan Laboratories Co., Ltd')
        | (MQD_df_CAM.Manufacturer == 'Chea Chamnan Laboratory Company Ltd'),
        'Manufacturer'] = 'Chea Chamnan Laboratory Company Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Cipla Ltd.') | (MQD_df_CAM.Manufacturer == 'Cipla Ltd'),
        'Manufacturer'] = 'Cipla Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'DOMESCO MEDICAL IMP EXP JOINT STOCK CORP')
        | (MQD_df_CAM.Manufacturer == 'DOMESCO MEDICAL IMP EXP JOINT_stock corp')
        | (MQD_df_CAM.Manufacturer == 'DOMESCO MEDICAL IMPORT EXPORT JOINT STOCK CORP')
        | (MQD_df_CAM.Manufacturer == 'Domesco'),
        'Manufacturer'] = 'Domesco'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Emcure Pharmaceutical') | (MQD_df_CAM.Manufacturer == 'Emcure'),
        'Manufacturer'] = 'Emcure'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Eurolife Healthcare Pvt Ltd') | (MQD_df_CAM.Manufacturer == 'Eurolife'),
        'Manufacturer'] = 'Eurolife'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Flamingo Pharmaceutical Limited') | (MQD_df_CAM.Manufacturer == 'Flamingo Pharmaceuticals Ltd'),
        'Manufacturer'] = 'Flamingo Pharmaceuticals Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Global Pharma Health care PVT-LTD')
        | (MQD_df_CAM.Manufacturer == 'GlobalPharma Healthcare Pvt-Ltd')
        | (MQD_df_CAM.Manufacturer == 'Global Pharma'),
        'Manufacturer'] = 'Global Pharma'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Gracure Pharmaceuticals Ltd.') | (MQD_df_CAM.Manufacturer == 'Gracure Pharmaceuticals'),
        'Manufacturer'] = 'Gracure Pharmaceuticals'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Il Dong Pharmaceutical Company Ltd') | (MQD_df_CAM.Manufacturer == 'Il Dong Pharmaceuticals Ltd'),
        'Manufacturer'] = 'Il Dong Pharmaceuticals Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Khandelwal Laboratories Ltd')
        | (MQD_df_CAM.Manufacturer == 'Khandewal Lab')
        | (MQD_df_CAM.Manufacturer == 'Khandelwal'),
        'Manufacturer'] = 'Khandelwal'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Laboratories EPHAC Co., Ltd')
        | (MQD_df_CAM.Manufacturer == 'EPHAC Laboratories Ltd'),
        'Manufacturer'] = 'Laboratories EPHAC Co., Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Lyka Laboratories Ltd')
        | (MQD_df_CAM.Manufacturer == 'Lyka Labs Limited.')
        | (MQD_df_CAM.Manufacturer == 'Lyka Labs'),
        'Manufacturer'] = 'Lyka Labs'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Marksans Pharmaceuticals Ltd') | (MQD_df_CAM.Manufacturer == 'Marksans Pharma Ltd.')
        | (MQD_df_CAM.Manufacturer == 'Marksans Pharma Ltd.,'),
        'Manufacturer'] = 'Marksans Pharma Ltd.'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'MASALAB') | (MQD_df_CAM.Manufacturer == 'Masa Lab Co., Ltd'),
        'Manufacturer'] = 'Masa Lab Co., Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Medical Supply Pharmaceutical Enterprise')
        | (MQD_df_CAM.Manufacturer == 'Medical Supply Pharmaceutical Enteprise'),
        'Manufacturer'] = 'Medical Supply Pharmaceutical Enterprise'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Medopharm Pvt. Ltd.')
        | (MQD_df_CAM.Manufacturer == 'Medopharm'),
        'Manufacturer'] = 'Medopharm'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Micro Laboratories Ltd') | (MQD_df_CAM.Manufacturer == 'MICRO LAB LIMITED')
        | (MQD_df_CAM.Manufacturer == 'Micro Labs Ltd') | (MQD_df_CAM.Manufacturer == 'Microlabs Limited'),
        'Manufacturer'] = 'Microlabs'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Millimed Co., Ltd Thailand')
        | (MQD_df_CAM.Manufacturer == 'Millimed'),
        'Manufacturer'] = 'Millimed'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Orchid Health Care') | (MQD_df_CAM.Manufacturer == 'Orchid Health'),
        'Manufacturer'] = 'Orchid Health'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Osoth Inter Laboratory Co., LTD') | (MQD_df_CAM.Manufacturer == 'Osoth Inter Laboratories'),
        'Manufacturer'] = 'Osoth Inter Laboratories'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'PHARMASANT LABORATORIES Co.,LTD') | (MQD_df_CAM.Manufacturer == 'Pharmasant Laboratories Co., Ltd'),
        'Manufacturer'] = 'Pharmasant Laboratories Co., Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Plethico Pharmaceuticals, Ltd')
        | (MQD_df_CAM.Manufacturer == 'Plethico Pharmaceuticals Ltd')
        | (MQD_df_CAM.Manufacturer == 'Plethico Pharmaceutical Ltd')
        | (MQD_df_CAM.Manufacturer == 'Plethico'),
        'Manufacturer'] = 'Plethico'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'PPM Laboratory') | (MQD_df_CAM.Manufacturer == 'PPM')
        | (MQD_df_CAM.Manufacturer == 'Pharma Product Manufacturing'),
        'Manufacturer'] = 'PPM'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Ranbaxy Laboratories Limited.')
        | (MQD_df_CAM.Manufacturer == 'Ranbaxy Pharmaceuticals'),
        'Manufacturer'] = 'Ranbaxy Pharmaceuticals'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Shijiazhuang Pharma Group Zhongnuo Pharmaceutical [Shijiazhuang] Co.,LTD')
        | (MQD_df_CAM.Manufacturer == 'Shijiazhuang Pharmaceutical Group Ltd'),
        'Manufacturer'] = 'Shijiazhuang Pharmaceutical Group Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Sanofi-Aventis Vietnam') | (MQD_df_CAM.Manufacturer == 'Sanofi Aventis'),
        'Manufacturer'] = 'Sanofi Aventis'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Stada Vietnam Joint Venture Co., Ltd.') | (MQD_df_CAM.Manufacturer == 'Stada Vietnam Joint Venture'),
        'Manufacturer'] = 'Stada Vietnam Joint Venture'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Shandong Reyoung Pharmaceutical Co., Ltd') | (
                    MQD_df_CAM.Manufacturer == 'Shandong Reyoung Pharmaceuticals Ltd'),
        'Manufacturer'] = 'Shandong Reyoung Pharmaceuticals Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'T Man Pharma Ltd. Part.')
        | (MQD_df_CAM.Manufacturer == 'T-MAN Pharma Ltd., Part')
        | (MQD_df_CAM.Manufacturer == 'T-Man Pharmaceuticals Ltd'),
        'Manufacturer'] = 'T-Man Pharmaceuticals Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Umedica Laboratories PVT. LTD.')
        | (MQD_df_CAM.Manufacturer == 'Umedica Laboratories PVT. Ltd')
        | (MQD_df_CAM.Manufacturer == 'Umedica Laboratories Pvt Ltd')
        | (MQD_df_CAM.Manufacturer == 'Umedica'),
        'Manufacturer'] = 'Umedica'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Utopian Co,.LTD') | (MQD_df_CAM.Manufacturer == 'Utopian Co., Ltd')
        | (MQD_df_CAM.Manufacturer == 'Utopian Company Ltd'),
        'Manufacturer'] = 'Utopian Company Ltd'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Vesco Pharmaceutical Ltd.,Part')
        | (MQD_df_CAM.Manufacturer == 'Vesco Pharmaceutical Ltd Part'),
        'Manufacturer'] = 'Vesco Pharmaceutical Ltd Part'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Yanzhou Xier Kangtai Pharmaceutical Co., Ltd')
        | (MQD_df_CAM.Manufacturer == 'Yanzhou Xier Kangtai Pharm'),
        'Manufacturer'] = 'Yanzhou Xier Kangtai Pharm'
    MQD_df_CAM.loc[
        (MQD_df_CAM.Manufacturer == 'Zhangjiakou DongFang pharmaceutical Co., Ltd')
        | (MQD_df_CAM.Manufacturer == 'Zhangjiakou Dongfang Phamaceutical'),
        'Manufacturer'] = 'Zhangjiakou Dongfang Phamaceutical'

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
    dataTblDict_CAM = util.testresultsfiletotable(dataDict['dataTbl_CAM'])
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
    util.plotPostSamples(logistigateDict_CAM,importerIndsSubset=highImporterInds,subTitleStr=['\nCambodia - Subset','\nCambodia'])
    util.printEstimates(logistigateDict_CAM,importerIndsSubset=highImporterInds)
    # Run with Cambodia provinces filtered for outlet-type samples
    dataTblDict_CAM_filt = util.testresultsfiletotable('MQDfiles/MQD_CAMBODIA_FACILITYFILTER.csv')
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
    dataTblDict_CAM_antimalarial = util.testresultsfiletotable('MQDfiles/MQD_CAMBODIA_ANTIMALARIAL.csv')
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
    outNames = [logistigateDict_CAM_antimalarial['outletNames'][i][9:] for i in outletIndsSubset]
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
    np.sum(logistigateDict_CAM_antimalarial['Y'][1])

    # Run with Ethiopia provinces
    dataTblDict_ETH = util.testresultsfiletotable('../examples/data/MQD_ETHIOPIA.csv')
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
    dataTblDict_GHA = util.testresultsfiletotable('../examples/data/MQD_GHANA.csv')
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
    dataTblDict_GHA_filt = util.testresultsfiletotable('../examples/data/MQD_GHANA_FACILITYFILTER.csv')
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
    dataTblDict_GHA_antimalarial = util.testresultsfiletotable('MQDfiles/MQD_GHANA_ANTIMALARIAL.csv')
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
    dataTblDict_KEN = util.testresultsfiletotable('../examples/data/MQD_KENYA.csv')
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
    dataTblDict_LAO = util.testresultsfiletotable('../examples/data/MQD_LAOS.csv')
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
    dataTblDict_MOZ = util.testresultsfiletotable('../examples/data/MQD_MOZAMBIQUE.csv')
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
    dataTblDict_NIG = util.testresultsfiletotable('../examples/data/MQD_NIGERIA.csv')
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
    dataTblDict_PER = util.testresultsfiletotable('../examples/data/MQD_PERU.csv')
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
    dataTblDict_PER_filt = util.testresultsfiletotable('../examples/data/MQD_PERU_FACILITYFILTER.csv')
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
    dataTblDict_PER_antibiotics = util.testresultsfiletotable('../examples/data/MQD_PERU_ANTIBIOTIC.csv')
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
    dataTblDict_PHI = util.testresultsfiletotable('MQDfiles/MQD_PHILIPPINES.csv')
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
    outNames = [logistigateDict_PHI['outletNames'][i][6:] for i in outletIndsSubset]
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
    dataTblDict_PHI_filt = util.testresultsfiletotable('../examples/data/MQD_PHILIPPINES_FACILITYFILTER.csv')
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
    dataTblDict_THA = util.testresultsfiletotable('../examples/data/MQD_THAILAND.csv')
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
    dataTblDict_VIE = util.testresultsfiletotable('../examples/data/MQD_VIETNAM.csv')
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


