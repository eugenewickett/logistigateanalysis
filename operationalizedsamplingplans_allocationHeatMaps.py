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

plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.family"] = "serif"

optparamdict = {'deptnames': ['Bakel', 'Bambey', 'Bignona', 'Birkilane', 'Bounkiling', 'Dagana', 'Dakar', 'Diourbel', 'Fatick', 'Foundiougne', 'Gossas', 'Goudiry', 'Goudoump', 'Guediawaye', 'Guinguineo', 'Kaffrine', 'Kanel', 'Kaolack', 'Kebemer', 'Kedougou', 'Keur Massar', 'Kolda', 'Koumpentoum', 'Koungheul', 'Linguere', 'Louga', 'Malem Hoddar', 'Matam', 'Mbacke', 'Mbour', 'Medina Yoro Foulah', 'Nioro du Rip', 'Oussouye', 'Pikine', 'Podor', 'Ranerou Ferlo', 'Rufisque', 'Saint-Louis', 'Salemata', 'Saraya', 'Sedhiou', 'Tambacounda', 'Thies', 'Tivaoune', 'Velingara', 'Ziguinchor'],
                'arcfixedcostmat': np.array([[  0.       ,  91.       ,  96.3333334, 123.6666666, 110.       ,
        252.3333334, 257.6666666, 113.3333334, 183.6666666, 133.3333334,
        189.6666666, 185.6666666,  84.       , 215.       ],
       [ 41.       ,   0.       ,  67.       ,  87.       ,  71.       ,
        212.3333334, 217.       , 104.       , 157.6666666, 125.       ,
        151.       , 149.       ,  75.3333334, 175.       ],
       [ 46.3333334,  67.       ,   0.       ,  89.3333334,  69.3333334,
        213.6666666, 218.3333334, 112.3333334, 213.6666666, 133.3333334,
        146.       , 151.       ,  85.       , 170.       ],
       [ 73.6666666,  87.       ,  89.3333334,   0.       ,  73.3333334,
        175.6666666, 180.3333334, 123.3333334, 173.6666666, 147.3333334,
        209.3333334, 112.3333334, 110.       , 239.3333334],
       [ 60.       ,  71.       ,  69.3333334,  73.3333334,   0.       ,
        198.6666666, 203.6666666, 121.6666666, 175.3333334, 144.6666666,
        130.6666666, 135.6666666,  96.3333334, 154.6666666],
       [202.3333334, 212.3333334, 213.6666666, 175.6666666, 198.6666666,
          0.       , 179.       , 236.3333334, 192.       , 273.6666666,
        207.6666666, 115.3333334, 237.       , 237.6666666],
       [207.6666666, 217.       , 218.3333334, 180.3333334, 203.6666666,
        179.       ,   0.       , 254.6666666, 197.       , 278.6666666,
         81.       , 120.3333334, 241.6666666, 111.3333334],
       [ 63.3333334, 104.       , 112.3333334, 123.3333334, 121.6666666,
        236.3333334, 254.6666666,   0.       , 139.6666666,  74.       ,
        202.3333334, 186.3333334,  93.       , 226.3333334],
       [133.6666666, 157.6666666, 213.6666666, 173.6666666, 175.3333334,
        192.       , 197.       , 139.6666666,   0.       , 164.6666666,
        225.       , 128.3333334, 173.3333334, 254.6666666],
       [ 83.3333334, 125.       , 133.3333334, 147.3333334, 144.6666666,
        273.6666666, 278.6666666,  74.       , 164.6666666,   0.       ,
        225.       , 209.6666666, 114.       , 248.       ],
       [139.6666666, 151.       , 146.       , 209.3333334, 130.6666666,
        207.6666666,  81.       , 202.3333334, 225.       , 225.       ,
          0.       , 148.3333334, 177.3333334,  90.6666666],
       [135.6666666, 149.       , 151.       , 112.3333334, 135.6666666,
        115.3333334, 120.3333334, 186.3333334, 128.3333334, 209.6666666,
        148.3333334,   0.       , 171.3333334, 177.3333334],
       [ 34.       ,  75.3333334,  85.       , 110.       ,  96.3333334,
        237.       , 241.6666666,  93.       , 173.3333334, 114.       ,
        177.3333334, 171.3333334,   0.       , 200.       ],
       [165.       , 175.       , 170.       , 239.3333334, 154.6666666,
        237.6666666, 111.3333334, 226.3333334, 254.6666666, 248.       ,
         90.6666666, 177.3333334, 200.       ,   0.       ]]),
                'regnames': ['Dakar', 'Diourbel', 'Fatick', 'Kaffrine', 'Kaolack', 'Kedougou', 'Kolda', 'Louga', 'Matam', 'Saint-Louis', 'Sedhiou', 'Tambacounda', 'Thies', 'Ziguinchor'],
                'dept_df': pd.read_csv('operationalizedsamplingplans/senegal_csv_files/deptfixedcosts.csv', header=0)
}

# Base allocations
init_n_700_BASE = np.array([ 0., 12.,  0.,  0.,  0.,  0., 57.,  0., 10.,  9.,  9.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0., 21.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  7.,  0.,  0.,  0.,  0.,  6.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.])
init_n_1400_BASE = np.array([ 0., 12.,  0.,  8.,  0.,  0., 58.,  0., 10.,  9.,  9.,  0.,  0.,
        0., 10.,  7.,  0., 35.,  0.,  0., 21.,  0.,  0., 59.,  0.,  0.,
       10.,  0.,  7.,  0.,  0., 15.,  0.,  6.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.])

n_LeastVisited_700_BASE = np.array([ 0., 20.,  0.,  0.,  0.,  0.,  0.,  0., 19., 19., 19.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0., 20.,  0.,  0.,  0.,  0.,  0.,
        0.,  0., 19.,  0.,  0.,  0.,  0., 20.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.])
n_MostSFPs_unif_700_BASE = np.array([ 0.,  0.,  0.,  0.,  0.,  0., 20., 19.,  0.,  0.,  0.,  0.,  0.,
       19.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 19.,  0.,  0., 19.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.])
n_MostSFPs_wtd_700_BASE = np.array([ 0.,  0.,  0.,  0.,  0.,  0., 15., 12.,  0.,  0.,  0.,  0.,  0.,
       19.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 36.,  0.,  0., 14.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.])
n_MoreDist_unif_700_BASE = np.array([0., 8., 0., 0., 0., 0., 9., 8., 0., 0., 0., 0., 0., 9., 0., 0., 0.,
       0., 0., 0., 9., 0., 0., 0., 0., 0., 0., 0., 8., 8., 0., 0., 0., 9.,
       0., 0., 8., 0., 0., 0., 0., 0., 8., 8., 0., 0.])
n_MoreDist_wtd_700_BASE = np.array([ 0., 13.,  0.,  0.,  0.,  0.,  6.,  5.,  0.,  0.,  0.,  0.,  0.,
        5.,  0.,  0.,  0.,  0.,  0.,  0., 13.,  0.,  0.,  0.,  0.,  0.,
        0.,  0., 13.,  6.,  0.,  0.,  0., 13.,  0.,  0.,  6.,  0.,  0.,
        0.,  0.,  0.,  5.,  7.,  0.,  0.])
n_MoreTests_unif_700_BASE = np.array([ 0.,  0.,  0.,  0.,  0.,  0., 22.,  0.,  0.,  0.,  0.,  0.,  0.,
       22.,  0.,  0.,  0.,  0.,  0.,  0., 22.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0., 22.,  0.,  0.,  0., 22.,  0.,  0., 22.,  0.,  0.,
        0.,  0.,  0., 22., 22.,  0.,  0.])
n_MoreTests_wtd_700_BASE = np.array([ 0.,  0.,  0.,  0.,  0.,  0., 13.,  0.,  0.,  0.,  0.,  0.,  0.,
       14.,  0.,  0.,  0.,  0.,  0.,  0., 43.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0., 15.,  0.,  0.,  0., 43.,  0.,  0., 15.,  0.,  0.,
        0.,  0.,  0., 14., 19.,  0.,  0.])

n_LeastVisited_1400_BASE = np.array([0., 4., 0., 4., 0., 0., 0., 0., 4., 4., 4., 4., 0., 0., 4., 0., 0.,
       0., 0., 0., 5., 0., 0., 4., 4., 4., 4., 0., 4., 0., 0., 4., 0., 4.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
n_MostSFPs_unif_1400_BASE = np.array([0., 0., 0., 0., 0., 0., 8., 7., 0., 0., 0., 0., 0., 8., 0., 0., 7.,
       0., 0., 0., 0., 7., 7., 0., 0., 0., 0., 7., 0., 0., 0., 0., 0., 0.,
       7., 0., 0., 7., 0., 0., 0., 8., 0., 0., 7., 0.])
n_MostSFPs_wtd_1400_BASE = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  6.,  5.,  0.,  0.,  0.,  0.,  0.,
        8.,  0.,  0.,  8.,  0.,  0.,  0.,  0.,  5.,  8.,  0.,  0.,  0.,
        0.,  6.,  0.,  0.,  0.,  0.,  0.,  0., 14.,  0.,  0.,  5.,  0.,
        0.,  0.,  6.,  0.,  0.,  9.,  0.])
n_MoreDist_unif_1400_BASE = np.array([0., 8., 0., 7., 0., 0., 8., 8., 8., 7., 7., 0., 0., 8., 7., 7., 0.,
       7., 0., 0., 8., 0., 0., 7., 0., 0., 7., 0., 8., 8., 0., 7., 0., 8.,
       0., 0., 8., 0., 0., 0., 0., 0., 8., 8., 0., 0.])
n_MoreDist_wtd_1400_BASE = np.array([ 0., 10.,  0.,  9.,  0.,  0.,  4.,  4., 10., 10., 10.,  0.,  0.,
        5.,  9.,  7.,  0.,  5.,  0.,  0.,  9.,  0.,  0.,  9.,  0.,  0.,
        9.,  0., 10.,  5.,  0.,  9.,  0.,  9.,  0.,  0.,  5.,  0.,  0.,
        0.,  0.,  0.,  5.,  6.,  0.,  0.])
n_MoreTests_unif_1400_BASE = np.array([ 0., 26.,  0.,  0.,  0.,  0., 27., 26., 26., 26., 26.,  0.,  0.,
       26.,  0.,  0.,  0.,  0.,  0.,  0., 26.,  0.,  0.,  0.,  0.,  0.,
        0.,  0., 26., 26.,  0.,  0.,  0., 26.,  0.,  0., 26.,  0.,  0.,
        0.,  0.,  0., 26., 26.,  0.,  0.])
n_MoreTests_wtd_1400_BASE = np.array([ 0., 36.,  0.,  0.,  0.,  0., 15., 15., 36., 36., 36.,  0.,  0.,
       16.,  0.,  0.,  0.,  0.,  0.,  0., 36.,  0.,  0.,  0.,  0.,  0.,
        0.,  0., 36., 16.,  0.,  0.,  0., 36.,  0.,  0., 16.,  0.,  0.,
        0.,  0.,  0., 16., 19.,  0.,  0.])

# First shuffle
optparamdict_shuf1 = optparamdict.copy()
numTN = optparamdict['dept_df'].shape[0]
np.random.seed(234)
shufinds = choice(np.arange(numTN), size=numTN, replace=False)
tempN, tempY = optparamdict['dept_df']['N'][shufinds].to_numpy(), optparamdict['dept_df']['Y'][shufinds].to_numpy()
temp_df = optparamdict['dept_df'].copy()
temp_df.update({'N': tempN, 'Y': tempY})
optparamdict_shuf1.update({'dept_df':temp_df})

init_n_700_shuf1 = np.array([ 0.,  0.,  0.,  0.,  0.,  0., 13.,  0.,  0.,  9.,  9.,  0.,  0.,
       21.,  0.,  0.,  0.,  0.,  0.,  0.,  9.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0., 33.,  0.,  0.,  0.,  0.,  0.,  0.,  8.,  0.,  0.,
        0.,  0.,  0., 19.,  0.,  0.,  0.])
init_n_1400_shuf1 = np.array([ 0.,  0.,  0., 10.,  0.,  0., 13., 16.,  0.,  9.,  9.,  0.,  0.,
       21., 12., 41.,  0.,  0.,  0.,  0.,  9.,  0.,  0., 33.,  0.,  0.,
        8.,  0.,  0., 13.,  0.,  0.,  0., 30.,  0.,  0.,  8.,  0.,  0.,
        0.,  0.,  0., 19.,  0.,  0.,  0.])

n_LeastVisited_700_shuf1 = np.array([0., 0., 0., 0., 0., 0., 4., 3., 0., 3., 3., 0., 0., 4., 3., 0., 0.,
       0., 0., 0., 4., 0., 0., 0., 0., 0., 0., 0., 0., 4., 0., 0., 0., 0.,
       0., 0., 4., 0., 0., 0., 0., 0., 0., 4., 0., 0.])
n_MostSFPs_unif_700_shuf1 = np.array([0., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       5., 6., 0., 0., 0., 0., 0., 0., 0., 0., 0., 5., 0., 0., 5., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 6., 0., 0., 0.])
n_MostSFPs_wtd_700_shuf1 = np.array([0., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       7., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0., 4., 0., 0., 5., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 7., 0., 0., 0.])
n_MoreDist_unif_700_shuf1 = np.array([0., 8., 0., 0., 0., 0., 9., 8., 0., 0., 0., 0., 0., 9., 0., 0., 0.,
       0., 0., 0., 9., 0., 0., 0., 0., 0., 0., 0., 8., 8., 0., 0., 0., 9.,
       0., 0., 8., 0., 0., 0., 0., 0., 8., 8., 0., 0.])
n_MoreDist_wtd_700_shuf1 = np.array([ 0.,  5.,  0.,  0.,  0.,  0., 10., 10.,  0.,  0.,  0.,  0.,  0.,
       10.,  0.,  0.,  0.,  0.,  0.,  0., 10.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  5., 10.,  0.,  0.,  0.,  6.,  0.,  0., 10.,  0.,  0.,
        0.,  0.,  0.,  6., 10.,  0.,  0.])
n_MoreTests_unif_700_shuf1 = np.array([ 0.,  0.,  0.,  0.,  0.,  0., 22.,  0.,  0.,  0.,  0.,  0.,  0.,
       22.,  0.,  0.,  0.,  0.,  0.,  0., 22.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0., 22.,  0.,  0.,  0., 22.,  0.,  0., 22.,  0.,  0.,
        0.,  0.,  0., 22., 22.,  0.,  0.])
n_MoreTests_wtd_700_shuf1 = np.array([ 0.,  0.,  0.,  0.,  0.,  0., 25.,  0.,  0.,  0.,  0.,  0.,  0.,
       25.,  0.,  0.,  0.,  0.,  0.,  0., 25.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0., 25.,  0.,  0.,  0., 13.,  0.,  0., 25.,  0.,  0.,
        0.,  0.,  0., 13., 25.,  0.,  0.])

n_LeastVisited_1400_shuf1 = np.array([0., 0., 0., 4., 0., 0., 4., 3., 0., 3., 3., 0., 0., 4., 4., 4., 0.,
       0., 0., 0., 4., 0., 0., 4., 0., 0., 4., 0., 0., 4., 0., 0., 0., 0.,
       3., 3., 4., 0., 0., 0., 0., 0., 0., 4., 0., 0.])
n_MostSFPs_unif_1400_shuf1 = np.array([8., 8., 0., 0., 0., 8., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       8., 9., 0., 0., 0., 0., 0., 0., 0., 0., 0., 8., 0., 0., 8., 0., 0.,
       0., 0., 0., 8., 0., 0., 0., 0., 9., 0., 0., 0.])
n_MostSFPs_wtd_1400_shuf1 = np.array([ 7.,  6.,  0.,  0.,  0.,  9.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  9.,  6.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  6.,  0.,  0.,  5.,  0.,  0.,  0.,  0.,  0., 16.,  0.,
        0.,  0.,  0., 10.,  0.,  0.,  0.])
n_MoreDist_unif_1400_shuf1 = np.array([0., 8., 0., 7., 0., 0., 8., 8., 8., 7., 7., 0., 0., 8., 7., 7., 0.,
       7., 0., 0., 8., 0., 0., 7., 0., 0., 7., 0., 8., 8., 0., 7., 0., 8.,
       0., 0., 8., 0., 0., 0., 0., 0., 8., 8., 0., 0.])
n_MoreDist_wtd_1400_shuf1 = np.array([0., 4., 0., 9., 0., 0., 9., 9., 5., 9., 9., 0., 0., 9., 9., 9., 0.,
       5., 0., 0., 9., 0., 0., 9., 0., 0., 9., 0., 4., 9., 0., 4., 0., 5.,
       0., 0., 9., 0., 0., 0., 0., 0., 5., 9., 0., 0.])
n_MoreTests_unif_1400_shuf1 = np.array([ 0., 26.,  0.,  0.,  0.,  0., 27., 26., 26., 26., 26.,  0.,  0.,
       26.,  0.,  0.,  0.,  0.,  0.,  0., 26.,  0.,  0.,  0.,  0.,  0.,
        0.,  0., 26., 26.,  0.,  0.,  0., 26.,  0.,  0., 26.,  0.,  0.,
        0.,  0.,  0., 26., 26.,  0.,  0.])
n_MoreTests_wtd_1400_shuf1 = np.array([ 0., 14.,  0.,  0.,  0.,  0., 32., 32., 16., 32., 32.,  0.,  0.,
       32.,  0.,  0.,  0.,  0.,  0.,  0., 32.,  0.,  0.,  0.,  0.,  0.,
        0.,  0., 14., 32.,  0.,  0.,  0., 15.,  0.,  0., 32.,  0.,  0.,
        0.,  0.,  0., 15., 32.,  0.,  0.])

# Second shuffle
optparamdict_shuf2 = optparamdict.copy()
numTN = optparamdict['dept_df'].shape[0]
np.random.seed(7654)
shufinds = choice(np.arange(numTN), size=numTN, replace=False)
tempN, tempY = optparamdict['dept_df']['N'][shufinds].to_numpy(), optparamdict['dept_df']['Y'][shufinds].to_numpy()
temp_df = optparamdict['dept_df'].copy()
temp_df.update({'N': tempN, 'Y': tempY})
optparamdict_shuf2.update({'dept_df':temp_df})

init_n_700_shuf2 = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  9.,  6., 47.,  0.,  0.,
       10.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0., 10.,  0.,  0.,  0.,  7.,  0.,  0.,  8.,  0.,  0.,
        0.,  0.,  0.,  0., 17.,  0.,  0.])
init_n_1400_shuf2 = np.array([ 0.,  0.,  0.,  7.,  0.,  0.,  0., 36.,  9.,  6., 10.,  0.,  0.,
       10., 41.,  0.,  0., 44.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0., 13., 10.,  0., 49.,  0.,  7.,  0.,  0.,  8.,  0.,  0.,
        0.,  0.,  0.,  0., 17.,  0.,  0.])

n_LeastVisited_700_shuf2 = np.array([ 0., 11.,  0.,  0.,  0.,  0.,  0.,  0., 11., 11., 10.,  0.,  0.,
       11.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0., 11., 11.,  0.,  0.,  0.,  0.,  0.,  0., 11.,  0.,  0.,
        0.,  0.,  0.,  0., 11.,  0.,  0.])
n_MostSFPs_unif_700_shuf2 = np.array([0., 0., 0., 0., 0., 0., 9., 8., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 9., 0., 9., 0., 0., 0., 8., 9., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 9., 0., 0., 0.])
n_MostSFPs_wtd_700_shuf2 = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  7.,  7.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0., 11.,  0., 10.,  0.,  0.,  0., 10., 11.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  6.,  0.,  0.,  0.])
n_MoreDist_unif_700_shuf2 = np.array([0., 8., 0., 0., 0., 0., 9., 8., 0., 0., 0., 0., 0., 9., 0., 0., 0.,
       0., 0., 0., 9., 0., 0., 0., 0., 0., 0., 0., 8., 8., 0., 0., 0., 9.,
       0., 0., 8., 0., 0., 0., 0., 0., 8., 8., 0., 0.])
n_MoreDist_wtd_700_shuf2 = np.array([ 0., 11.,  0.,  0.,  0.,  0.,  5.,  5.,  0.,  0.,  0.,  0.,  0.,
       10.,  0.,  0.,  0.,  0.,  0.,  0.,  5.,  0.,  0.,  0.,  0.,  0.,
        0.,  0., 11., 11.,  0.,  0.,  0.,  8.,  0.,  0., 10.,  0.,  0.,
        0.,  0.,  0.,  5., 11.,  0.,  0.])
n_MoreTests_unif_700_shuf2 = np.array([ 0.,  0.,  0.,  0.,  0.,  0., 22.,  0.,  0.,  0.,  0.,  0.,  0.,
       22.,  0.,  0.,  0.,  0.,  0.,  0., 22.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0., 22.,  0.,  0.,  0., 22.,  0.,  0., 22.,  0.,  0.,
        0.,  0.,  0., 22., 22.,  0.,  0.])
n_MoreTests_wtd_700_shuf2 = np.array([ 0.,  0.,  0.,  0.,  0.,  0., 12.,  0.,  0.,  0.,  0.,  0.,  0.,
       29.,  0.,  0.,  0.,  0.,  0.,  0., 13.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0., 30.,  0.,  0.,  0., 21.,  0.,  0., 29.,  0.,  0.,
        0.,  0.,  0., 12., 30.,  0.,  0.])

n_LeastVisited_1400_shuf2 = np.array([0., 7., 0., 7., 0., 0., 0., 0., 7., 7., 7., 0., 0., 8., 7., 0., 7.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 7., 7., 7., 0., 7., 0., 0.,
       0., 0., 7., 0., 0., 0., 0., 0., 0., 7., 0., 0.])
n_MostSFPs_unif_1400_shuf2 = np.array([0., 0., 0., 0., 0., 0., 8., 7., 0., 0., 0., 0., 0., 0., 0., 7., 0.,
       0., 7., 0., 8., 0., 0., 0., 7., 8., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 7., 8., 0., 7., 0.])
n_MostSFPs_wtd_1400_shuf2 = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  6.,  6.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  6.,  0.,  0., 10.,  0.,  9.,  0.,  0.,  0.,  9., 10.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  6.,  5.,  0.,  7.,  0.])
n_MoreDist_unif_1400_shuf2 = np.array([0., 8., 0., 7., 0., 0., 8., 8., 8., 7., 7., 0., 0., 8., 7., 7., 0.,
       7., 0., 0., 8., 0., 0., 7., 0., 0., 7., 0., 8., 8., 0., 7., 0., 8.,
       0., 0., 8., 0., 0., 0., 0., 0., 8., 8., 0., 0.])
n_MoreDist_wtd_1400_shuf2 = np.array([ 0., 10.,  0., 10.,  0.,  0.,  4.,  4., 10., 10., 10.,  0.,  0.,
        9.,  9.,  4.,  0.,  5.,  0.,  0.,  5.,  0.,  0.,  6.,  0.,  0.,
        5.,  0., 10.,  9.,  0., 10.,  0.,  7.,  0.,  0.,  9.,  0.,  0.,
        0.,  0.,  0.,  4.,  9.,  0.,  0.])
n_MoreTests_unif_1400_shuf2 = np.array([ 0., 26.,  0.,  0.,  0.,  0., 27., 26., 26., 26., 26.,  0.,  0.,
       26.,  0.,  0.,  0.,  0.,  0.,  0., 26.,  0.,  0.,  0.,  0.,  0.,
        0.,  0., 26., 26.,  0.,  0.,  0., 26.,  0.,  0., 26.,  0.,  0.,
        0.,  0.,  0., 26., 26.,  0.,  0.])
n_MoreTests_wtd_1400_shuf2 = np.array([ 0., 32.,  0.,  0.,  0.,  0., 14., 14., 32., 32., 32.,  0.,  0.,
       31.,  0.,  0.,  0.,  0.,  0.,  0., 15.,  0.,  0.,  0.,  0.,  0.,
        0.,  0., 32., 31.,  0.,  0.,  0., 23.,  0.,  0., 31.,  0.,  0.,
        0.,  0.,  0., 14., 32.,  0.,  0.])

# Population emphasis
init_n_700_POPemph = np.array([ 0., 13.,  0.,  0.,  0.,  0., 50.,  0., 13., 12.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0., 23.,  0.,  0.,  0.,  0.,  0.,
        0.,  0., 34.,  0.,  0.,  0.,  0.,  7.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.])
init_n_1400_POPemph = np.array([ 0., 13.,  0.,  0.,  0.,  0., 50.,  0., 13., 12.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0., 23.,  0.,  0.,  0.,  0.,  0.,
        0.,  0., 81., 51.,  0.,  0.,  0., 18.,  0.,  0., 54.,  0.,  0.,
        0.,  0.,  0., 57., 31.,  0.,  0.])

n_LeastVisited_700_POPemph = np.array([ 0., 20.,  0.,  0.,  0.,  0.,  0.,  0., 19., 19., 19.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0., 20.,  0.,  0.,  0.,  0.,  0.,
        0.,  0., 19.,  0.,  0.,  0.,  0., 20.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.])
n_MostSFPs_unif_700_POPemph = np.array([ 0.,  0.,  0.,  0.,  0.,  0., 20., 19.,  0.,  0.,  0.,  0.,  0.,
       19.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 19.,  0.,  0., 19.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.])
n_MostSFPs_wtd_700_POPemph = np.array([ 0.,  0.,  0.,  0.,  0.,  0., 15., 12.,  0.,  0.,  0.,  0.,  0.,
       19.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 36.,  0.,  0., 14.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.])
n_MoreDist_unif_700_POPemph = np.array([0., 8., 0., 0., 0., 0., 9., 8., 0., 0., 0., 0., 0., 9., 0., 0., 0.,
       0., 0., 0., 9., 0., 0., 0., 0., 0., 0., 0., 8., 8., 0., 0., 0., 9.,
       0., 0., 8., 0., 0., 0., 0., 0., 8., 8., 0., 0.])
n_MoreDist_wtd_700_POPemph = np.array([ 0., 13.,  0.,  0.,  0.,  0.,  6.,  5.,  0.,  0.,  0.,  0.,  0.,
        5.,  0.,  0.,  0.,  0.,  0.,  0., 13.,  0.,  0.,  0.,  0.,  0.,
        0.,  0., 13.,  6.,  0.,  0.,  0., 13.,  0.,  0.,  6.,  0.,  0.,
        0.,  0.,  0.,  5.,  7.,  0.,  0.])
n_MoreTests_unif_700_POPemph = np.array([ 0.,  0.,  0.,  0.,  0.,  0., 22.,  0.,  0.,  0.,  0.,  0.,  0.,
       22.,  0.,  0.,  0.,  0.,  0.,  0., 22.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0., 22.,  0.,  0.,  0., 22.,  0.,  0., 22.,  0.,  0.,
        0.,  0.,  0., 22., 22.,  0.,  0.])
n_MoreTests_wtd_700_POPemph = np.array([ 0.,  0.,  0.,  0.,  0.,  0., 13.,  0.,  0.,  0.,  0.,  0.,  0.,
       14.,  0.,  0.,  0.,  0.,  0.,  0., 43.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0., 15.,  0.,  0.,  0., 43.,  0.,  0., 15.,  0.,  0.,
        0.,  0.,  0., 14., 19.,  0.,  0.])

n_LeastVisited_1400_POPemph = np.array([0., 4., 0., 4., 0., 0., 0., 0., 4., 4., 4., 4., 0., 0., 4., 0., 0.,
       0., 0., 0., 5., 0., 0., 4., 4., 4., 4., 0., 4., 0., 0., 4., 0., 4.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
n_MostSFPs_unif_1400_POPemph = np.array([0., 0., 0., 0., 0., 0., 8., 7., 0., 0., 0., 0., 0., 8., 0., 0., 7.,
       0., 0., 0., 0., 7., 7., 0., 0., 0., 0., 7., 0., 0., 0., 0., 0., 0.,
       7., 0., 0., 7., 0., 0., 0., 8., 0., 0., 7., 0.])
n_MostSFPs_wtd_1400_POPemph = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  6.,  5.,  0.,  0.,  0.,  0.,  0.,
        8.,  0.,  0.,  8.,  0.,  0.,  0.,  0.,  5.,  8.,  0.,  0.,  0.,
        0.,  6.,  0.,  0.,  0.,  0.,  0.,  0., 14.,  0.,  0.,  5.,  0.,
        0.,  0.,  6.,  0.,  0.,  9.,  0.])
n_MoreDist_unif_1400_POPemph = np.array([0., 8., 0., 7., 0., 0., 8., 8., 8., 7., 7., 0., 0., 8., 7., 7., 0.,
       7., 0., 0., 8., 0., 0., 7., 0., 0., 7., 0., 8., 8., 0., 7., 0., 8.,
       0., 0., 8., 0., 0., 0., 0., 0., 8., 8., 0., 0.])
n_MoreDist_wtd_1400_POPemph = np.array([ 0., 10.,  0.,  9.,  0.,  0.,  4.,  4., 10., 10., 10.,  0.,  0.,
        5.,  9.,  7.,  0.,  5.,  0.,  0.,  9.,  0.,  0.,  9.,  0.,  0.,
        9.,  0., 10.,  5.,  0.,  9.,  0.,  9.,  0.,  0.,  5.,  0.,  0.,
        0.,  0.,  0.,  5.,  6.,  0.,  0.])
n_MoreTests_unif_1400_POPemph = np.array([ 0., 26.,  0.,  0.,  0.,  0., 27., 26., 26., 26., 26.,  0.,  0.,
       26.,  0.,  0.,  0.,  0.,  0.,  0., 26.,  0.,  0.,  0.,  0.,  0.,
        0.,  0., 26., 26.,  0.,  0.,  0., 26.,  0.,  0., 26.,  0.,  0.,
        0.,  0.,  0., 26., 26.,  0.,  0.])
n_MoreTests_wtd_1400_POPemph = np.array([ 0., 36.,  0.,  0.,  0.,  0., 15., 15., 36., 36., 36.,  0.,  0.,
       16.,  0.,  0.,  0.,  0.,  0.,  0., 36.,  0.,  0.,  0.,  0.,  0.,
        0.,  0., 36., 16.,  0.,  0.,  0., 36.,  0.,  0., 16.,  0.,  0.,
        0.,  0.,  0., 16., 19.,  0.,  0.])

# Manufacturer emphasis
init_n_700_SNemph = np.array([ 0.,  0.,  0.,  0.,  0.,  0., 59.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0., 43.,  0.,  0., 44.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0., 62.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.])
init_n_1400_SNemph = np.array([ 0.       ,  0.       ,  0.       , 24.       ,  0.       ,
        0.       , 65.       ,  0.       , 39.       , 30.       ,
       24.       ,  0.       ,  0.       ,  0.       ,  0.       ,
        0.       ,  0.       , 43.       ,  0.       ,  0.       ,
       44.       ,  0.       ,  0.       , 70.9999999,  0.       ,
        0.       ,  0.       ,  0.       ,  0.       ,  0.       ,
        0.       , 62.       ,  0.       ,  0.       ,  0.       ,
        0.       ,  0.       ,  0.       ,  0.       ,  0.       ,
        0.       ,  0.       ,  0.       ,  0.       ,  0.       ,
        0.       ])

n_LeastVisited_700_SNemph = np.array([ 0., 20.,  0.,  0.,  0.,  0.,  0.,  0., 19., 19., 19.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0., 20.,  0.,  0.,  0.,  0.,  0.,
        0.,  0., 19.,  0.,  0.,  0.,  0., 20.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.])
n_MostSFPs_unif_700_SNemph = np.array([ 0.,  0.,  0.,  0.,  0.,  0., 20., 19.,  0.,  0.,  0.,  0.,  0.,
       19.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 19.,  0.,  0., 19.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.])
n_MostSFPs_wtd_700_SNemph = np.array([ 0.,  0.,  0.,  0.,  0.,  0., 15., 12.,  0.,  0.,  0.,  0.,  0.,
       19.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 36.,  0.,  0., 14.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.])
n_MoreDist_unif_700_SNemph = np.array([0., 8., 0., 0., 0., 0., 9., 8., 0., 0., 0., 0., 0., 9., 0., 0., 0.,
       0., 0., 0., 9., 0., 0., 0., 0., 0., 0., 0., 8., 8., 0., 0., 0., 9.,
       0., 0., 8., 0., 0., 0., 0., 0., 8., 8., 0., 0.])
n_MoreDist_wtd_700_SNemph = np.array([ 0., 13.,  0.,  0.,  0.,  0.,  6.,  5.,  0.,  0.,  0.,  0.,  0.,
        5.,  0.,  0.,  0.,  0.,  0.,  0., 13.,  0.,  0.,  0.,  0.,  0.,
        0.,  0., 13.,  6.,  0.,  0.,  0., 13.,  0.,  0.,  6.,  0.,  0.,
        0.,  0.,  0.,  5.,  7.,  0.,  0.])
n_MoreTests_unif_700_SNemph = np.array([ 0.,  0.,  0.,  0.,  0.,  0., 22.,  0.,  0.,  0.,  0.,  0.,  0.,
       22.,  0.,  0.,  0.,  0.,  0.,  0., 22.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0., 22.,  0.,  0.,  0., 22.,  0.,  0., 22.,  0.,  0.,
        0.,  0.,  0., 22., 22.,  0.,  0.])
n_MoreTests_wtd_700_SNemph = np.array([ 0.,  0.,  0.,  0.,  0.,  0., 13.,  0.,  0.,  0.,  0.,  0.,  0.,
       14.,  0.,  0.,  0.,  0.,  0.,  0., 43.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0., 15.,  0.,  0.,  0., 43.,  0.,  0., 15.,  0.,  0.,
        0.,  0.,  0., 14., 19.,  0.,  0.])

n_LeastVisited_1400_SNemph = np.array([0., 4., 0., 4., 0., 0., 0., 0., 4., 4., 4., 4., 0., 0., 4., 0., 0.,
       0., 0., 0., 5., 0., 0., 4., 4., 4., 4., 0., 4., 0., 0., 4., 0., 4.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
n_MostSFPs_unif_1400_SNemph = np.array([0., 0., 0., 0., 0., 0., 8., 7., 0., 0., 0., 0., 0., 8., 0., 0., 7.,
       0., 0., 0., 0., 7., 7., 0., 0., 0., 0., 7., 0., 0., 0., 0., 0., 0.,
       7., 0., 0., 7., 0., 0., 0., 8., 0., 0., 7., 0.])
n_MostSFPs_wtd_1400_SNemph = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  6.,  5.,  0.,  0.,  0.,  0.,  0.,
        8.,  0.,  0.,  8.,  0.,  0.,  0.,  0.,  5.,  8.,  0.,  0.,  0.,
        0.,  6.,  0.,  0.,  0.,  0.,  0.,  0., 14.,  0.,  0.,  5.,  0.,
        0.,  0.,  6.,  0.,  0.,  9.,  0.])
n_MoreDist_unif_1400_SNemph = np.array([0., 8., 0., 7., 0., 0., 8., 8., 8., 7., 7., 0., 0., 8., 7., 7., 0.,
       7., 0., 0., 8., 0., 0., 7., 0., 0., 7., 0., 8., 8., 0., 7., 0., 8.,
       0., 0., 8., 0., 0., 0., 0., 0., 8., 8., 0., 0.])
n_MoreDist_wtd_1400_SNemph = np.array([ 0., 10.,  0.,  9.,  0.,  0.,  4.,  4., 10., 10., 10.,  0.,  0.,
        5.,  9.,  7.,  0.,  5.,  0.,  0.,  9.,  0.,  0.,  9.,  0.,  0.,
        9.,  0., 10.,  5.,  0.,  9.,  0.,  9.,  0.,  0.,  5.,  0.,  0.,
        0.,  0.,  0.,  5.,  6.,  0.,  0.])
n_MoreTests_unif_1400_SNemph = np.array([ 0., 26.,  0.,  0.,  0.,  0., 27., 26., 26., 26., 26.,  0.,  0.,
       26.,  0.,  0.,  0.,  0.,  0.,  0., 26.,  0.,  0.,  0.,  0.,  0.,
        0.,  0., 26., 26.,  0.,  0.,  0., 26.,  0.,  0., 26.,  0.,  0.,
        0.,  0.,  0., 26., 26.,  0.,  0.])
n_MoreTests_wtd_1400_SNemph = np.array([ 0., 36.,  0.,  0.,  0.,  0., 15., 15., 36., 36., 36.,  0.,  0.,
       16.,  0.,  0.,  0.,  0.,  0.,  0., 36.,  0.,  0.,  0.,  0.,  0.,
        0.,  0., 36., 16.,  0.,  0.,  0., 36.,  0.,  0., 16.,  0.,  0.,
        0.,  0.,  0., 16., 19.,  0.,  0.])


def MakeAllocationHeatMap(n, optparamdict, plotTitle='', savename='', cmapstr='gray', vlist='NA', sortby='districtcost'):
    """Generate an allocation heat map"""
    distNames = optparamdict['deptnames']
    percDistVisited = 100*np.count_nonzero(n)/n.shape[0]
    # Sort regions by distance to HQ, taken to be row 0
    reg_sortinds = np.argsort(optparamdict['arcfixedcostmat'][0])
    regNames_sort = [optparamdict['regnames'][x] for x in reg_sortinds]
    # District list for each region of regNames_sort
    dist_df = optparamdict['dept_df']
    distinreglist = []
    for currReg in regNames_sort:
        currDists = opf.GetDeptChildren(currReg, dist_df)
        # todo: CAN SORT BY OTHER THINGS HERE
        currDistFixedCosts = [dist_df.loc[dist_df['Department'] == x]['DeptFixedCostDays'].to_numpy()[0] for x in currDists]
        distinreglist.append([currDists[x] for x in np.argsort(currDistFixedCosts)])
    listlengths = [len(x) for x in distinreglist]
    maxdistnum = max(listlengths)
    # Initialize storage matrix
    dispmat = np.zeros((len(regNames_sort), maxdistnum))

    for distind, curralloc in enumerate(n):
        currDistName = distNames[distind]
        currRegName = opf.GetRegion(currDistName, dist_df)
        regmatind = regNames_sort.index(currRegName)
        distmatind = distinreglist[regmatind].index(currDistName)
        dispmat[regmatind, distmatind] = curralloc
    if vlist != 'NA':
        plt.imshow(dispmat, cmap=cmapstr, interpolation='nearest', vmin=vlist[0], vmax=vlist[1])
    else:
        plt.imshow(dispmat, cmap=cmapstr, interpolation='nearest')
    plt.ylabel('Ranked distance from HQ region')
    plt.xlabel('Ranked distance from regional capital')
    plt.title(plotTitle)
    plt.text(maxdistnum+0.4, 1, "Districts visited: {:.0f}%".format(percDistVisited), fontsize=8)
    plt.colorbar(location='right', anchor=(0,0.3), shrink=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join('operationalizedsamplingplans', 'plots', savename), bbox_inches='tight')
    plt.show()
    return

col = 'Purples'
vmax=60

###############
# Make plots
###############

# Base
MakeAllocationHeatMap(init_n_700_BASE, optparamdict,
                      plotTitle='B=700, base setting\nIP-RP solution', savename='base_700_IPRP',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(init_n_1400_BASE, optparamdict,
                      plotTitle='B=1400, base setting\nIP-RP solution', savename='base_1400_IPRP',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_LeastVisited_700_BASE, optparamdict,
                      plotTitle='B=700, base setting\nLeast Visited policy', savename='base_700_LeastVisited',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_LeastVisited_1400_BASE, optparamdict,
                      plotTitle='B=1400, base setting\nLeast Visited policy', savename='base_1400_LeastVisisted',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MostSFPs_unif_700_BASE, optparamdict,
                      plotTitle='B=700, base setting\nPast SFPs policy, uniform allocation',
                      savename='base_700_PastSFPs_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MostSFPs_unif_1400_BASE, optparamdict,
                      plotTitle='B=1400, base setting\nPast SFPs policy, uniform allocation',
                      savename='base_1400_PastSFPs_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MostSFPs_wtd_700_BASE, optparamdict,
                      plotTitle='B=700, base setting\nPast SFPs policy, weighted allocation',
                      savename='base_700_PastSFPs_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MostSFPs_wtd_1400_BASE, optparamdict,
                      plotTitle='B=1400, base setting\nPast SFPs policy, weighted allocation',
                      savename='base_1400_PastSFPs_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreDist_unif_700_BASE, optparamdict,
                      plotTitle='B=700, base setting\nMore Districts policy, uniform allocation',
                      savename='base_700_MoreDist_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreDist_unif_1400_BASE, optparamdict,
                      plotTitle='B=1400, base setting\nMore Districts policy, uniform allocation',
                      savename='base_1400_MoreDist_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreDist_wtd_700_BASE, optparamdict,
                      plotTitle='B=700, base setting\nMore Districts policy, weighted allocation',
                      savename='base_700_MoreDist_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreDist_wtd_1400_BASE, optparamdict,
                      plotTitle='B=1400, base setting\nMore Districts policy, weighted allocation',
                      savename='base_1400_MoreDist_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreTests_unif_700_BASE, optparamdict,
                      plotTitle='B=700, base setting\nMore Tests policy, uniform allocation',
                      savename='base_700_MoreTest_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreTests_unif_1400_BASE, optparamdict,
                      plotTitle='B=1400, base setting\nMore Tests policy, uniform allocation',
                      savename='base_1400_MoreTest_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreTests_wtd_700_BASE, optparamdict,
                      plotTitle='B=700, base setting\nMore Tests policy, weighted allocation',
                      savename='base_700_MoreTest_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreTests_wtd_1400_BASE, optparamdict,
                      plotTitle='B=1400, base setting\nMore Tests policy, weighted allocation',
                      savename='base_1400_MoreTest_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])

# First shuffle
MakeAllocationHeatMap(init_n_700_shuf1, optparamdict_shuf1,
                      plotTitle='B=700, first data shuffle\nIP-RP solution', savename='shuf1_700_IPRP',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(init_n_1400_shuf1, optparamdict_shuf1,
                      plotTitle='B=1400, first data shuffle\nIP-RP solution', savename='shuf1_1400_IPRP',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_LeastVisited_700_shuf1, optparamdict_shuf1,
                      plotTitle='B=700, first data shuffle\nLeast Visited policy', savename='shuf1_700_LeastVisited',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_LeastVisited_1400_shuf1, optparamdict_shuf1,
                      plotTitle='B=1400, first data shuffle\nLeast Visited policy', savename='shuf1_1400_LeastVisisted',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MostSFPs_unif_700_shuf1, optparamdict_shuf1,
                      plotTitle='B=700, first data shuffle\nPast SFPs policy, uniform allocation',
                      savename='shuf1_700_PastSFPs_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MostSFPs_unif_1400_shuf1, optparamdict_shuf1,
                      plotTitle='B=1400, first data shuffle\nPast SFPs policy, uniform allocation',
                      savename='shuf1_1400_PastSFPs_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MostSFPs_wtd_700_shuf1, optparamdict_shuf1,
                      plotTitle='B=700, first data shuffle\nPast SFPs policy, weighted allocation',
                      savename='shuf1_700_PastSFPs_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MostSFPs_wtd_1400_shuf1, optparamdict_shuf1,
                      plotTitle='B=1400, first data shuffle\nPast SFPs policy, weighted allocation',
                      savename='shuf1_1400_PastSFPs_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreDist_unif_700_shuf1, optparamdict_shuf1,
                      plotTitle='B=700, first data shuffle\nMore Districts policy, uniform allocation',
                      savename='shuf1_700_MoreDist_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreDist_unif_1400_shuf1, optparamdict_shuf1,
                      plotTitle='B=1400, first data shuffle\nMore Districts policy, uniform allocation',
                      savename='shuf1_1400_MoreDist_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreDist_wtd_700_shuf1, optparamdict_shuf1,
                      plotTitle='B=700, first data shuffle\nMore Districts policy, weighted allocation',
                      savename='shuf1_700_MoreDist_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreDist_wtd_1400_shuf1, optparamdict_shuf1,
                      plotTitle='B=1400, first data shuffle\nMore Districts policy, weighted allocation',
                      savename='shuf1_1400_MoreDist_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreTests_unif_700_shuf1, optparamdict_shuf1,
                      plotTitle='B=700, first data shuffle\nMore Tests policy, uniform allocation',
                      savename='shuf1_700_MoreTest_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreTests_unif_1400_shuf1, optparamdict_shuf1,
                      plotTitle='B=1400, first data shuffle\nMore Tests policy, uniform allocation',
                      savename='shuf1_1400_MoreTest_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreTests_wtd_700_shuf1, optparamdict_shuf1,
                      plotTitle='B=700, first data shuffle\nMore Tests policy, weighted allocation',
                      savename='shuf1_700_MoreTest_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreTests_wtd_1400_shuf1, optparamdict_shuf1,
                      plotTitle='B=1400, first data shuffle\nMore Tests policy, weighted allocation',
                      savename='shuf1_1400_MoreTest_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])

# Second shuffle
MakeAllocationHeatMap(init_n_700_shuf2, optparamdict_shuf2,
                      plotTitle='B=700, second data shuffle\nIP-RP solution', savename='shuf2_700_IPRP',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(init_n_1400_shuf2, optparamdict_shuf2,
                      plotTitle='B=1400, second data shuffle\nIP-RP solution', savename='shuf2_1400_IPRP',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_LeastVisited_700_shuf2, optparamdict_shuf2,
                      plotTitle='B=700, second data shuffle\nLeast Visited policy', savename='shuf2_700_LeastVisited',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_LeastVisited_1400_shuf2, optparamdict_shuf2,
                      plotTitle='B=1400, second data shuffle\nLeast Visited policy', savename='shuf2_1400_LeastVisisted',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MostSFPs_unif_700_shuf2, optparamdict_shuf2,
                      plotTitle='B=700, second data shuffle\nPast SFPs policy, uniform allocation',
                      savename='shuf2_700_PastSFPs_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MostSFPs_unif_1400_shuf2, optparamdict_shuf2,
                      plotTitle='B=1400, second data shuffle\nPast SFPs policy, uniform allocation',
                      savename='shuf2_1400_PastSFPs_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MostSFPs_wtd_700_shuf2, optparamdict_shuf2,
                      plotTitle='B=700, second data shuffle\nPast SFPs policy, weighted allocation',
                      savename='shuf2_700_PastSFPs_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MostSFPs_wtd_1400_shuf2, optparamdict_shuf2,
                      plotTitle='B=1400, second data shuffle\nPast SFPs policy, weighted allocation',
                      savename='shuf2_1400_PastSFPs_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreDist_unif_700_shuf2, optparamdict_shuf2,
                      plotTitle='B=700, second data shuffle\nMore Districts policy, uniform allocation',
                      savename='shuf2_700_MoreDist_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreDist_unif_1400_shuf2, optparamdict_shuf2,
                      plotTitle='B=1400, second data shuffle\nMore Districts policy, uniform allocation',
                      savename='shuf2_1400_MoreDist_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreDist_wtd_700_shuf2, optparamdict_shuf2,
                      plotTitle='B=700, second data shuffle\nMore Districts policy, weighted allocation',
                      savename='shuf2_700_MoreDist_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreDist_wtd_1400_shuf2, optparamdict_shuf2,
                      plotTitle='B=1400, second data shuffle\nMore Districts policy, weighted allocation',
                      savename='shuf2_1400_MoreDist_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreTests_unif_700_shuf2, optparamdict_shuf2,
                      plotTitle='B=700, second data shuffle\nMore Tests policy, uniform allocation',
                      savename='shuf2_700_MoreTest_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreTests_unif_1400_shuf2, optparamdict_shuf2,
                      plotTitle='B=1400, second data shuffle\nMore Tests policy, uniform allocation',
                      savename='shuf2_1400_MoreTest_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreTests_wtd_700_shuf2, optparamdict_shuf2,
                      plotTitle='B=700, second data shuffle\nMore Tests policy, weighted allocation',
                      savename='shuf2_700_MoreTest_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreTests_wtd_1400_shuf2, optparamdict_shuf2,
                      plotTitle='B=1400, second data shuffle\nMore Tests policy, weighted allocation',
                      savename='shuf2_1400_MoreTest_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])

# Pop. emphasis
MakeAllocationHeatMap(init_n_700_POPemph, optparamdict,
                      plotTitle='B=700, population emphasis\nIP-RP solution', savename='POPemph_700_IPRP',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(init_n_1400_POPemph, optparamdict,
                      plotTitle='B=1400, population emphasis\nIP-RP solution', savename='POPemph_1400_IPRP',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_LeastVisited_700_POPemph, optparamdict,
                      plotTitle='B=700, population emphasis\nLeast Visited policy', savename='POPemph_700_LeastVisited',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_LeastVisited_1400_POPemph, optparamdict,
                      plotTitle='B=1400, population emphasis\nLeast Visited policy', savename='POPemph_1400_LeastVisisted',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MostSFPs_unif_700_POPemph, optparamdict,
                      plotTitle='B=700, population emphasis\nPast SFPs policy, uniform allocation',
                      savename='POPemph_700_PastSFPs_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MostSFPs_unif_1400_POPemph, optparamdict,
                      plotTitle='B=1400, population emphasis\nPast SFPs policy, uniform allocation',
                      savename='POPemph_1400_PastSFPs_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MostSFPs_wtd_700_POPemph, optparamdict,
                      plotTitle='B=700, population emphasis\nPast SFPs policy, weighted allocation',
                      savename='POPemph_700_PastSFPs_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MostSFPs_wtd_1400_POPemph, optparamdict,
                      plotTitle='B=1400, population emphasis\nPast SFPs policy, weighted allocation',
                      savename='POPemph_1400_PastSFPs_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreDist_unif_700_POPemph, optparamdict,
                      plotTitle='B=700, population emphasis\nMore Districts policy, uniform allocation',
                      savename='POPemph_700_MoreDist_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreDist_unif_1400_POPemph, optparamdict,
                      plotTitle='B=1400, population emphasis\nMore Districts policy, uniform allocation',
                      savename='POPemph_1400_MoreDist_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreDist_wtd_700_POPemph, optparamdict,
                      plotTitle='B=700, population emphasis\nMore Districts policy, weighted allocation',
                      savename='POPemph_700_MoreDist_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreDist_wtd_1400_POPemph, optparamdict,
                      plotTitle='B=1400, population emphasis\nMore Districts policy, weighted allocation',
                      savename='POPemph_1400_MoreDist_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreTests_unif_700_POPemph, optparamdict,
                      plotTitle='B=700, population emphasis\nMore Tests policy, uniform allocation',
                      savename='POPemph_700_MoreTest_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreTests_unif_1400_POPemph, optparamdict,
                      plotTitle='B=1400, population emphasis\nMore Tests policy, uniform allocation',
                      savename='POPemph_1400_MoreTest_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreTests_wtd_700_POPemph, optparamdict,
                      plotTitle='B=700, population emphasis\nMore Tests policy, weighted allocation',
                      savename='POPemph_700_MoreTest_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreTests_wtd_1400_POPemph, optparamdict,
                      plotTitle='B=1400, population emphasis\nMore Tests policy, weighted allocation',
                      savename='POPemph_1400_MoreTest_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])

# Manuf. emphasis
MakeAllocationHeatMap(init_n_700_SNemph, optparamdict,
                      plotTitle='B=700, manufacturer emphasis\nIP-RP solution', savename='SNemph_700_IPRP',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(init_n_1400_SNemph, optparamdict,
                      plotTitle='B=1400, manufacturer emphasis\nIP-RP solution', savename='SNemph_1400_IPRP',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_LeastVisited_700_SNemph, optparamdict,
                      plotTitle='B=700, manufacturer emphasis\nLeast Visited policy', savename='SNemph_700_LeastVisited',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_LeastVisited_1400_SNemph, optparamdict,
                      plotTitle='B=1400, manufacturer emphasis\nLeast Visited policy', savename='SNemph_1400_LeastVisisted',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MostSFPs_unif_700_SNemph, optparamdict,
                      plotTitle='B=700, manufacturer emphasis\nPast SFPs policy, uniform allocation',
                      savename='SNemph_700_PastSFPs_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MostSFPs_unif_1400_SNemph, optparamdict,
                      plotTitle='B=1400, manufacturer emphasis\nPast SFPs policy, uniform allocation',
                      savename='SNemph_1400_PastSFPs_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MostSFPs_wtd_700_SNemph, optparamdict,
                      plotTitle='B=700, manufacturer emphasis\nPast SFPs policy, weighted allocation',
                      savename='SNemph_700_PastSFPs_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MostSFPs_wtd_1400_SNemph, optparamdict,
                      plotTitle='B=1400, manufacturer emphasis\nPast SFPs policy, weighted allocation',
                      savename='SNemph_1400_PastSFPs_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreDist_unif_700_SNemph, optparamdict,
                      plotTitle='B=700, manufacturer emphasis\nMore Districts policy, uniform allocation',
                      savename='SNemph_700_MoreDist_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreDist_unif_1400_SNemph, optparamdict,
                      plotTitle='B=1400, manufacturer emphasis\nMore Districts policy, uniform allocation',
                      savename='SNemph_1400_MoreDist_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreDist_wtd_700_SNemph, optparamdict,
                      plotTitle='B=700, manufacturer emphasis\nMore Districts policy, weighted allocation',
                      savename='SNemph_700_MoreDist_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreDist_wtd_1400_SNemph, optparamdict,
                      plotTitle='B=1400, manufacturer emphasis\nMore Districts policy, weighted allocation',
                      savename='SNemph_1400_MoreDist_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreTests_unif_700_SNemph, optparamdict,
                      plotTitle='B=700, manufacturer emphasis\nMore Tests policy, uniform allocation',
                      savename='SNemph_700_MoreTest_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreTests_unif_1400_SNemph, optparamdict,
                      plotTitle='B=1400, manufacturer emphasis\nMore Tests policy, uniform allocation',
                      savename='SNemph_1400_MoreTest_unif',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreTests_wtd_700_SNemph, optparamdict,
                      plotTitle='B=700, manufacturer emphasis\nMore Tests policy, weighted allocation',
                      savename='SNemph_700_MoreTest_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])
MakeAllocationHeatMap(n_MoreTests_wtd_1400_SNemph, optparamdict,
                      plotTitle='B=1400, manufacturer emphasis\nMore Tests policy, weighted allocation',
                      savename='SNemph_1400_MoreTest_wtd',
                      cmapstr=col, sortby='districtcost', vlist=[0,vmax])



