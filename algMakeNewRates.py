# -*- coding: utf-8 -*-
'''
Implementation of MakeNewRates algorithm from paper. Form a supply chain with a given number of test nodes, number of
supply nodes, and number of edges, find a set of SFP rates with the same likelihood as some given set of SFP rates;
this new set is initialized with some epsilon value.
'''

### INITIALIZATION OF SYSTEM ###
numSN, numTN = 30, 30
