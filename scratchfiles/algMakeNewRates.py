# -*- coding: utf-8 -*-
'''
Implementation of MakeNewRates algorithm from paper. Form a supply chain with a given number of test nodes, number of
supply nodes, and number of edges, find a set of SFP rates with the same likelihood as some given set of SFP rates;
this new set is initialized with some epsilon value.
'''

import numpy as np
import random
from scipy.stats import bernoulli
import networkx as nx
import matplotlib.pyplot as plt

### INITIALIZATION OF SYSTEM ###
numSN, numTN = 20, 20
SNset = ['SN ' + str(i) for i in range(numSN)]
TNset = ['TN ' + str(i) for i in range(numTN)]
SNtodraw = SNset.copy()
TNtodraw = TNset.copy()
random.seed(548)
random.shuffle(SNtodraw)
random.shuffle(TNtodraw)
arcset = [(TNtodraw.pop(), SNtodraw.pop())] # Initialize the arc set

iter = 0
while (TNtodraw or SNtodraw):
    # Choose TN or SN probabilistically by number of TNs/SNs remaining
    TNprob = len(TNtodraw) / (len(TNtodraw) + len(SNtodraw))
    if (bernoulli(TNprob).rvs() == 1):
        pickTN = True
    else:
        pickTN = False
    if pickTN:
        nextTN = TNtodraw.pop()
        randSN = random.choice(arcset)[1]
        arcset.append((nextTN, randSN))
    else:
        nextSN = SNtodraw.pop()
        randTN = random.choice(arcset)[0]
        arcset.append((randTN, nextSN))

G = nx.Graph()
G.add_edges_from(arcset)
nx.draw_networkx(G, pos = nx.drawing.layout.bipartite_layout(G, TNset,align='horizontal',
                                                             aspect_ratio=4),width = 2)
plt.show()

# Now run algorithm
def NeighborsA(tn, arcs):
    # Return a list of the neighbors of test node tn in arcs
    neighborSet = []
    for arc in arcs:
        if arc[0] == tn:
            neighborSet.append(arc[1])
    return neighborSet

def NeighborsB(sn, arcs):
    # Return a list of the neighbors of supply node sn in arcs
    neighborSet = []
    for arc in arcs:
        if arc[1] == sn:
            neighborSet.append(arc[0])
    return neighborSet

def PrioritizeA(a, arcs):
    # Move arcs containing a to top of arc list
    retarcs = arcs.copy()
    for ind, val in enumerate(arcs):
        if val[0] == a:
            retarcs.insert(0, retarcs.pop(retarcs.index(val)))
    return retarcs

def PrioritizeB(b, arcs):
    # Move arcs containing a to top of arc list
    retarcs = arcs.copy()
    for ind, val in enumerate(arcs):
        if val[1] == b:
            retarcs.insert(0, retarcs.pop(retarcs.index(val)))
    return retarcs

# Generate set of SFP rates; give some wiggle room to avoid values very close to 0 or 1
random.seed(45)
TNorigRates = [random.uniform(0.02,0.98) for i in range(numTN)]
SNorigRates = [random.uniform(0.02,0.98) for i in range(numSN)]

# Initialize arcs to pull from and adjusted SFP rates
eps = 0.01
arcsetToDraw = arcset.copy()
TNadjRates = np.zeros(numTN)
SNadjRates = np.zeros(numSN)

# First adjustment
ind = TNset.index(arcsetToDraw[0][0])
TNadjRates[ind] = TNorigRates[ind]-eps

# Loop through arcsetToDraw
iter = 1
while arcsetToDraw and iter < 10*(numSN+numTN):
    currArc = arcsetToDraw[0]
    a, b = currArc[0],currArc[1]
    aind, bind = TNset.index(a), SNset.index(b)
    if TNadjRates[aind] != 0:
        for bprime in NeighborsA(a,arcsetToDraw):
            bprimeind = SNset.index(bprime)
            SNadjRates[bprimeind] = (SNorigRates[bprimeind]*(1-TNorigRates[aind]) +
                                     TNorigRates[aind]-TNadjRates[aind])/(1 - TNadjRates[aind])
            arcsetToDraw.remove((a, bprime))
            print('Removed arc ' + str((a, bprime)))
            #arcsetToDraw = PrioritizeB(bprime,arcsetToDraw)
    if SNadjRates[bind] != 0:
        for aprime in NeighborsB(b,arcsetToDraw):
            aprimeind = TNset.index(aprime)
            TNadjRates[aprimeind] = (TNorigRates[aprimeind]*(1-SNorigRates[bind]) +
                                     SNorigRates[bind]-SNadjRates[bind])/(1 - SNadjRates[bind])
            arcsetToDraw.remove((aprime, b))
            print('Removed arc ' + str((aprime, b)))
            #arcsetToDraw = PrioritizeA(aprime, arcsetToDraw)
    iter += 1
print(arcsetToDraw)

# Check that resulting rates give same z* values
toler = 0.0001
for arc in arcset:
    aind, bind = TNset.index(arc[0]), SNset.index(arc[1])
    z1 = TNorigRates[aind] + (1 - TNorigRates[aind]) * SNorigRates[bind]
    z2 = TNadjRates[aind] + (1 - TNadjRates[aind]) * SNadjRates[bind]
    if np.abs(z1-z2) > toler:
        print('Arc ' + str(arc) + ' is a problem')
    else:
        print(z1-z2)
















