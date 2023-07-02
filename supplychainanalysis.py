"""
Initial script for sourcing matrix (Q) analysis
"""
import numpy as np
import scipy.optimize as spo
import scipy.stats as spstat
import scipy.special as sps

class prior_flatsourcing:
    """
    Defines the class instance of priors for a sourcing vector with a flat distribution across all supply nodes, and
    the following methods:
        rand: generate random draws from the distribution
        lpdf: log-likelihood of a given vector
        lpdf_jac: Jacobian of the log-likelihood at the given vector
        lpdf_hess: Hessian of the log-likelihood at the given vector
    beta inputs may be a Numpy array of vectors
    """
    def __init__(self, numSNs=2):
        self.mu = np.ones(numSNs) / numSNs

    def rand(self, numsamps=1, numvecs=1):
        return np.random.dirichlet(alpha=self.mu) (n=numsamps, pvals=mu, size=numvecs)

    def lpdf(self, pi):
        if pi.ndim == 1: # reshape to 2d
            pi = np.reshape(pi,(1,-1))

        lik = -(1/(2*self.var)) * np.sum((beta - (self.mu))**2,axis=1) - np.log(self.var*2*np.pi)*np.size(beta)/2
        return np.squeeze(lik)
    def lpdf_jac(self,beta):
        if beta.ndim == 1: # reshape to 2d
            beta = np.reshape(beta,(1,-1))
        jac = -(1/self.var) * (beta - self.mu)
        return np.squeeze(jac)
    def lpdf_hess(self,beta):
        if beta.ndim == 1: # reshape to 2d
            beta = np.reshape(beta,(1,-1))
        k,n = len(beta[:,0]),len(beta[0])
        hess = np.tile(np.zeros(shape=(n,n)),(k,1,1))
        for i in range(k):
            hess[i] = np.diag(np.repeat(-(1/self.var),n))
        return np.squeeze(hess)


# Use a Dirichlet-Multinomial conjugate?
class Dirichlet(object):
    def __init__(self, alpha):
        from math import gamma
        from operator import mul
        self._alpha = np.array(alpha)
        self._coef = gamma(np.sum(self._alpha)) / \
                           np.multiply.reduce([gamma(a) for a in self._alpha])
    def pdf(self, x):
        """Returns pdf value for x"""
        from operator import mul
        return self._coef * np.multiply.reduce([xx ** (aa - 1)
                                               for (xx, aa)in zip(x, self._alpha)])
    def rand(self, n):
        """Returns random draws from this distribution"""
        return np.random.dirichlet(self._alpha, size=n)

numdata = 180
vec = np.random.multinomial(numdata,np.array([0.1,0.6,0.3]))

dirObj = Dirichlet([vec[0]+1/3, vec[1]+1/3, vec[2]+1/3])
temp = dirObj.rand(1000)

for i in range(temp.shape[1]):
    plt.hist(temp[:,i],alpha=0.2)
plt.show()
plt.close()

def initflatdirichlet(Q):
    """Initialize a list of Dirichlet objects for each test node using the observations contained in Q"""
    return [Dirichlet(Q[i] + np.ones(Q.shape[1])/Q.shape[1]) for i in range(Q.shape[0])]

def getranddraws(Qobjlist):
    retlist = []
    for dirobj in Qobjlist:
        retlist.append(dirobj.rand(1))
    return np.squeeze(np.array(retlist))

Q = np.array([[6, 11],
              [12, 6],
              [2, 13]])

Qobjlist = initflatdirichlet(Q)
getranddraws(Qobjlist)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
AREA = 0.5 * 1 * 0.75**0.5
triangle = tri.Triangulation(corners[:, 0], corners[:, 1])


refiner = tri.UniformTriRefiner(triangle)
trimesh = refiner.refine_triangulation(subdiv=4)

plt.figure(figsize=(8, 4))
for (i, mesh) in enumerate((triangle, trimesh)):
    plt.subplot(1, 2, i+ 1)
    plt.triplot(mesh)
    plt.axis('off')
    plt.axis('equal')

# For each corner of the triangle, the pair of other corners
pairs = [corners[np.roll(range(3), -i)[1:]] for i in range(3)]
# The area of the triangle formed by point xy and another pair or points
tri_area = lambda xy, pair: 0.5 * np.linalg.norm(np.cross(*(pair - xy)))

def xy2bc(xy, tol=1.e-4):
    '''Converts 2D Cartesian coordinates to barycentric.'''
    coords = np.array([tri_area(xy, p) for p in pairs]) / AREA
    return np.clip(coords, tol, 1.0 - tol)



def draw_pdf_contours(dist, nlevels=200, subdiv=8, **kwargs):
    import math

    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    pvals = [dist.pdf(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]

    plt.tricontourf(trimesh, pvals, nlevels, cmap='jet', **kwargs)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')
    plt.show()
    plt.close()

draw_pdf_contours(Dirichlet([5, 5, 5]))






