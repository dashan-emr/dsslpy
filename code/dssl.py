import autograd.numpy as np
from autograd import grad
from sklearn.base import BaseEstimator
from scipy.optimize import minimize

def huber(x,h):
    """
        Huber loss
    """
    assert 0 < h and h < 1
    condlist = [x < -h, np.abs(x) <= h, x > h]
    choicelist = [0.0, (h+x)**2/(4*h), x]
    return np.select(condlist,choicelist)

class DSSL(BaseEstimator):
    """
        DSSL class
    """
    def __init__(self,l2_reg=0.0,smoothness_reg=0.0,h=0.01,maxiter=1000,tol=1e-8,gtol=1e-6,disp=False):
        """
            Args:
                l2_reg - float:
                    L2 regularization parameter
        
                smoothness_reg - float:
                    Weight given to the temporal smoothness objective
        

                h - float:
                    Huber loss parameter. Must be between 0 and 1. Values closer
                    to zero give a closer approximation to the hinge loss.
        
                maxiter - int:
                    Maximum number of optimization iterations
        
                tol - float:
                    Loss tolerance (see scipy.optimize.minimize doc for details)
        
                gtol - float:
                    Gradient magnitude tolerance (see scipy.optimize.minimize 
                    doc for details)
                
                disp - bool:
                    Set to True to display optimization iterations.
        """
        self.__dict__.update(locals())
        
    def loss(self,ranked_pair_diffs,smoothed_pair_diffs,T_diffs):
        """
            DSSL loss
        """
        f = 0.0
        
        # mean huber loss of the difference in scores between ranked pairs
        f += np.sum(huber(1.0-np.dot(ranked_pair_diffs,self.w),self.h))/(ranked_pair_diffs.shape[0] + 1e-12)
        
        # mean squared normalized difference in scores between smoothed pairs
        smoothed_pair_score_diffs = np.dot(smoothed_pair_diffs,self.w)
        f += self.smoothness_reg*np.sum((smoothed_pair_score_diffs/T_diffs)**2)/(smoothed_pair_diffs.shape[0] + 1e-12)
        
        # \ell_2 regularization
        f += self.l2_reg*np.dot(self.w,self.w)/2.0
        
        return f
    
    def set_params(self,w):
        self.w = w
        
    def get_obj(self,X,T,ranked_pairs,smoothed_pairs):
        # precalculate differences between pairs
        if ranked_pairs is None: 
            ranked_pair_diffs = np.zeros((0,X.shape[1]))
        else:
            ranked_pair_diffs = X[ranked_pairs[:,0]] - X[ranked_pairs[:,1]]
        if smoothed_pairs is None:
            smoothed_pair_diffs = np.zeros((0,X.shape[1]))
            T_diffs = np.array([])
        else:
            smoothed_pair_diffs = X[smoothed_pairs[:,0]] - X[smoothed_pairs[:,1]]
            T_diffs = T[smoothed_pairs[:,0]] - T[smoothed_pairs[:,1]]
        
        def obj(w):
            self.set_params(w)
            return self.loss(ranked_pair_diffs,smoothed_pair_diffs,T_diffs)
            
        return obj
        
    def fit(self,X,T,ranked_pairs,smoothed_pairs):
        """
            Fit the DSSL loss
            Args:
                X - (n_samples,n_features) ndarray:
                    Design matrix
        
                T - (n_samples,) ndarray of:
                    Vector of continuous timestamps
        
                ranked_pairs - (n_ranked_pairs,2) integer ndarray:
                    Contains ranked pairs of samples. Model will try to find 
                    parameters such that score(ranked_pairs[i,0]) > score(ranked_pairs[i,1])
                    for all i.
            
                smoothed_pairs - (n_smoothed_pairs,2) integer ndarray:
                    Contains pairs of samples that are close in time. Model will 
                    try to find parameters such that minimizes 
                    (score(ranked_pairs[i,0]) - score(ranked_pairs[i,1]))**2/(T(ranked_pairs[i,0]) - T(ranked_pairs[i,1]))**2
                    for all i.
        """
        assert X.shape[0] > 0
        assert T.shape == (X.shape[0],)
        assert ranked_pairs is None or np.issubdtype(ranked_pairs.dtype, np.dtype(int).type)
        assert smoothed_pairs is None or np.issubdtype(smoothed_pairs.dtype, np.dtype(int).type)
        
        assert ranked_pairs is None or np.all(np.logical_and(ranked_pairs >= 0,ranked_pairs <= X.shape[0]))
        assert smoothed_pairs is None or np.all(np.logical_and(smoothed_pairs >= 0,smoothed_pairs <= X.shape[0]))
        
        assert ranked_pairs is None or np.all(ranked_pairs[:,0] != ranked_pairs[:,1])
        assert smoothed_pairs is None or np.all(smoothed_pairs[:,0] != smoothed_pairs[:,1])
        
        # get obj
        obj = self.get_obj(X,T,ranked_pairs,smoothed_pairs)
        
        # get the gradient function using autograd  
        gfun = grad(obj)
        
        # init params
        w0 = np.zeros(X.shape[1])
        
        # optimize objective
        self.res = minimize(obj,w0,method="L-BFGS-B",jac=gfun,options={"gtol":self.gtol,"maxiter":self.maxiter,"disp":self.disp},tol=self.tol)
        
        self.set_params(self.res.x)
        
        return self
        
    def predict(self,X):
        """
            Calculate scores
        """
        return np.dot(X,self.w)