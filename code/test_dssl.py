import numpy as np
from dssl import huber,DSSL
import unittest

class TestDSSL(unittest.TestCase):
            
    def test_toy(self):
        """
            Test the loss on a couple of toy examples
        """
        l2_reg = 0.75
        smoothness_reg = 0.0
        h = 0.01
        X = np.array([[1,2],
                      [3,4],
                      [5,6]])
                      
        ranked_pairs = np.array([[0,1],
                                 [2,1]])
                                 
        smoothed_pairs = None
        
        w = np.array([1.0,-1.0])
        
        T = np.arange(3)
        
        l_true = (huber(1-np.dot(X[0]-X[1],w),h) + huber(1-np.dot(X[2]-X[1],w),h))/2.0
        l_true += l2_reg * np.sum(w*w)/2.0
        
        dssl = DSSL(l2_reg=l2_reg)
        obj = dssl.get_obj(X,T,ranked_pairs,smoothed_pairs)
        
        l_dssl = obj(w)
        
        self.assertAlmostEqual(l_true,l_dssl)
        

        l2_reg = 0.75
        smoothness_reg = 1.0
        h = 0.01
        X = np.random.randn(4,2)
                      
        ranked_pairs = np.array([[0,1],
                                 [2,1]])
                                 
        smoothed_pairs = np.array([[1,0],
                                   [0,2]])
                             
        
        w = np.random.randn(2)
        
        T = np.arange(4)
        
        l_true = (huber(1-np.dot(X[0]-X[1],w),h) + huber(1-np.dot(X[2]-X[1],w),h))/2.0
        l_true += smoothness_reg*(np.dot(X[1]-X[0],w)**2 + np.dot(X[0]-X[2],w)**2/4.0)/2.0
        l_true += l2_reg * np.sum(w*w)/2.0
        
        dssl = DSSL(l2_reg=l2_reg,smoothness_reg=smoothness_reg)
        obj = dssl.get_obj(X,T,ranked_pairs,smoothed_pairs)
        
        l_dssl = obj(w)
        
        self.assertAlmostEqual(l_true,l_dssl)
        
    def test_fit_toy(self):
        """
            Ranked pairs is created such that X is perfectly ordered and
            the final loss should be zero.
        """
        N = 10
        X = np.random.randn(N,2)
        T = np.arange(N)
        
        w = np.random.randn(2)
        
        scores = np.dot(X,w)
        
        ranked_pairs = []
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                if scores[i] < scores[j]:
                    ranked_pairs.append([i,j])
                    
        ranked_pairs = np.array(ranked_pairs)
        
        smoothed_pairs = None
        
        smoothness_reg = 0.0
        l2_reg = 0.0
        
        dssl = DSSL(disp=False)
        dssl.fit(X,T,ranked_pairs,smoothed_pairs=None)
        
        self.assertAlmostEqual(0.0,dssl.res.fun)
        
    def test_fit(self):
        """
            Test ability to run on reasonable data
        """
        N = 1000
        X = np.random.randn(N,2)
        T = np.arange(N)
        
        w = np.random.randn(2)
        
        scores = np.dot(X,w)
        
        ranked_pairs = []
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                if scores[i] < scores[j] and np.random.rand() < 2.0/N:
                    ranked_pairs.append([i,j])
                    
        ranked_pairs = np.array(ranked_pairs)
        
        smoothed_pairs = np.vstack([np.random.choice(100,N,replace=True),np.random.choice(100,N,replace=True)]).T
        smoothed_pairs = smoothed_pairs[smoothed_pairs[:,0] != smoothed_pairs[:,1]]
        
        smoothness_reg = 1.0
        l2_reg = 1.0
        
        dssl = DSSL(l2_reg=l2_reg,smoothness_reg=smoothness_reg,disp=False)
        dssl.fit(X,T,ranked_pairs,smoothed_pairs)
        # print(dssl.w)
        
if __name__ == "__main__":
    unittest.main()