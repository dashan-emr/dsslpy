import numpy as np
from dssl import DSSL

###########################
# Gen data
###########################
N = 100
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

###########################
# Fit model
###########################
smoothness_reg = 1.0
l2_reg = 1.0

dssl = DSSL(l2_reg=l2_reg,smoothness_reg=smoothness_reg,disp=True)
dssl.fit(X,T,ranked_pairs,smoothed_pairs)

###########################
# Get scores
###########################
scores = dssl.predict(X)