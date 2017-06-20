import numpy as np
x = np.array([2,2])
a = np.linalg.norm(x, ord=2)
print(a)

###Thangs to do:
#1. k-means: cluster assignment step, move centroid step
#2. cost function
#3. random initialization
#4. outer for loop

class k_means(object):
    '''K-means clustering algorithm, unsupervised learning.'''

    def __init__(self, m):
        '''Constructor.
        :param m: Number of training examples.
        '''
        self.m = m

    def clustering(self, K):
        '''Cluster assignment and move centroid step. Looks for K clusters in data.
        :param K: Number of clusters algorithm looks for.
        :return mu_K: K-dimensional array with centroid positions.'''
        for i in range(0, self.m):
            print(i)

test = k_means(10)
print(test.clustering(2))

