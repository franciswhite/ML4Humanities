import numpy as np
import random
# x = np.array([2,2])
# a = np.linalg.norm(x, ord=2)

###Thangs to do:
#1. k-means: cluster assignment step, move centroid step
#2. cost function
#3. random initialization
#4. outer for loop

class k_means(object):
    '''K-means clustering algorithm, unsupervised learning.'''

    def __init__(self, m, data):
        '''Constructor.
        :param m: Number of training examples.
        :param data: A numpy array of feature values.
        '''
        self.m = m
        self.data = data

    def clustering(self, K):
        '''Cluster assignment and move centroid step. Looks for K clusters in data.
        :param K: Number of clusters algorithm looks for.
        :return mu_K: K-dimensional array with centroid positions.'''

        #Initialize K cluster_centroids by picking K points from dataset
        cluster_centroids = random.sample(self.data, K)

        #Build dictionary: key = number of centroid, value = position of centroid
        clusters_dictionary = {k+1 : cluster_centroids[k] for k in range(0,K)}   #Contains cluster-position value pairs
        print(clusters_dictionary)

        #Initializing indexation dictionary
        assignment_dictionary = {}  #Will contain datapoint-assignment value pairs

        for x in range(0, self.m):
            temp_0 = 1000000  # Initialize minimal distance
            for k in range(1,K+1):
                distance = np.linalg.norm((self.data[x] - clusters_dictionary[k]), ord=2)
                if distance <= temp_0:
                    assignment_dictionary.update({x : k})
                    temp_0 = distance
                else:
                    pass
        return assignment_dictionary


test_data = np.array([[-1.1],[-2],[-3],[14],[15],[16]])
test = k_means(6, test_data)
print(test.clustering(3))