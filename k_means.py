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
    '''K-means clustering algorithm, unsupervised learning. Most of code assumes only one feature.'''

    def __init__(self, m, data):
        '''Constructor.
        :param m: Number of training examples.
        :param data: A numpy array of feature values.
        '''
        self.m = m
        self.data = data

    def cluster_centroid_initialization(self, K):
        #Initialize K cluster_centroids by picking K points from dataset
        cluster_centroids = random.sample(self.data, K)
        return cluster_centroids

    def clustering(self, K, cluster_centroids):
        '''Cluster assignment and move centroid step. Looks for K clusters in data.
        :param K: Number of clusters algorithm looks for.
        :return cluster_centroid_output: K-dimensional array with centroid positions.'''

        #Build dictionary: key = number of centroid, value = position of centroid
        clusters_dictionary = {k+1 : cluster_centroids[k] for k in range(0,K)}   #Contains cluster-position value pairs

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
        #Move centroid step
        for k in range(1, K+1):
            list_of_points = []
            for x, cluster in assignment_dictionary.iteritems():
                if cluster == k:
                    list_of_points.append(self.data[x])

            average_point = sum(list_of_points)/len(list_of_points)
            clusters_dictionary[k] = average_point

        cluster_centroid_output = np.arange(K)
        for k in range(0,K):
            cluster_centroid_output[k] = clusters_dictionary[k+1]
        return cluster_centroid_output

    def find_clusters(self, K):
        '''Finds locally optimal clusters.
        :param K: Number of clusters.
        :return final_cluster_centroids: K dimensional array with final centroid positions.'''
        init = self.cluster_centroid_initialization(K)
        iterations = 100
        for i in range(0,iterations):
            temp0 = self.clustering(K, init)
            init = temp0
        return temp0


# test_data = np.array([[-1.1],[-2],[-3],[14],[15],[16]])
# test = k_means(6, test_data)
#
# init = test.cluster_centroid_initialization(2)
# print(test.clustering(2, init))

test_data_2 = np.array([[2],[4],[80],[82]])
test2 = k_means(4, test_data_2)
initialize = test2.cluster_centroid_initialization(2)
print(initialize)
clustering_step = test2.clustering(2, initialize)
print(clustering_step)
runit = test2.find_clusters(2)
print(runit)