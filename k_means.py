import numpy as np
import random
from word_indexing import word_indexer
import operator
from math import pow
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
            for k in range(1, K+1):
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

    def distorsion_function(self, K, cluster_centroids):
        '''Compute cost given data and K cluster centroids with their associated datapoints.
        First, compute dictionaries again; second, compute cost.
        :param cluster_centroids: K cluster centroids.
        :param K: Number of cluster centroids.
        :return total_distorsion: Mean squared distance from assigned cluster centroids.'''

        clusters_dictionary = {k+1 : cluster_centroids[k] for k in range(0,K)}   #Contains cluster-position value pairs
        assignment_dictionary = {}  #Will contain datapoint-assignment value pairs
        for x in range(0, self.m):
            temp_0 = 1000000  # Initialize minimal distance
            for k in range(1,K+1):
                distance = np.linalg.norm((self.data[x] - clusters_dictionary[k]), ord=2)
                if distance <= temp_0:
                    assignment_dictionary.update({x : k})
                    temp_0 = distance

        total_distorsion = 0.0      #Initialize total cost
        for x in range(0, self.m):
            associated_cluster = assignment_dictionary[x]
            distance2 = self.data[x] - clusters_dictionary[associated_cluster]
            total_distorsion += np.linalg.norm(distance2, ord=2)
        total_distorsion = total_distorsion / self.m
        return total_distorsion

    def robust_centroids(self, K):
        '''Finds K centroids for 100 different initalizations. Picks that set of centroids with minimal distorsion.
        :param K: Number of centroids.
        :return optimal_centroids: Array of optimal centroids.
        :return optimal_cost: Distorsion optimal centroids incur.'''
        init_cost = pow(10,100)
        for i in range(0,100):
            temp_clusters = self.find_clusters(K)
            temp_cost = self.distorsion_function(K, temp_clusters)

            if temp_cost < init_cost:
                optimal_centroids = temp_clusters
                init_cost = temp_cost
            else:
                pass
        return init_cost#, optimal_centroids        Now just outputs cost, could also output actual position of centroids!

test_data = np.array([[-1.1],[-2],[-3],[14],[15],[16]])
test = k_means(6, test_data)
#
# init = test.cluster_centroid_initialization(2)
# print(test.clustering(2, init))
#
# test_data_2 = word_indexer("//home//sh//Desktop//june_project//data_quine//all_texts//1953e_On Mental Entities_Quine (1).txt")
#
# test2 = k_means(3, test_data_2)
# initialize = test2.cluster_centroid_initialization(2)
# print(initialize)
#
# clustering_step = test2.clustering(2, initialize)
# print(clustering_step)
# runit = test2.find_clusters(2)
# print(runit)
# cost = test2.distorsion_function(2, runit)
# print(cost)
#
# a = test.robust_centroids(2)
# print(a)

