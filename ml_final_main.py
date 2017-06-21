import numpy as np
import string
from word_indexing import word_indexer
from k_means import k_means

data_path = "//home//sh//Desktop//june_project//data_quine//all_texts//1953e_On Mental Entities_Quine (1).txt"
data_points = word_indexer(data_path)
print(data_points)

test1 = k_means(3, data_points)
initialize = test1.cluster_centroid_initialization(2)
print(initialize)
clustering_step = test1.clustering(2, initialize)
print(clustering_step)
runit = test1.find_clusters(2)
print(runit)
cost = test1.distorsion_function(2, runit)
print(cost)