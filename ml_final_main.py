import numpy as np
import string
from word_indexing import word_indexer
from k_means import k_means

#Corpus to statistic
data_path = "/home/sh/Desktop/june_project/data_quine/all_texts/toy_text.txt"
data_points = word_indexer(data_path)

#Statistic to K-means object
m = len(data_points)
#K = 4
test1 = k_means(m, data_points)
#run = test1.robust_centroids(K)



#Plot K - Cost
import matplotlib.pyplot as plt

init_K = []
init_cost = []
for i in range(1,11):
    init_K.append(i)
    init_cost.append(test1.robust_centroids(i))
    print(init_K, init_cost)

plt.plot(init_K, init_cost, 'ro')
plt.axis([0, 10, 0, 200])
plt.xlabel('K')
plt.ylabel('Mean Squared Error')
plt.title('Plot: Number of Clusters vs. Distorsion')
plt.show()
