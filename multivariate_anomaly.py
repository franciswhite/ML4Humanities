import numpy as np
import matplotlib.pyplot as plt

mean1 = [100,50]
covariance1 = [[10,1],[3,35]]
x = np.random.multivariate_normal(mean1, covariance1)

x, y = np.random.multivariate_normal(mean1, covariance1, 5000).T #x, y are 5000 dimensional row vector arrays
plt.plot(x, y, 'x')
plt.axis('equal')
data = np.array([x,y])
data_test = np.array([[1,2,3],[100,101,102]])
#plt.show()

class multivariate_model(object):

    def __init__(self, data):
        self.observation_number = len(data[0])
        self.feature_number = len(data)
        self.data = data

    def mean_estimator(self):
        mean_vector = np.zeros(shape=(self.feature_number,1))
        for feature in range(0, self.feature_number):
            init_mean = 0.0
            for observation in range(0, self.observation_number):
                init_mean += self.data[feature, observation]/float(self.observation_number)
            mean_vector[feature] = init_mean

        return mean_vector

    def covariance_matrix(self, mean_vector):
        covariance_matrix = np.zeros(shape=(self.feature_number,self.feature_number))
        for observation in range(0, self.observation_number):
            temp0 = self.data[:,observation] - mean_vector
            temp1 = temp0 * temp0[np.newaxis].T
            covariance_matrix += temp1
        return covariance_matrix

test = multivariate_model(data)
me = test.mean_estimator()
print(test.covariance_matrix(me))


