# Single Variable Regression Model

import numpy as np


class mvr(object):
    '''Multivariate linear regression.'''

    def __init__(self, number_coefficients):
        '''Constructor.
        :param number_coefficients: How many regressors (additionally to intercept).'''

        self.number_coefficients = number_coefficients
        pass

    def read_data(self, data_path):
        '''Read and process data for use in other methods.
        :param data_path: Path to data file.
        :returns: data_array: Np-array with data.
        '''
        with open(data_path) as data:
            temp_list = []
            for line in data:
                line = line.split()  # to deal with blank
                if line:  # lines (ie skip them)
                    line = [float(i) for i in line]
                    temp_list.append(line)
        data_array = np.array(temp_list)
        return data_array

    def number_observations(self, data_path):
        '''Compute number of data points.
        :param data_path: Path to data.
        :return number_of_observations
        '''
        number_of_observations = 0  # initialize
        with open(data_path) as data:
            for line in data:
                number_of_observations += 1
        return number_of_observations

    def cost_fct(self, theta, X, y):
        return sum((np.dot(X, theta) - y) ** 2) / (2 * len(X))

    def deriv(self, theta, X, y):
        m = len(X)          #Number of observations/rows
        deriv = []          #Initialize list

        for feature in range(0, m):
            deriv.append(sum((np.dot(X, theta) - y) * X[:, feature]) / m)

        deriv = np.array(deriv)
        deriv = np.transpose(deriv)
        return deriv


    def minimize(self, data_path, learning_rate=0.01, convergence_treshold=0.00001):
        '''Estimates the optimal parameters for MVR using Gradient Descent.
        :param data_path: Path to raw data.
        :param learning_rate: Coefficient of steps at each iteration.
        :return theta: Vector containing estimated coefficients.
        '''
        data_array = self.read_data(data_path)

        m = len(data_array) #Number of observations/rows

        X = np.column_stack([np.ones(m), data_array[:, 0]])  # Initialize matrix of observations: Frst column all 1's, second column data.
        for feature in range(1, self.number_coefficients):  #Subsequently add all the columns
            next_column = np.transpose(data_array[:, feature])
            X = np.column_stack([X, next_column])

        y = data_array[:, self.number_coefficients]    #Get vector of target variable

        print(X, "X")
        print(y, "y")


        theta = [0] * (self.number_coefficients+1) # Initialize parameter vector
        theta = np.array(theta)
        print(theta, "theta")

        new_theta = theta - learning_rate * self.deriv(theta, X, y)

        # while (np.sqrt(sum((theta - new_theta) ** 2)) > convergence_treshold):
        #     theta = new_theta
        #     new_theta = theta - learning_rate * self.deriv(theta, X, y)
        #
        # return theta
        #

# Testing
test = mvr(4)
# print(test.number_observations("toy_data_linear.py"))
a = test.read_data("toy_data_linear.py")
#print(len(a))
print(a)
# print(test.cost_fct("toy_data_linear.py", 1, 1))
test.minimize("toy_data_linear.py")

# print(test.read_data("toy_data_linear.py")[2][1])