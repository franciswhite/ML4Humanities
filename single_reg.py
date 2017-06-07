#Single Variable Regression Model

import numpy as np

class svr(object):
    '''Single variable regression.'''

    def __init__(self):

        '''Constructor.'''

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
        number_of_observations = 0    #initialize
        with open(data_path) as data:
            for line  in data:
                number_of_observations += 1
        return number_of_observations



    def cost_fct(self, data_path, intersect, co1):
        '''Computes some squared error, given data and single variable model.
        :param data_path: Raw data file, with quotes..
        :param intersect: Value of the intersect paramater.
        :param co1: Value of slope paramater.
        :return total_cost: Squared error, given data and model.
        '''
        total_error = 0 #initialize
        number_of_observations = self.number_observations(data_path)
        data_array = self.read_data(data_path)
        for observation in range(0, number_of_observations):
            total_error += ((intersect+data_array[observation][0]* co1) - data_array[observation][1])**2
        total_error = total_error / (2*number_of_observations)
        return total_error

    def minimize(self, data_path, learning_rate = 0.01, convergence_treshold = 0.01):
        '''Estimates the optimal parameters for SVR using Gradient Descent.
        :param data_path: Path to raw data.
        :param learning_rate: Coefficient of steps at each iteration.
        :return good_intersect:
        :return good_co1:'''
        good_intersect = -10 #initialize
        good_co1 = 10

        number_of_observations = self.number_observations(data_path)
        data_array = self.read_data(data_path)

        #Compute update
        total_observation_intersect = 0
        total_observation_co1 = 0
        keyword = "perform update"
        iteration_counter = 0 #Debugging
        while keyword == "perform update":
            for observation in range(0, number_of_observations):
                individual_observation_intersect = ((good_intersect + (data_array[observation][0] * good_co1)) - data_array[observation][1])
                total_observation_intersect += individual_observation_intersect

            for observation in range(0, number_of_observations):
                individual_observation_co1 = ((good_intersect + (data_array[observation][0] * good_co1)) - data_array[observation][1]) * data_array[observation][0]
                total_observation_co1 += individual_observation_co1

            temp_coeff = total_observation_co1 / number_of_observations
            temp_intersect = total_observation_intersect / number_of_observations

            if abs(good_intersect - (learning_rate * temp_intersect)) > convergence_treshold and abs(good_co1 - (learning_rate * temp_coeff)) > convergence_treshold:
                good_intersect = good_intersect - (learning_rate * temp_intersect)
                print(0, good_intersect - (learning_rate * temp_intersect))
                good_co1 = good_co1 - (learning_rate * temp_coeff)
                print(1, good_co1 - (learning_rate * temp_coeff))
                total_observation_intersect = 0
                total_observation_co1 = 0
            else:
                coefficients = np.array([good_intersect, good_co1])
                keyword = "stop updating"
            iteration_counter += 1  #Debugging

        return coefficients, iteration_counter

#Testing
test = svr()
#print(test.number_observations("toy_data_linear.py"))
#print(test.read_data("toy_data_linear.py")[2][0])
#print(test.cost_fct("toy_data_linear.py", 1, 1))
print(test.minimize("toy_data_linear.py")) #It seems that this is extremely sensitive to initalization.

#print(test.read_data("toy_data_linear.py")[2][1])