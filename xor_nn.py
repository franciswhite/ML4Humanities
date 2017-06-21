import numpy as np
import math

def sigma(value):
    '''Computes Sigma values for a vector of inputs.
    :param value: A value.
    :return sigma_value: An outputs.
    '''
    sigma_value = (1/ (1+math.exp(-(value))))
    return sigma_value

#Compute z(i)
#Computes a(i) based on z(i) element-wise
#Loop around this to move through network

def compute_z(theta,a_vector):
    return np.dot(theta,a_vector)

def compute_a(vector_z):
    m = len(vector_z)
    init_a = np.ones(m+1)
    #init_a.append(1.0) #First value always 1

    for element in range(0,m):
        temp = sigma(vector_z[element])
        init_a[element+1] = temp
    #print(np.shape(init_a))
    return init_a

# t = np.array([-10,20,20])
# z = np.array([[1],
#              [0],
#              [1]])
#
#
theta_1 = np.array([[-10,20,-20],
                     [-10,-20,20]])

theta_2 = np.array([[-10,20,20]])
example_1 = 1
example_2 = 0
#
a_1 = np.array([[1],[0],[1]])
# z_2 = compute_z(theta_1,a_1)
# a_2 = compute_a(z_2)
# z_3 = compute_z(theta_2,a_2)
# a_3 = compute_a(z_3)

random_theta_1 = np.random.rand(2, 3)
random_theta_2 = np.random.rand(1, 3)


def predict(theta_1, theta_2, input):
    '''Compute binary prediction based on two variables as input and a matrices of weights.
    :param theta_1: First set of weights
    :param theta_2: Second set of weights
    :param input: Vector of 1,x_1,x_2
    :return prediction: A value (0,1)'''
    a_1 = input
    z_2 = compute_z(theta_1, a_1)
    a_2 = compute_a(z_2)
    z_3 = compute_z(theta_2, a_2)
    a_3 = compute_a(z_3)
    return a_3[1]


#test = predict(theta_1, theta_2 ,a_1)


def delta_3(theta_1, theta_2, input, labeled_solution):
    delta_3 = predict(theta_1,theta_2,input) - labeled_solution
    return delta_3

def delta_2(theta_1,theta_2, input, labeled_solution, a_2):
    temp_0 = np.transpose(theta_2)*delta_3(theta_1,theta_2,input,labeled_solution)
    temp_1 = np.multiply(a_2, (1-a_2))
    delta_2 = np.multiply(temp_0, temp_1)
    return delta_2

init_big_delta = np.array([0, 0, 0])

def big_delta_2(init_big_delta, delta_3, a_2):
    update_vector = a_2*delta_3
    big_delta = init_big_delta + np.transpose(update_vector)
    return big_delta

test_2 = delta_3(random_theta_1, random_theta_2, a_1, example_1)
# print(test_2)
a_2 = np.array([[1],[2],[3]])

test_3 = big_delta_2(init_big_delta, test_2, a_2)
print(test_3)

#print(delta_2(theta_1,theta_2, a_1, example_2, a_2))
# x = np.transpose(theta_2) * 0.5
# print(x)

