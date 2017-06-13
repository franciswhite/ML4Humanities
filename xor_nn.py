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
z = np.array([[1],
             [0],
             [1]])
#
# test = compute_z(t,a)
# print(test)
#print(compute_a(z))

# x = np.array([[1, 5, 6, 7]])
# y = np.transpose(x)
# print(y)

theta_1 = np.array([[-10,20,-20],
                    [-10,-20,20]])

theta_2 = np.array([[-10,20,20]])

a_1 = np.array([[1],[0],[1]])

z_2 = compute_z(theta_1,a_1)

a_2 = compute_a(z_2)

z_3 = compute_z(theta_2,a_2)

a_3 = compute_a(z_3)

print(a_3[1])
# solutions = sigma(a_3[1])
# print(solutions)

