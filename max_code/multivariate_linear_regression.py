import numpy as np
def get_data(datapath):
    """

    :param datapath: path to a .txt file containing data
    :return: a list of the data
    """
    datalist=[]
    raw_data=open(datapath)
    for i in raw_data:
        line=i.split()
        datalist.append(line)
    return datalist

def get_predictors(datalist):
    """

    :param data: a list of the data
    :return: an array of predictor values
    """
    temp_predictors=[]
    for line in datalist:
        line=[x for x in line[:-1]]
        #line=[line[:-1]]
        line=['1']+line #this enables vector and matrix methods
        line=[float(i) for i in line]
        temp_predictors.append(line)

    predictors=np.array( temp_predictors)

    return predictors

def get_independents(datalist):
    """

    :param data: a list of the data
    :return: an array of independent values
    """
    temp_independents=[]
    for line in datalist:
        temp_independents.append(float(line[-1]))
        independents=np.array(temp_independents)
    return independents

def scale(predictors):
    """
    :param predictors: an array of predictor values
    :return: a scaled array of predictor values
    """
    temp_predictor_means=[]
    for i in range(predictors.shape[1]): #first we center the means
        if i==0:
            predictor_mean=[0]
            temp_predictor_means=temp_predictor_means+predictor_mean
        if i>0:
            predictor_mean= [np.sum(predictors[:,i])/predictors.shape[0]]
            #print(predictor_mean)
            temp_predictor_means=temp_predictor_means+predictor_mean
            #print(temp_predictor_means)
    predictor_means=np.array([temp_predictor_means]*predictors.shape[0])
    #print(predictor_means)
    predictors=predictors-predictor_means
    #print(predictors)
    #next we divide by maxmin difference to scale predictors
    temp_scaled_predictors=[]
    for i in range (predictors.shape[1]):
        if i==0:
            maxmin_difference=1
        if i>0:
            maxmin_difference=np.max(predictors[:,i])-np.min(predictors[:,i])
            #maxmin_difference=np.max(predictor_means[:,i])-np.min(predictor_means[:,i])
        scaled_predictor=[x/maxmin_difference for x in predictors[:,i]]
        temp_scaled_predictors=temp_scaled_predictors+[scaled_predictor]
        #print(temp_scaled_predictors)
    predictors=np.transpose(np.array(temp_scaled_predictors))
    #print(predictors)
    return predictors

def scaleind(independents):
    """
    :param predictors: an array of independent values
    :return: a scaled array of independent values
    """
    #mean normalization
    #print(independents)
    independents_mean=np.sum(independents)/independents.shape[0]
    independents_means=np.array([independents_mean]*independents.shape[0])
    #print(independents_means)
    independents=independents-independents_means
    #print(independents)
    #Scaling
    maxmin_difference=np.max(independents)-np.min(independents)
    scaled_independents=[x/maxmin_difference for x in independents]
    independents=np.array(scaled_independents)
    #print(independents)
    return(independents)

def multiply_predictors(predictors, i, j):
    """

    :param predictors: an array of predictors
    :param i: index of first predictor to be merged
    :param j: index of second predictor to be merged
    :return: array of predictors with i,j merged
    """
    multiply=predictors[:,i]*predictors[:,j]
    #print(multiply)

    #print(np.delete(predictors,[i,j], axis=1))
    temp_predictors=[x for x in np.delete(predictors, [i,j], axis=1)]
    #print(temp_predictors)
    predictors=np.c_[np.array(temp_predictors),multiply]
    return predictors

def add_predictors(predictors, i, j):
    """

    :param predictors: an array of predictors
    :param i: index of first predictor to be merged
    :param j: index of second predictor to be merged
    :return: array of predictors with i,j merged
    """
    add=predictors[:,i]+predictors[:,j]
    temp_predictors=[x for x in np.delete(predictors, [i,j], axis=1)]
    predictors=np.c_[np.array(temp_predictors),add]
    return predictors

def square_predictor(predictors, i):
    """

    :param predictors: an array of predictors
    :param i: index of first predictor to be manipulated
    :return: array of predictors with manipulated i appended to the end
    """
    square=predictors[:,i]*predictors[:,i]
    #temp_predictors=[x for x in np.delete(predictors, [i], axis=1)]
    predictors=np.c_[np.array(predictors),square]
    return predictors

def cube_predictor(predictors, i):
    """

    :param predictors: an array of predictors
    :param i: index of first predictor to be manipulated
    :return: array of predictors with manipulated i appended to the end
    """
    cube=predictors[:,i]*predictors[:,i]*predictors[:,i]
    #temp_predictors=[x for x in np.delete(predictors, [i], axis=1)]
    predictors=np.c_[np.array(predictors),cube]
    return predictors

def sqrt_predictor(predictors,i):
    """

    :param predictors: an array of predictors
    :param i: index of first predictor to be manipulated
    :return: array of predictors with manipulated i appended to the end
    """
    sqrt=np.sqrt(predictors[:,i])
    #temp_predictors=[x for x in np.delete(predictors, [i], axis=1)]
    predictors=np.c_[np.array(predictors),sqrt]
    return predictors

def linear_predictions(predictors, parameters):
    """

    :param predictors: an array of predictors
    :param parameters: an array of parameters
    :return: an array of predictions for linear regression
    """
    predictions=np.dot(predictors, parameters)
    return(predictions)

def logistic_predictions(predictors, parameters):
    """

    :param predictors: an array of predictors
    :param parameters: an array of parameters
    :return: an array of predictions for logistic regression
    """
    predictions=1/(1+np.exp((-1)*(np.dot(predictors, parameters))))
    return predictions

def linear_cost(linear_predictions, independents):
    """

    :param linear_predictions: an array of predictions for linear regression
    :param independents: an array of independent variable values
    :return: the cost value (scalar)
    """
    cost= 1/linear_predictions.shape[0]*np.sum(1/2*(linear_predictions-independents)**2)
    return cost


def logistic_cost(logistic_predictions, independents):
    """

    :param logistic_predictions: an array of predictions for linear regression
    :param independents: an array of independent variable values
    :return: the cost value (scalar)
    """
    cost=(-1)/logistic_predictions.shape[0]*(np.dot(independents, np.log(logistic_predictions)+(np.dot((1-independents),np.log(1-logistic_predictions)))))
    return cost

def linear_cost_derivative(predictors, predictions, independents, checker):
    """

    :param predictors: An array containing predictor values
    :param predictions: An array of predictions
    :param independents: An array containting independent values
    :param: checker: a list of convergence checkers
    :return: Array of partial derivatives of cost function for linear regression
    """
    temp_cost_derivatives=[]
    #print(parameters)
    for i in range(predictors.shape[1]):
        if checker[i]==0:
            cost_derivative=0
            temp_cost_derivatives=temp_cost_derivatives+[cost_derivative]
        if checker[i]==1:
            cost_derivative= np.dot(np.transpose(predictors[:,i]),(predictions.shape[0]*(predictions-independents)))
            #print(cost_derivative)
            temp_cost_derivatives=temp_cost_derivatives+[cost_derivative]
    cost_derivatives=np.array(temp_cost_derivatives)
    #print(cost_derivatives)
    return cost_derivatives

def logistic_cost_derivative(predictors, predictions, independents, checker):
    """

    :param predictors: An array containing predictor values
    :param predictions: An array of predictions
    :param independents: An array containting independent values
    :param checker: a list of convergence checkers
    :return: Array of partial derivatives of cost function for logistic regression
    """
    temp_cost_derivatives=[]
    for i in range(predictors.shape[1]):
        if checker[i]==0:
            cost_derivative=0
            temp_cost_derivatives=temp_cost_derivatives+[cost_derivative]
        if checker[i]==1:
            cost_derivative= np.dot(np.transpose(predictors[:,i]),(predictions.shape[0]*(predictions-independents)))
            temp_cost_derivatives=temp_cost_derivatives+[cost_derivative]
    cost_derivatives=np.array(temp_cost_derivatives)
    return cost_derivatives

def gradient_descent(parameters, cost_derivatives, checker, alpha=0.001):
    """

    :param parameters: An array containing parameter values
    :param cost: An array containting cost derivative values
    :param checker: a list of convergence checkers
    :param alpha: a float representing the learning rate
    :return: updated parameters after performing one step of gradient descent
    """
    temp_parameters=[i for i in parameters]
    for i in range(len(checker)):
        if checker[i]==1:
            temp_parameters[i]=parameters[i]-alpha*cost_derivatives[i]
    temp_parameters=np.array(temp_parameters)
    #print(parameters)
    return temp_parameters


    # counter=0
    # while checker!=[0]*predictors.shape[1]:#Stop when convergence occurs
    #     #Using vector and matrix methods
    #     predictions=np.dot(predictors, parameters)
    #     for i in range(predictors.shape[1]):
    #         if checker[i]==1:
    #             derivative=np.dot(np.transpose(predictors[:,i]),(predictions.shape[0]*(predictions-independents)))
    #             temp_parameters[i]=parameters[i]-alpha*derivative
    #         if abs(temp_parameters[i]-parameters[i])<epsilon:#check for convergence
    #             checker[i]=0
    #        # print(temp_parameters)
    #     parameters=np.array(temp_parameters)
    #     counter+=1
    #     print(counter)
    #     #print(parameters)
    # return parameters

def convergence_checker(checker, temp_parameters, parameters, epsilon=0.0001):
    """
    :param checker: a list of convergence switches
    :param temp_parameters: An array containing parameter values
    :param parameters: An array containting parameters
    :param epsilon: a convergence criterion
    :return: updated checker
    """

    for i in range(len(parameters)):
        if abs(temp_parameters[i]-parameters[i])<epsilon:#check for convergence
            checker[i]=0
    return checker


#def update_parameters(temp_parameters):
 #   parameters=temp_parameters
  #  return parameters

def linear_regression(predictors, independents):
    """

    :param predictors: an array of predictors
    :param independents: an array of independent values
    :return: Optimal parameters arrived at through linear regression using gradient descent
    """
    counter=0
    checker=[1]*predictors.shape[1]
    parameters=[0]*predictors.shape[1]
    while checker!=[0]*predictors.shape[1]:
        temp_parameters=gradient_descent(parameters, linear_cost_derivative(predictors, linear_predictions(predictors, parameters), independents, checker), checker, alpha=0.01)
        #print(temp_parameters)
        #print(parameters)
        convergence_checker(checker, temp_parameters, parameters, epsilon=0.0001)
        parameters=temp_parameters
        #print(parameters)
        counter+=1
        #print(checker)
        print(counter)
        #print(parameters)
    return parameters

def logistic_regression(predictors, independents):
    """

    :param predictors: an array of predictors
    :param independents: an array of binary/categorical independent values
    :return: Optimal parameters arrived at through logistic regression using gradient descent
    """
    counter=0
    checker=[1]*predictors.shape[1]
    parameters=[0]*predictors.shape[1]
    while checker!=[0]*predictors.shape[1]:
        temp_parameters=gradient_descent(parameters, logistic_cost_derivative(predictors, logistic_predictions(predictors, parameters), independents, checker), checker, alpha=0.01)
        convergence_checker(checker, temp_parameters, parameters, epsilon=0.001)
        parameters=temp_parameters
        counter+=1
        #print(checker)
        print(counter)
        #print(parameters)
    #print(parameters)

    return parameters




# def gradient_descent(predictors, independents, alpha=0.001, epsilon=0.000001):
#     """
#
#     :param predictors: An array containing predictor values
#     :param independents: An array containting independent values
#     :return: optimal predictor parameters
#     """
#     #Initializing variables
#     parameters=np.zeros(predictors.shape[1])
#     checker=[1]*predictors.shape[1]
#     temp_parameters=[1]*predictors.shape[1]
#     counter=0
#     while checker!=[0]*predictors.shape[1]:#Stop when convergence occurs
#         #Using vector and matrix methods
#         predictions=np.dot(predictors, parameters)
#         for i in range(predictors.shape[1]):
#             if checker[i]==1:
#                 derivative=np.dot(np.transpose(predictors[:,i]),(predictions.shape[0]*(predictions-independents)))
#                 temp_parameters[i]=parameters[i]-alpha*derivative
#             if abs(temp_parameters[i]-parameters[i])<epsilon:#check for convergence
#                 checker[i]=0
#            # print(temp_parameters)
#         parameters=np.array(temp_parameters)
#         counter+=1
#         print(counter)
#         #print(parameters)
#     return parameters


def normal_linear_regression(predictors, independents):
    """

    :param predictors: an array of predictors
    :param independents: an array of binary/categorical independent values
    :return: Optimal parameters arrived at through linear regression using the normal equation
    """

    transposed_predictors=np.transpose(predictors)
    parameters= np.dot(np.linalg.inv(np.dot(transposed_predictors,predictors)),np.dot(transposed_predictors, independents))

    return parameters


def get_unsupervised_predictors(datalist):
    """

    :param data: a list of the data
    :return: an array of predictor values for unsupervised learning
    """
    temp_predictors=[]
    for line in datalist:
        line=[x for x in line]
        #line=[line[:-1]]
        line=['1']+line #this enables vector and matrix methods
        line=[float(i) for i in line]
        temp_predictors.append(line)

    predictors=np.array( temp_predictors)

    return predictors


def neural_AND(predictors):
    """

    :param data: an array of predictor values
    :return: An array of truth values
    """
    weights=np.array([-30, 20, 20])
    predictions=1/(1+np.exp((-1)*(np.dot(predictors, weights))))
    return predictions

def neural_OR(predictors):
    """

    :param data: an array of predictor values
    :return: An array of truth values
    """
    weights=np.array([-10, 20, 20])
    predictions=1/(1+np.exp((-1)*(np.dot(predictors, weights))))
    return predictions


def neural_XOR(predictors):
    """

    :param data: an array of predictor values
    :return: An array of truth values
    """
    weights1=np.transpose(np.array([[-30, 20, 20],[30, -40, -40]]))
    weights2=np.array([30, -50, -50])
    hidden_predictions= 1/(1+np.exp((-1)*(np.dot(predictors, weights1))))
    #hidden_predictions=np.ones(hidden_predictions.shape[0])
    hidden_predictions=np.c_[np.ones(hidden_predictions.shape[0]), hidden_predictions]
    hidden_predictions=np.array(hidden_predictions)
    predictions=1/(1+np.exp((-1)*(np.dot(hidden_predictions, weights2))))
    return predictions

def neural_XNOR(predictors):
    """

    :param data: an array of predictor values
    :return: An array of truth values
    """
    weights1=np.transpose(np.array([[-30, 20, 20],[30, -40, -40]]))
    weights2=np.array([-30, 50, 50])
    hidden_predictions= 1/(1+np.exp((-1)*(np.dot(predictors, weights1))))
    #hidden_predictions=np.ones(hidden_predictions.shape[0])
    hidden_predictions=np.c_[np.ones(hidden_predictions.shape[0]), hidden_predictions]
    hidden_predictions=np.array(hidden_predictions)
    predictions=1/(1+np.exp((-1)*(np.dot(hidden_predictions, weights2))))
    return predictions

def neural_NOR(predictors):
    weights=np.array([10, -20, -20])
    predictions=1/(1+np.exp((-1)*(np.dot(predictors, weights))))
    return predictions

def neural_NAND(predictors):
    """

    :param data: an array of predictor values
    :return: An array of truth values
    """
    weights=np.array([30, -20, -20])
    predictions=1/(1+np.exp((-1)*(np.dot(predictors, weights))))
    return predictions

def neural_network(predictors, neural_function):
    """

    :param data: an array of predictor values
    :param neural_function: the neural network to be used
    :return: An array of truth values
    """
    return neural_function(predictors)


# def normal_equation(predictors, independents):
#     """
#
#     :param predictors: an array of predictors
#     :param independents: an array of independent values
#     :return: The optimal parameters solved for using the normal equation
#     """
#     transposed_predictors=np.transpose(predictors)
#     parameters= np.dot(np.linalg.inv(np.dot(transposed_predictors,predictors)),np.dot(transposed_predictors, independents))
#
#     return parameters


