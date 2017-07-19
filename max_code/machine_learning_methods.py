import numpy as np
import scipy.optimize as sci
import matplotlib.pyplot as matp
import copy
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
            temp_predictor_means=temp_predictor_means+predictor_mean
    predictor_means=np.array([temp_predictor_means]*predictors.shape[0])
    predictors=predictors-predictor_means
    #next we divide by maxmin difference to scale predictors
    temp_scaled_predictors=[]
    for i in range (predictors.shape[1]):
        if i==0:
            maxmin_difference=1
        if i>0:
            maxmin_difference=np.max(predictors[:,i])-np.min(predictors[:,i])
        #adding a very small constant to avoid dividing by zero
        scaled_predictor=[x/(maxmin_difference+0.000000001) for x in predictors[:,i]]
        temp_scaled_predictors=temp_scaled_predictors+[scaled_predictor]
    predictors=np.transpose(np.array(temp_scaled_predictors))
    return predictors

def scaleind(independents):
    """
    :param predictors: an array of independent values
    :return: a scaled array of independent values
    """
    independents_mean=np.sum(independents)/independents.shape[0]
    independents_means=np.array([independents_mean]*independents.shape[0])
    independents=independents-independents_means
    #Scaling
    maxmin_difference=np.max(independents)-np.min(independents)
    scaled_independents=[x/maxmin_difference for x in independents]
    independents=np.array(scaled_independents)
    return(independents)

def multiclass_independents(independents):
    initialize=0
    for i in range (int(np.max(independents)+1)):
        independents_column=[1 if x==i else 0 for x in independents]
        if initialize==0:
            temp_independents=independents_column
            initialize=1
        else:
            temp_independents=np.c_[temp_independents, independents_column]

    print(temp_independents)
    return temp_independents


def multiply_predictors(predictors, i, j):
    """

    :param predictors: an array of predictors
    :param i: index of first predictor to be merged
    :param j: index of second predictor to be merged
    :return: array of predictors with i,j merged
    """
    multiply=predictors[:,i]*predictors[:,j]

    temp_predictors=[x for x in np.delete(predictors, [i,j], axis=1)]
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
    predictors=np.c_[np.array(predictors),square]
    return predictors

def cube_predictor(predictors, i):
    """

    :param predictors: an array of predictors
    :param i: index of first predictor to be manipulated
    :return: array of predictors with manipulated i appended to the end
    """
    cube=predictors[:,i]*predictors[:,i]*predictors[:,i]
    predictors=np.c_[np.array(predictors),cube]
    return predictors

def sqrt_predictor(predictors,i):
    """

    :param predictors: an array of predictors
    :param i: index of first predictor to be manipulated
    :return: array of predictors with manipulated i appended to the end
    """
    sqrt=np.sqrt(predictors[:,i])
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

def linear_cost(linear_predictions, independents, reg=0, parameters=np.zeros(1)):
    """

    :param linear_predictions: an array of predictions for linear regression
    :param independents: an array of independent variable values
    :param reg: a regularization value. If 0 no regularization is applied
    :param parameters: an array of parameters. Only needed if regularization is wished.
    :return: the cost value (scalar)
    """
    if reg==0:
        cost= 1/(2*linear_predictions.shape[0])*np.sum(np.multiply(1/2,np.square(linear_predictions-independents)))
    else:
        cost= 1/(2*linear_predictions.shape[0])*np.sum(np.multiply(1/2,np.square(linear_predictions-independents)))+reg*np.sum(np.square(parameters[1:]))
    return cost


def logistic_cost(logistic_predictions, independents, reg=0, parameters=np.zeros(1)):
    """

    :param logistic_predictions: an array of predictions for linear regression
    :param independents: an array of independent variable values
    :return: the cost value (scalar)
    """
    if reg==0:
        cost=1/logistic_predictions.shape[0]*(-np.dot(np.transpose(independents), np.log(logistic_predictions))-(np.dot(np.transpose(1-independents),np.log(1-logistic_predictions))))
    else:
        cost=1/logistic_predictions.shape[0]*(-np.dot(np.transpose(independents), np.log(logistic_predictions))-(np.dot(np.transpose(1-independents),np.log(1-logistic_predictions))))+reg/(2*logistic_predictions.shape[0])*np.sum(np.square(parameters[1:]))

    return cost

def plot_cost(cost, y, li, ax, fig):

    y[:-1] = y[1:]
    y[-1:] =cost
    li.set_ydata(y)
    ax.relim()
    ax.autoscale_view(True,True,False)
    ax.margins(y=1)
    fig.canvas.draw()
    matp.pause(0.01)
    pass


def logistic_cost2(parameters, predictors, independents, reg=0):
    """

    :param logistic_predictions: an array of predictions for linear regression
    :param independents: an array of independent variable values
    :return: the cost value (scalar)
    """
    logistic_predictions=1/(1+np.exp((-1)*(np.dot(predictors, parameters))))
    if reg==0:
        cost=1/logistic_predictions.shape[0]*(-np.dot(np.transpose(independents), np.log(logistic_predictions))-(np.dot(np.transpose(1-independents),np.log(1-logistic_predictions))))
    else:
        cost=1/logistic_predictions.shape[0]*(-np.dot(np.transpose(independents), np.log(logistic_predictions))-(np.dot(np.transpose(1-independents),np.log(1-logistic_predictions))))+reg/(2*logistic_predictions.shape[0])*np.sum(np.square(parameters[1:]))
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
    for i in range(predictors.shape[1]):
        if checker[i]==0:
            cost_derivative=0
            temp_cost_derivatives=temp_cost_derivatives+[cost_derivative]
        if checker[i]==1:
            cost_derivative= 1/predictions.shape[0]*np.dot(np.transpose(predictors[:,i]),(predictions.shape[0]*(predictions-independents)))
            temp_cost_derivatives=temp_cost_derivatives+[cost_derivative]
    cost_derivatives=np.array(temp_cost_derivatives)
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
            cost_derivative= 1/predictions.shape[0]*np.dot(np.transpose(predictors[:,i]),(predictions-independents))
            temp_cost_derivatives=temp_cost_derivatives+[cost_derivative]
    cost_derivatives=np.array(temp_cost_derivatives)
    return cost_derivatives

def gradient_descent(parameters, cost_derivatives, checker, alpha=3, reg=0, number_of_values=0):
    """

    :param parameters: An array containing parameter values
    :param cost_derivatives: An array containting cost derivative values
    :param checker: a list of convergence checkers
    :param alpha: a float representing the learning rate
    :param reg: the regularization constant. No regularization if 0
    :param number_of values: needed for regularization. Set to predictors.shape[0]
    :return: updated parameters after performing one step of gradient descent
    """
    temp_parameters=[i for i in parameters]
    if reg==0:
        for i in range(len(checker)):
            if checker[i]==1:
                temp_parameters[i]=parameters[i]-alpha*cost_derivatives[i]
        temp_parameters=np.array(temp_parameters)
    else:
        for i in range(len(checker)):
            if checker[i]==1:
                if i==0:
                    temp_parameters[i]=parameters[i]-alpha*cost_derivatives[i]
                else:
                    temp_parameters[i]=parameters[i]*(1-alpha*reg/number_of_values)-alpha*cost_derivatives[i]
        temp_parameters=np.array(temp_parameters)
    return temp_parameters

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

def linear_regression(predictors, independents, reg=0, number_of_values=0):
    """

    :param predictors: an array of predictors
    :param independents: an array of independent values
    :return: Optimal parameters arrived at through linear regression using gradient descent
    """
    counter=0
    checker=[1]*predictors.shape[1]
    parameters=[0]*predictors.shape[1]
    while checker!=[0]*predictors.shape[1]:
        temp_parameters=gradient_descent(parameters, linear_cost_derivative(predictors, linear_predictions(predictors, parameters), independents, checker), checker, 0.01, reg, number_of_values)
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

def logistic_regression(predictors, independents, reg=0, number_of_values=0, max_iter=100000, life_graph="n"):
    """

    :param predictors: an array of predictors
    :param independents: an array of binary/categorical independent values
    :return: Optimal parameters arrived at through logistic regression using gradient descent
    """
    counter=0
    checker=[1]*predictors.shape[1]
    parameters=[0]*predictors.shape[1]
    if life_graph=="y":
        fig = matp.figure()
        ax = fig.add_subplot(111)
        x = np.arange(10000)
        y = np.zeros(10000)
        li, = ax.plot(x, y)
        fig.canvas.draw()
        matp.show(block=False)

    while checker!=[0]*predictors.shape[1] and counter<=max_iter:
        predictions=logistic_predictions(predictors, parameters)
        logistic_cost_derivatives=logistic_cost_derivative(predictors, predictions, independents, checker)
        #print(logistic_cost_derivatives)
        temp_parameters=gradient_descent(parameters, logistic_cost_derivatives, checker, 0.01, reg, number_of_values)
        convergence_checker(checker, temp_parameters, parameters, epsilon=0.00001)
        parameters=temp_parameters
        counter+=1
        if life_graph=="y":
            plot_cost(logistic_cost(predictions, independents, reg, parameters), y, li, ax, fig)
        #print(checker)
        print(counter)
        #print(parameters)
    #print(parameters)
    #print(predictors)
    return parameters

def multiclass_logistic_regression(predictors, independents_array, reg=0, max_iter=100000, life_graph="n"):
    """

    :param predictors: an array of predictors
    :param independents_array: an array of multiclass independents
    :param reg: a regularization constant
    :param max_iter: maximum number of steps for gradient descent
    :param life_graph: "y" for life plotting of gradient descent
    :return: trained parameters for multiclass prediction
    """
    multiparameters=[]
    for i in range(independents_array.shape[1]):
        print(independents_array[:,i])
        independents=independents_array[:,i]
        counter=0
        checker=[1]*predictors.shape[1]
        parameters=[0]*predictors.shape[1]
        if life_graph=="y":
            fig = matp.figure()
            ax = fig.add_subplot(111)
            x = np.arange(10000)
            y = np.zeros(10000)
            li, = ax.plot(x, y)
            fig.canvas.draw()
            matp.show(block=False)

        while checker!=[0]*predictors.shape[1] and counter<=max_iter:
            predictions=logistic_predictions(predictors, parameters)
            logistic_cost_derivatives=logistic_cost_derivative(predictors, predictions, independents, checker)
            #print(logistic_cost_derivatives)
            temp_parameters=gradient_descent(parameters, logistic_cost_derivatives, checker, alpha=0.01)
            convergence_checker(checker, temp_parameters, parameters, epsilon=0.000000001)
            parameters=temp_parameters
            counter+=1
            if life_graph=="y":
                plot_cost(logistic_cost(predictions, independents, reg, parameters), y, li, ax, fig)
            #print(checker)
            print(counter)
            #print(parameters)
        #print(parameters)
        #print(predictors)
        multiparameters=multiparameters+[parameters]
    return np.array(multiparameters)

def multiclass_predictions(predictors, parameters_array):
    """

    :param predictors: an array of multiclass predictors
    :param parameters_array: an array of parameters trained by logistic regression
    :return: predictions for a dataset (or the training set)
    """

    initialize=0
    print(parameters_array.shape[0])
    for i in range(parameters_array.shape[0]):
        predictions=1/(1+np.exp((-1)*(np.dot(predictors, parameters_array[i]))))
        print(predictions)
        if initialize==0:
            predictions_array=predictions
            initialize=1
        else:
            predictions_array=np.c_[predictions_array, predictions]
    return predictions_array

def multiclass_predict(value, parameters_array):
    """

    :param value: a datapoint
    :param parameters_array: trained parameters from logistic regression
    :return: an array containing the predicted class as well as the confidence in the prediction
    """
    initialize=0
    value_array=np.array([1, value])
    for i in range(parameters_array.shape[0]):
        prediction=1/(1+np.exp((-1)*(np.dot(value_array, parameters_array[i]))))
        if initialize==0:
            prediction_array=prediction
            initialize=1
        else:
            prediction_array=np.c_[prediction_array, prediction]
    prediction_index=[np.argmax(prediction_array),np.max((prediction_array))]
    return prediction_index




def optimized_logistic_regression(predictors, independents, reg=0):
    parameters=np.array([0]*predictors.shape[1])
    #logistic_cost(logistic_predictions(predictors, parameters).__call__
    #print(type(logistic_cost(logistic_predictions(predictors, parameters), independents)))
    logistic_predictions1=logistic_predictions(predictors, parameters)
    final_parameters=sci.minimize(logistic_cost2, parameters, args=(predictors, independents, reg), method='Nelder-Mead')
    return final_parameters


def normal_linear_regression(predictors, independents, reg=0):
    """

    :param predictors: an array of predictors
    :param independents: an array of binary/categorical independent values
    :return: Optimal parameters arrived at through linear regression using the normal equation
    """
    if reg==0:
        transposed_predictors=np.transpose(predictors)
        parameters= np.dot(np.linalg.inv(np.dot(transposed_predictors,predictors)),np.dot(transposed_predictors, independents))
    else:
        transposed_predictors=np.transpose(predictors)
        reg_matrix=np.identity(transposed_predictors.shape[0])
        #reg_matrix=np.put(reg_matrix,[0], [0])
        reg_matrix.flat[0]=0
        parameters= np.dot(np.linalg.inv(np.dot(transposed_predictors,predictors)+reg*reg_matrix),np.dot(transposed_predictors, independents))
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

def neural_cost(architecture, predictions, independents, reg):
    """

    :param architecure: A list of parameter/weights matrices
    :param predictions: A list of arrays of predicted values
    :param untrained_neural_network: A function providing a neural network
    :param reg: 0 to turn regularization off. Desired value of regularization constant else.
    :return: a float: A cost value
    """
    neural_cost=0
    regularization=0

    for i in range(independents.shape[1]):
        temp_predictions=np.array([np.array(x) for x in predictions[:,-1]])
        neural_cost+=logistic_cost(temp_predictions[:,i], independents[:,i], 0)
    if reg!=0:
        for i in range(len(architecture)):
            regularization+= np.sum(reg/(2*independents.shape[0])*np.square(architecture[i][1:]))
    neural_cost=neural_cost+regularization
    return neural_cost

def neural_architecture(layers, units, epsilon=0.01):
    """

    :param layers: an integer: number of layers.
    :param units: a list of length "layers", specifying the number of units per layer (without bias unit). Input units should be predictors.shape[1]-1; output units should be independents_array.shape[1].
    :param epsilon: a parameter for random initialization.
    :return: a randomly initialized list of parameter matrices specifying a neural architecture.
    """
    architecture=[]
    for i in range(layers):
        layer=(np.random.uniform(0,1,(units[i+1],units[i]+1))*2*epsilon)-epsilon
        architecture=architecture+[layer]
    return architecture


def forward_propagation(predictor, architecture):
    """

    :param predictor: An array containing ONE predictor value
    :param architecture: a list of parameter matrices specifying a neural architecture
    :return: the neural network's hidden activations and predictions as a list of arrays
    """
    prediction=[[x for x in predictor]]

    for i in range(len(architecture)):
        predictor=1/(1+np.exp((-1)*(np.dot(architecture[i], predictor))))
        if i<len(architecture)-1:
            predictor=np.insert(predictor, 0, np.array([1]))
        prediction=prediction+[[x for x in predictor]]
    return prediction

def backward_propagation(prediction, architecture, independent):
    """

    :param prediction: an array: Prediction for ONE predictor value
    :param architecture: a list of parameter matrices specifying a neural architecture
    :param independent: an array: (Multiclass) Independent value correcponding to predictor value
    :return: The errors of the neural network
    """
    delta=prediction[-1]-independent
    deltas=[]
    deltas=deltas+[delta]
    for i in range(len(architecture)-1):

        if i==0:
            delta=np.dot(np.transpose(architecture[len(architecture)-i-1]),delta)*np.array(prediction[-(i+2)])*(1-np.array(prediction[-(i+2)]))
        else:
            delta=np.dot(np.transpose(architecture[len(architecture)-i-1]),delta[1:])*np.array(prediction[-(i+2)])*(1-np.array(prediction[-(i+2)]))
        deltas=[delta]+deltas
    return deltas

def neural_derivatives(prediction, deltas, architecture):
    """

    :param prediction: an array: Prediction for ONE predictor value
    :param deltas: a list of arrays of errors
    :param architecture: list of parameter matrices specifying a neural architecture
    :return: A list of matrices (arrays) of partial derivatives of errors for each parameter
    """
    derivatives=[]
    for i in range(len(architecture)):
        derivative=np.outer(deltas[i], np.transpose(prediction[i]))
        derivatives=derivatives+[derivative]
    return derivatives

def unroll(list_of_arrays):
    """

    :param list_of_arrays: a list of arrays such as a neural architecture
    :return: unrolled (or flattened) version of the list
    """
    unrolled_list=np.array([])
    for i in range(len(list_of_arrays)):
        unrolled_array=list_of_arrays[i].ravel()
        unrolled_list=np.concatenate((unrolled_list, unrolled_array))
    return unrolled_list

def rollin (unrolled_architecture, original_list_of_arrays):
    """

    :param unrolled_array: a unrolled(flattened) list of arrays
    :param original_list_of_arrays: the former unflattened list
    :return: a list of arrays
    """
    rolledin_list=[]
    for i in range(len(original_list_of_arrays)):
        if i==0:
            rolled_in_array=np.reshape(unrolled_architecture[:original_list_of_arrays[i].size],original_list_of_arrays[i].shape)
            rolledin_list=rolledin_list+[rolled_in_array]
        else:
            preceding_array_size=0
            for j in range(i):
                preceding_array_size+=rolledin_list[j].size

            rolled_in_array=np.reshape(unrolled_architecture[preceding_array_size:preceding_array_size+original_list_of_arrays[i].size],original_list_of_arrays[i].shape)
            rolledin_list=rolledin_list+[rolled_in_array]
    return rolledin_list




def gradient_check(architecture, predictors, independents, reg, epsilon=0.0001):
    """

    :param architecture: a list of parameter matrices specifying a neural architecture
    :param predictors: an array of predictor values
    :param independents: an array of (multiclass) independent values
    :param reg: a regularization constant to be passed to the cost function
    :param epsilon: a constant for gradient checking
    :return: an array of approximations of the gradient
    """
    grad_approx=[]
    #Gradient checking works better using unrolled parameters. For this purpose the list of parameter matrices is converted into one long vector, modified in one position and then "rolled in" again.
    unrolled_architecture=unroll(architecture)
    for i in range(unrolled_architecture.size):
        temp_architecture=[x for x in unrolled_architecture]
        temp2_architecture=copy.deepcopy(temp_architecture)
        temp_architecture[i]+=epsilon
        temp2_architecture[i]-=epsilon
        temp_architecture=rollin(temp_architecture, architecture)
        temp2_architecture=rollin(temp2_architecture, architecture)
    #predictions for gradient estimation are calculated.
        predictions1 = []
        for i in range(predictors.shape[0]):
            prediction = forward_propagation(predictors[i], temp_architecture)
            predictions1 = predictions1 + [prediction]
        predictions1=np.array(predictions1)
        predictions2=[]
        for i in range(predictors.shape[0]):
            prediction = forward_propagation(predictors[i], temp2_architecture)
            predictions2 = predictions2 + [prediction]
        predictions2=np.array(predictions2)
        grad_approx_i=(neural_cost(temp_architecture, predictions1, independents, reg)-neural_cost(temp2_architecture, predictions2, independents, reg))/(2*epsilon)
        grad_approx=grad_approx+[grad_approx_i]
    grad_approx=np.array(grad_approx)
    return grad_approx

def train_neural_network(predictors, architecture, independents_array, reg=0.01, max_iter=10000, alpha=0.1, life_graph="y", optimization_function=gradient_descent, gradient_checker=0):
    """

    :param predictors: An array of predictor values
    :param architecture: a list of parameter matrices specifying a randomly initialized neural architecture
    :param independents_array: a (multiclass) array of independent values
    :param reg: an integer: desired regularization constant
    :param max_iter: an integer: manual cutoff to be passed to the optimization function (gradient descent)
    :param alpha: a float: a learning rate for the optimization function (gradient descent)
    :param life_graph: "n" for no; "y" for life plotting optimization (gradient descent)
    :param optimization_algorithm=The optimization algorithm to be used (currently only works with gradient descent).
    :return: parameters for a trained neural network
    """
    #initialize a counter to keep track of iterations and set up life plotting the optimization.
    counter = 0
    if life_graph=="y":
        fig = matp.figure()
        ax = fig.add_subplot(111)
        x = np.arange(10000)
        y = np.zeros(10000)
        li, = ax.plot(x, y)
        fig.canvas.draw()
        matp.show(block=False)
    #The checker is not updated in this function and only passed on to gradient descent. The function could however easily be amended to use automated conversion checking.
    checker = [1] * len(unroll(architecture))
    while checker != [0] * predictors.shape[1] and counter <= max_iter:
        predictions = []
        #Computing the Deltas at every step of the optimization. Refer to Andrew Ng's course for details.
        for i in range(predictors.shape[0]):
            prediction = forward_propagation(predictors[i], architecture)
            predictions = predictions + [prediction]
            deltas = backward_propagation(prediction, architecture, independents_array[i])
            derivatives = neural_derivatives(prediction, deltas, architecture)
            #Overall Deltas are computed as sums of the derivatives.
            if i == 0:
                Delta = derivatives
            else:
                Delta = [np.add(x[0], x[1]) for x in zip(Delta, derivatives)]

        # The following code is to regularize the Deltas while taking care of bias units which are not regularized.

        Reg_Delta = [np.multiply(1 / predictors.shape[0], x) for x in Delta]
        Reg_Delta = [np.delete(x, 0, axis=0) for x in Reg_Delta[:-1]]+[Reg_Delta[-1]]
        Reg_Delta_0=[x[:,0] for x in Reg_Delta]
        Reg_Delta = [np.delete(x, 0, axis=1) for x in Reg_Delta]
        Reg_architecture = [np.multiply((reg / predictors.shape[0]), x) for x in architecture]
        Reg_architecture=[np.delete(x, 0, axis=1) for x in Reg_architecture]
        Reg_Delta = [np.add(x[0], x[1]) for x in zip(Reg_Delta, Reg_architecture)]
        Reg_Delta = [np.c_[x[0], x[1]] for x in zip(Reg_Delta_0, Reg_Delta)]
        predictions=np.array(predictions)
        # The gradient checker is a sanity check of the code computing the derivatives in a different way and comparing them to Reg_Delta. Apply without regularization/i.e. setting reg=0.
        if gradient_checker == 1:
            print("gradient_check", gradient_check(architecture, predictors, independents_array, reg))
            print("Reg_Delta", unroll(Reg_Delta))
        temp_architecture= unroll(architecture)
        temp_Reg_Delta= unroll(Reg_Delta)
        #the following commented code was used to achieve faster conversion in a specific case. In general it can be useful to use variable learning rates but an implementation using a clever loop to self-adjust the learning rate would be preferable (and not too hard to do).
        # if counter<=7500:
        #     alpha=3
        # if counter<=1000:
        #     alpha=10
        # if counter<=100:
        #     alpha=30
        # if counter>7500:
        #     alpha=1
        # print(alpha)
        #One step of optimization (gradient descent).
        temp_parameters = optimization_function(temp_architecture, temp_Reg_Delta, checker, alpha)
        temp_architecture = temp_parameters
        temp_architecture=rollin(temp_architecture, architecture)
        architecture = temp_architecture
        counter += 1
        #for the life plot:
        if life_graph=="y":
            plot_cost(neural_cost(architecture, predictions, independents_array, reg), y, li, ax, fig)
        print(counter)

    print("predictions", predictions)
    return architecture





def neural_network(predictors, neural_function):
    """

    :param data: an array of predictor values
    :param neural_function: the neural network to be used
    :return: An array of truth values
    """
    return neural_function(predictors)



