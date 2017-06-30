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

def multiclass_independents(independents):
    initialize=0
    for i in range (int(np.max(independents)+1)):
        independents_column=[1 if x==i else 0 for x in independents]
        if initialize==0:
            temp_independents=independents_column
            initialize=1
        else:
            temp_independents=np.c_[temp_independents, independents_column]
        #print(temp_independents)
    #independents=np.array(temp_independents)
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
    #print(predictions)
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
    #cost=(-1)/logistic_predictions.shape[0]*(np.dot(independents, np.log(logistic_predictions)+(np.dot((1-independents),np.log(1-logistic_predictions)))))
    #cost=1/logistic_predictions.shape[0]*(np.dot(-(np.transpose(independents)), np.log(logistic_predictions)-((np.dot(np.transpose(1-independents),np.log(1-logistic_predictions))))))
    if reg==0:
        cost=1/logistic_predictions.shape[0]*(-np.dot(np.transpose(independents), np.log(logistic_predictions))-(np.dot(np.transpose(1-independents),np.log(1-logistic_predictions))))
    else:
        cost=1/logistic_predictions.shape[0]*(-np.dot(np.transpose(independents), np.log(logistic_predictions))-(np.dot(np.transpose(1-independents),np.log(1-logistic_predictions))))+reg/(2*logistic_predictions.shape[0])*np.sum(np.square(parameters[1:]))

    return cost

def plot_cost(cost, y, li, ax, fig):
    # fig = matp.figure()
    # ax = fig.add_subplot(111)
    # x = np.arange(10000)
    # y = np.zeros(10000)
    # li, = ax.plot(x, y)
    # fig.canvas.draw()
    # matp.show(block=False)

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
    #cost=(-1)/logistic_predictions.shape[0]*(np.dot(independents, np.log(logistic_predictions)+(np.dot((1-independents),np.log(1-logistic_predictions)))))
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
    #print(parameters)
    for i in range(predictors.shape[1]):
        if checker[i]==0:
            cost_derivative=0
            temp_cost_derivatives=temp_cost_derivatives+[cost_derivative]
        if checker[i]==1:
            cost_derivative= 1/predictions.shape[0]*np.dot(np.transpose(predictors[:,i]),(predictions.shape[0]*(predictions-independents)))
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
    #print(predictors)
    #print(predictions.shape[0])
    for i in range(predictors.shape[1]):
        if checker[i]==0:
            cost_derivative=0
            temp_cost_derivatives=temp_cost_derivatives+[cost_derivative]
        if checker[i]==1:
            cost_derivative= 1/predictions.shape[0]*np.dot(np.transpose(predictors[:,i]),(predictions-independents))
            #cost_derivative= (1/predictions.shape[0])*np.dot((predictions-independents),predictors[:,i])
            temp_cost_derivatives=temp_cost_derivatives+[cost_derivative]
    cost_derivatives=np.array(temp_cost_derivatives)
    return cost_derivatives

def gradient_descent(parameters, cost_derivatives, checker, alpha=0.01, reg=0, number_of_values=0):
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
                print("parameters", parameters[i])
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

def neural_cost(parameter_matrices, predictions, independents, reg):
    """

    :param parameter_matrices: A list of parameter/weights matrices
    :param predictions: A list of arrays of predicted values
    :param untrained_neural_network: A function providing a neural network
    :param reg: 0 to turn regularization off. Desired value of regularization constant else.
    :return: a float: A cost value
    """
    neural_cost=0
    regularization=0
    #predictions=untrained_neural_network(predictors, parameter_matrices)
    print("Predictionsarray", predictions)
    for i in range(predictions.shape[0]):
        print("Predictions", [x for x in predictions[:,-1][i]])
        print("Indepentents", independents[i])
        neural_cost+=logistic_cost(np.array([x for x in predictions[:,-1][i]]), independents[i], 0) #The extra list comprehenseion is actually necessary to avoid a very weird and stupid bug. Arrays behaving stupidly...
    if reg!=0:
        for i in range(len(parameter_matrices)):
            #reg_parameter_matrix=np.c_(parameter_matrices[i])
            regularization+= np.sum(reg/(2*independents.shape[0])*np.square(parameter_matrices[i][1:]))
    neural_cost=neural_cost+regularization
    print("cost",neural_cost)
    return neural_cost

def neural_architecture(layers, units, epsilon=0.001):
    """

    :param layers: an integer: number of layers.
    :param units: a list of length "layers", specifying the number of units per layer (without bias unit). Input units should be predictors.shape[1]-1; output units should be independents_array.shape[1].
    :param epsilon: a parameter for random initialization.
    :return: a randomly initialized list of parameter matrices specifying a neural architecture.
    """
    architecture=[]
    for i in range(layers):
        layer=(np.random.uniform(0,1,(units[i+1],units[i]+1))*2*epsilon)-epsilon
        #layer=(np.random.uniform(0,1,(units[i]+1,units[i+1]))*2*epsilon)-epsilon
        #print("layer",layer)
        architecture=architecture+[layer]
    return architecture


def forward_propagation(predictor, architecture):
    """

    :param predictor: An array containing ONE predictor value
    :param architecture: a list of parameter matrices specifying a neural architecture
    :return: the neural network's hidden activations and predictions as a list of arrays
    """
    #print("predictor", predictor)
    prediction=[[x for x in predictor]]
    print("start", prediction)
    #print("archinput",architecture)
    for i in range(len(architecture)):
        #print(np.dot(predictor, architecture[i]))
        predictor=1/(1+np.exp((-1)*(np.dot(architecture[i], predictor))))
        #print(predictor)
        #predictor=np.c_[np.ones(predictor.shape[0]), predictor]
        if i<len(architecture)-1:
            predictor=np.insert(predictor, 0, np.array([1]))
        #print(predictor)
        prediction=prediction+[[x for x in predictor]]
    #print("prediction0", prediction)
    #prediction=np.array(prediction)
    print("prediction1", prediction)
    return prediction

def backward_propagation(prediction, architecture, independent):
    """

    :param prediction: an array: Prediction for ONE predictor value
    :param architecture: a list of parameter matrices specifying a neural architecture
    :param independent: an array: (Multiclass) Independent value correcponding to predictor value
    :return: The errors of the neural network
    """
    #print(prediction)
    delta=prediction[-1]-independent
    deltas=[]
    deltas=deltas+[delta]
    for i in range(len(architecture)-2): #check the indexing later
        delta=np.dot(np.transpose(architecture[len(architecture)-i]),delta)*prediction[-(i+2)]*(1-prediction[-(i+2)])
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
    #print("deltas", deltas)
    #print("prediction",prediction)
    for i in range(len(architecture)):
        #print("prediction[i]", prediction[i])
        derivative=np.outer(deltas[i], np.transpose(prediction[i])) #double check this
        derivatives=derivatives+[derivative]
        #print(derivative)
    #print("derivatives", derivatives)
    return derivatives

def unroll(list_of_arrays):
    """

    :param list_of_arrays: a list of arrays such as a neural architecture
    :return: unrolled (or flattened) version of the list
    """
    unrolled_list=np.array([])
    for i in range(len(list_of_arrays)):
        unrolled_array=list_of_arrays[i].ravel()
        #print("unrolled_array", unrolled_array)
       #print("init unrolled lsit", unrolled_list)
        unrolled_list=np.concatenate((unrolled_list, unrolled_array))
    #print("unrolled_list", unrolled_list)
    return unrolled_list

def rollin (unrolled_array, original_list_of_arrays):
    """

    :param unrolled_array: a unrolled(flattened) list of arrays
    :param original_list_of_arrays: the former unflattened list
    :return: a list of arrays
    """
    rolledin_list=[]
    for i in range(len(original_list_of_arrays)):
        if i==0:
            rolled_in_array=np.reshape(unrolled_array[:original_list_of_arrays[i].size],original_list_of_arrays[i].shape)
            rolledin_list=rolledin_list+[rolled_in_array]
        else:
            rolled_in_array=np.reshape(unrolled_array[original_list_of_arrays[i-1].size:original_list_of_arrays[i-1].size+original_list_of_arrays[i].size],original_list_of_arrays[i].shape)
            rolledin_list=rolledin_list+[rolled_in_array]
    #print("rolled in list", rolledin_list)
    return rolledin_list




def gradient_check(architecture, predictions, independents, reg, epsilon=0.0001):
    """

    :param architecture: a list of parameter matrices specifying a neural architecture
    :param predictions: a list of arrays of prediction values
    :param independents: an array of (multiclass) independent values
    :param reg: a regularization constant to be passed to the cost function
    :param epsilon: a constant for gradient checking
    :return: an array of approximations of the gradient
    """
    grad_approx=[]



    unrolled_architecture=unroll(architecture)
    for i in range(unrolled_architecture.size):
        temp_architecture=[x for x in unrolled_architecture]
        temp2_architecture=copy.deepcopy(temp_architecture)
        temp_architecture[i]+=epsilon
        temp2_architecture[i]-=epsilon
        temp_architecture=rollin(temp_architecture, architecture)
        temp2_architecture=rollin(temp2_architecture, architecture)
        #print("archtiectures", temp_architecture, temp2_architecture)
        grad_approx_i=(neural_cost(temp_architecture, predictions, independents, reg)-neural_cost(temp2_architecture, predictions, independents, reg))/(2*epsilon)
        grad_approx=grad_approx+[grad_approx_i]
    #print("grad_approx", grad_approx)
    return grad_approx


    # for i in range(len(architecture)):
    #     for j in architecture[i]:
    #         temp_architecture=copy.deepcopy(architecture)
    #         temp2_architecture=copy.deepcopy(temp_architecture)
    #         print(j)
    #         temp_parameters1=j+epsilon
    #         temp_parameters2=j-epsilon
    #         print(j)
    #         temp_architecture[i][j]+=epsilon
    #         temp2_architecture[i][j]-=epsilon
    #         #unrolled=unroll([np.array([[1,2],[3,4]]), np.array([5,6])])
    #         #rollin(unrolled, [np.array([[1,2],[3,4]]), np.array([5,6])])
    #         grad_approx_i=(neural_cost(temp_architecture, predictions, independents, 0.1)-neural_cost(temp2_architecture, predictions, independents, 0.1))/(2*epsilon)
    #         print("grad_approx_i",grad_approx_i)
    #         grad_approx=grad_approx+[grad_approx_i] #Check here whether it has to be the other way round
    # print("grad_approx", grad_approx)
    # return grad_approx


def train_neural_network(predictors, architecture, independents_array, reg=0.0001, max_iter=2000, life_graph="n", optimization_function=gradient_descent, gradient_checker=1):
    """

    :param predictors: An array of predictor values
    :param architecture: a list of parameter matrices specifying a randomly initialized neural architecture
    :param independents_array: a (multiclass) array of independent values
    :param reg: an integer: desired regularization constant
    :param max_iter: an integer: manual cutoff to be passed to gradient descent
    :param life_graph: "n" for no; "y" for life plotting gradient descent
    :param optimization_algorithm=The optimization algorithm to be used.
    :return: a trained neural network
    """
    #for i in range(independents_array.shape[1]):
        #independents=independents_array[:,i]
    counter = 0
    checker = [1] * predictors.shape[1]
    while checker != [0] * predictors.shape[1] and counter <= max_iter:
        predictions = []
        for i in range(predictors.shape[0]):
            prediction = forward_propagation(predictors[i], architecture)
            #print("prediction", prediction)
            predictions = predictions + [prediction]
            deltas = backward_propagation(prediction, architecture, independents_array[i])
            derivatives = neural_derivatives(prediction, deltas, architecture)
            #print(derivatives)
            if i == 0:
                Delta = derivatives
                #print(Delta)
            else:
                Delta = [np.add(x[0], x[1]) for x in zip(Delta, derivatives)]
        # The following code is to regularize the Deltas
        #print("Delta", Delta)
        #print(predictions)
        Reg_Delta = [1 / predictors.shape[0] * x for x in Delta]
        #print("Reg DElt", Reg_Delta)
        Reg_Delta_0 = [x[:,0] for x in Reg_Delta]
        Reg_Delta = [np.delete(x, 0, axis=1) for x in Reg_Delta]
        Reg_architecture = [reg * x for x in architecture]
        #print("Reg architecture", Reg_architecture)
        Reg_architecture=[np.delete(x, 0, axis=1) for x in Reg_architecture]
        #print("Reg Delta deleted", Reg_Delta)
        #print(architecture)
        #print(Reg_architecture)
        #print("Reg Delta 0", Reg_Delta_0)
        Reg_Delta = [np.add(x[0], x[1]) for x in zip(Reg_Delta, Reg_architecture)]
        #print("Reg Delta prefinished", Reg_Delta)
        Reg_Delta = [np.c_[x[0], x[1]] for x in zip(Reg_Delta_0, Reg_Delta)]
        #print("Reg Delta finished", Reg_Delta)
        predictions=np.array(predictions)
        if gradient_checker == 1:
            print("gradient_check", gradient_check(architecture, predictions, independents_array, reg))
            print("Reg_Delta", unroll(Reg_Delta))
        print("prearchitecture", architecture)
        temp_architecture= unroll(architecture)
        temp_Reg_Delta= unroll(Reg_Delta)
        #for i in range(architecture.size):
        temp_parameters = optimization_function(temp_architecture, temp_Reg_Delta, checker)  # check whether flattening is needed
        temp_architecture = temp_parameters
        temp_architecture=rollin(temp_architecture, architecture)
        # convergence_checker(checker, temp_architecture, architecture, epsilon=0.0001)
        architecture = temp_architecture
        print("architecture", architecture)
        counter += 1
    return architecture





def neural_network(predictors, neural_function):
    """

    :param data: an array of predictor values
    :param neural_function: the neural network to be used
    :return: An array of truth values
    """
    return neural_function(predictors)



