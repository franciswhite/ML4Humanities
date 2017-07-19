from ML4Humanities.max_code import machine_learning_methods as mv
import numpy as np

#The following code reads several data files if uncommented. Enter paths to data. For the neural network, data should have three columns.
data=mv.get_data('toy_data')
predictors=mv.get_predictors(data)
independents=mv.get_independents(data)
scaled_independents=mv.scaleind(independents)
scaled_predictors=mv.scale(predictors)

logistic_data=mv.get_data('logistic_toy_data')
logistic_predictors=mv.get_predictors(logistic_data)
logistic_predictors=mv.scale(logistic_predictors)
logistic_independents=mv.get_independents(logistic_data)

multiclass_data=mv.get_data('multiclass_toy_data')
multiclass_predictors=mv.get_predictors(multiclass_data)
#The commented line scales unscaled data.
#multiclass_predictors=(mv.scale(multiclass_predictors))
multiclass_independents=mv.get_independents(multiclass_data)
multiclass_independents=mv.multiclass_independents(multiclass_independents)


neural_data=mv.get_data('neural_network_toy')
unsupervised_predictors=mv.get_predictors(neural_data)

#The following code contains various functions to manipulate predictors (merging, squaring etc...). Enter desired parameters and apply BEFORE scaling.
#mv.multiply_predictors(predictors, 1,2)
#mv.add_predictors(predictors, 1,2)
#mv.square_predictor(predictors, 1)
#mv.cube_predictor(predictors, 1)
#mv.sqrt_predictor(predictors, 1)



#The following code runs multivariate linear regression using gradient descent on the data if uncommented.

print(mv.linear_regression(scaled_predictors, scaled_independents, 0, predictors.shape[0]))

#The following code runs multivariate linear regression using the normal equation on the data if uncommented.

print(mv.normal_linear_regression(predictors, independents, 0))

#The following code runs multivariate logistic regression using gradient descent on the logistic data if uncommented.

result1=mv.logistic_regression(logistic_predictors, logistic_independents, 0.01, logistic_predictors.shape[0])

#The following code runs logistic regressioni using an advanced optimization algorithm

result2=mv.optimized_logistic_regression(logistic_predictors, logistic_independents, 1)

#The following code runs multiclass logistic regression using gradient descent
multiresult=mv.multiclass_logistic_regression(multiclass_predictors,multiclass_independents)
print("Multiresult", multiresult)
#predictions for the toy data based on trained parameters
print("Multipredictions",mv.multiclass_predictions(multiclass_predictors, multiresult))
#Prediction for some value based on trained parameters. Gives the predicted class and the confidence in the prediction.
print("multipredict 9", mv.multiclass_predict(9, multiresult))

#The following code executes the XNOR logic gate implemented as neural networks on the data if uncommented.

print("XNOR predictions",mv.neural_network(unsupervised_predictors,mv.neural_XNOR))




