from ML4Humanities.max_code import multivariate_linear_regression as mv

#The following code reads several toy data files if uncommented
data=mv.get_data('toy_data')
predictors=mv.get_predictors(data)
independents=mv.get_independents(data)
scaled_independents=mv.scaleind(independents)
scaled_predictors=mv.scale(predictors)

logistic_data=mv.get_data('logistic_toy_data')
logistic_predictors=mv.get_predictors(logistic_data)
logistic_predictors=mv.scale(logistic_predictors)
logistic_independents=mv.get_independents(logistic_data)

neural_data=mv.get_data('neural_network_toy')
unsupervised_predictors=mv.get_unsupervised_predictors(neural_data)

#The following code contains various functions to manipulate predictors (merging, squaring etc...). Enter desired parameters and apply BEFORE scaling.
#mv.multiply_predictors(predictors, 1,2)
#mv.add_predictors(predictors, 1,2)
#mv.square_predictor(predictors, 1)
#mv.cube_predictor(predictors, 1)
#mv.sqrt_predictor(predictors, 1)



#The following code runs multivariate linear regression using gradient descent on the toy data if uncommented.

print(mv.linear_regression(scaled_predictors, scaled_independents))

#The following code runs multivariate linear regression using the normal equation on the toy data if uncommented.

print(mv.normal_linear_regression(predictors, independents))

#The following code runs multivariate logistic regression using gradient descent on the logistic toy data if uncommented.

print(mv.logistic_regression(logistic_predictors, logistic_independents))


#The following code executes various logic gates implemented as neural networks on toy data if uncommented.

print(mv.neural_network(unsupervised_predictors,mv.neural_XNOR))





