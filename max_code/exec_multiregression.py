from ML4Humanities.max_code import multivariate_linear_regression as mv
import numpy as np

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

multiclass_data=mv.get_data('multiclass_toy_data')
multiclass_predictors=mv.get_predictors(multiclass_data)
#multiclass_predictors=(mv.scale(multiclass_predictors))
multiclass_independents=mv.get_independents(multiclass_data)
multiclass_independents=mv.multiclass_independents(multiclass_independents)


neural_data=mv.get_data('neural_network_toy')
unsupervised_predictors=mv.get_unsupervised_predictors(neural_data)

#The following code contains various functions to manipulate predictors (merging, squaring etc...). Enter desired parameters and apply BEFORE scaling.
#mv.multiply_predictors(predictors, 1,2)
#mv.add_predictors(predictors, 1,2)
#mv.square_predictor(predictors, 1)
#mv.cube_predictor(predictors, 1)
#mv.sqrt_predictor(predictors, 1)



#The following code runs multivariate linear regression using gradient descent on the toy data if uncommented.

print(mv.linear_regression(scaled_predictors, scaled_independents, 0, predictors.shape[0]))

#The following code runs multivariate linear regression using the normal equation on the toy data if uncommented.

print(mv.normal_linear_regression(predictors, independents, 0))

#The following code runs multivariate logistic regression using gradient descent on the logistic toy data if uncommented.

result1=mv.logistic_regression(logistic_predictors, logistic_independents, 0.01, logistic_predictors.shape[0])

#The following code runs logistic regressioni using an advanced optimization algorithm

result2=mv.optimized_logistic_regression(logistic_predictors, logistic_independents, 1)

#The following is to compare the results of gradient descent and the advanced algorithm
print(result1)
print(result2)
print(mv.logistic_predictions(logistic_predictors,result1))
print(mv.logistic_predictions(logistic_predictors,np.array( [ 0.12378762,  1.50871727])))
print(mv.logistic_cost2(result1, logistic_predictors, logistic_independents))
print(mv.logistic_cost2(np.array( [ 0.12378762,  1.50871727]), logistic_predictors, logistic_independents))
#print(mv.logistic_cost(mv.logistic_predictions(logistic_predictors,result1), logistic_independents))


#The following code runs multiclass logistic regression using gradient descent
#multiresult=mv.multiclass_logistic_regression(multiclass_predictors,multiclass_independents)
#print(multiresult)
#print(mv.multiclass_predictions(multiclass_predictors, multiresult))
#print(mv.multiclass_predict(2, multiresult))


#The following code executes various logic gates implemented as neural networks on toy data if uncommented.

print(mv.neural_network(unsupervised_predictors,mv.neural_XNOR))




