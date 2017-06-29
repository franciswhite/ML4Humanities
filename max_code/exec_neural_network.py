from ML4Humanities.max_code import multivariate_linear_regression as mv
import numpy as np

data=mv.get_data('neural_network_toy')
predictors=mv.get_predictors(data)
independents=mv.get_independents(data)
independents=mv.multiclass_independents(independents)

#print(predictors.shape[0])
architecture=mv.neural_architecture(1, [predictors.shape[1]-1,independents.shape[1]])
#print(independents)
#print(architecture)

result=mv.train_neural_network(predictors, architecture, independents, 0.001)
print(result)
print(mv.forward_propagation(np.array[1,1], result))
