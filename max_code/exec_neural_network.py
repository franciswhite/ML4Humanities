from ML4Humanities.max_code import multivariate_linear_regression as mv
import numpy as np

data=mv.get_data('neural_network_toy')
predictors=mv.get_predictors(data)
independents=mv.get_independents(data)
independents=mv.multiclass_independents(independents)

#print(predictors.shape[0])
architecture=mv.neural_architecture(2, [predictors.shape[1]-1,3, independents.shape[1]])
#architecture=mv.neural_architecture(2, [predictors.shape[1]-1,5, 1])
#print(independents)
#print(architecture)

result=mv.train_neural_network(predictors, architecture, independents)
#print("NN_result",result)
#print("NNprediction",mv.forward_propagation(np.array([1,1,1]), result))
#print(independents)

#multiresult=mv.multiclass_logistic_regression(predictors,independents)
print("independents",independents)
print("predictors", predictors)
#print("Multiresult", multiresult)
#print("Multipredictions",mv.multiclass_predictions(predictors, multiresult))
#print("multipredict 2", mv.multiclass_predict(np.array([1,1,1]), multiresult))
print("NN_result",result)
print("NNprediction",mv.forward_propagation(np.array([1,1,1]), result))
