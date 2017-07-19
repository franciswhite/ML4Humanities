from ML4Humanities.max_code import machine_learning_methods as mv
import numpy as np

data=mv.get_data('neural_network_toy') #Enter the path to the data here. Should be a .txt file with spaces&tabs as delimiters, one datapoint per line.
predictors=mv.get_predictors(data)
independents=mv.get_independents(data)
independents=mv.multiclass_independents(independents)

#Enter here the desired number of layers (first parameter) and units in the hidden layers ("middle" elements of the second parameters. E.g. for a neural network with two hidden layers with 5 units each (without bias unit): mv.neural_architecture(3, [predictors.shape[1]-1, 5,5, independents.shape[1]])
architecture=mv.neural_architecture(2, [predictors.shape[1]-1,5, independents.shape[1]])


result=mv.train_neural_network(predictors, architecture, independents)
#Enter here a path to which the result should be safed.
np.save("trained_architecture", result)
print("NN_result",result)
