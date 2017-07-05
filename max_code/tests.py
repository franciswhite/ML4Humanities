import numpy as np
from ML4Humanities.max_code import multivariate_linear_regression as mv
predictors=mv.get_predictors(result)
predictors=mv.scale(predictors)

predict2=mv.forward_propagation(np.array(predictors[0]), np.load("parameters3.npy"))
