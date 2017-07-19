from ML4Humanities.max_code import machine_learning_methods as mv
import numpy as np

result=np.load("result.npy")#Enter here a path to a trained architecture
#predictor should be result of dataparsing, i.e. the ith element from an array of predictors. Provide a path and an i!
predictor=np.load("predictors")[i]
prediction=mv.forward_propagation(predictor, result)
np.save("trained_architecture", result)
print(prediction)
