import os
import codecs
import itertools
import numpy as np
from ML4Humanities.max_code import multivariate_linear_regression as mv
import random

def get_contexts(path="c:\\users\\maxgr\\Downloads\\Machine Learning for the Humanities\\Bolzano_Dataset"):
    """

    :param path: a path leading to a dataset consisting of a directory with subdirectories containing text files.
    :return: One ginormous string concatenating all the individual text files (other files are ignored) separating them by "§" signs.
    """
    dataset=""
    print ('checking', path, os.path.isfile("c:\\users\\maxgr\\Downloads\\Machine Learning for the Humanities\\bolzanostring.txt"))
    for (dirpath, dirnames, filenames) in os.walk(path):
        base, tail = os.path.split(dirpath)
        if base != path: continue
        #filenames=(sorted(filenames))
        for file in filenames:
            print("path", os.path.join(dirpath, file))
            #paragraph=bytes(0xA7).decode("latin-1")
            paragraph="§"
            with open(os.path.join(dirpath, file), encoding="utf 8", errors="ignore") as f:#The encoding takes care of non-ASCII symbols
                dataset=dataset+f.read()+paragraph
    output=open("c:\\users\\maxgr\\Downloads\\Machine Learning for the Humanities\\bolzanostring.txt", "w", encoding="utf 8")#put here the path of the output .txt-file
    output.write(dataset)
    output.close()
    return dataset

def open_datastring(path):
    """

    :param path:
    :return:
    """
    with open(path, encoding="utf 8") as f:
        datastring=f.read()
    return datastring

def datalist(datastring):
    """
    param: datastring: the data as a string
    :return:
    """
    datalist=datastring.split("§")
    return datalist

#get_contexts()
datastring=open_datastring("c:\\users\\maxgr\\Downloads\\Machine Learning for the Humanities\\bolzanostring.txt")
datalist=datalist(datastring.lower())
datalist=[x for x in datalist if "unendlich" in x]
#print(datalist[1])
print(len(datalist))
countera=0
counterg=0
counterag=0
countere=0
counterc=0
arithmetic_list=[]
geometric_list=[]
euclid_list=[]
cantor_list=[]
#arithmetic_contexts=[ "ziffer", "arithmet", "zahl", "unbenannt", "zählbar", "konkret", "vielheit", "buchstabe", "reihe"]
arithmetic_contexts=[ "ziffer", "arithmet", "zahl", "unbenannt", "konkret", "vielheit", "reihe", "begriff","zeichen", "gegenst"]
geometric_contexts=["geomet", "gerade", "linie", "ebene", "körper", "raum", "entfernung", "entfernt", "punkt", "figur"]
euclidian_context=["teil","theil", "in sich faßt", "euklid", "ganz", "in sich schließ", "größer", "kleiner"]
cantorian_contexts=["paar", "zählbar", "weite", "bestimmungsgründe", "enstehungsweise", "gleich", "verbindung", "regel", "verglich"]
# arith_combs=set(itertools.combinations(arithmetic_contexts, 4))
# geo_combs=set(itertools.combinations(geometric_contexts, 4))
# eucl_combs=set(itertools.combinations(euclidian_context, 4))
# cantor_combs=set(itertools.combinations(cantorian_contexts, 4))
#
#
# datalist=[x for x in datalist if "unendlich" in x]
# print(len(datalist))
#
#
# for i in range(len(datalist)):
#     for x in arith_combs:
#         if len([y for y in x if y in datalist[i]])==4 and datalist[i] not in arithmetic_list:
#             arithmetic_list=arithmetic_list+[datalist[i]]
#             countera+=1
#             print("countera",countera)
#         #if datalist[i]:
#             #arithmetic_list=arithmetic_list+[datalist[i]]
#             #countera+=1
# for i in range(len(datalist)):
#     for x in geo_combs:
#         if len([y for y in x if y in datalist[i]])==4 and datalist[i] not in geometric_list:
#             geometric_list=geometric_list+[datalist[i]]
#             counterg+=1
#             print("counterg", counterg)
# for i in range(len(datalist)):
#     for x in eucl_combs:
#         if len([y for y in x if y in datalist[i]])==4 and datalist[i] not in euclid_list:
#             euclid_list=euclid_list+[datalist[i]]
#             countere+=1
#             print("countere", countere)
# for i in range(len(datalist)):
#     for x in cantor_combs:
#         if len([y for y in x if y in datalist[i]])==4 and datalist[i] not in cantor_list:
#             cantor_list=cantor_list+[datalist[i]]
#             counterc+=1
#             print("counterc", counterc)
#
# print("Arithmetic", countera)
# print("Geometric",counterg)
# print("Euclid",countere)
# print("Cantor",counterc)
# print("Arith Geo",len([x for x in arithmetic_list if x in geometric_list]))
# print("Euclid Cantor", len([x for x in euclid_list if x in cantor_list]))
# print("Arith Cantor",len([x for x in arithmetic_list if x in cantor_list]))
# print("Arith Euclid",len([x for x in arithmetic_list if x in euclid_list]))
# print("Geo Cantor",len([x for x in geometric_list if x in cantor_list]))
# print("Geo Euclid",len([x for x in geometric_list if x in euclid_list]))
#print(len([x for "Arithmetik" in datalist[i]]))

def prepare_data(datalist, features, independent_feature):
    """

    :param datalist: A list of strings
    :param features: A list of features to be counted
    :param independent_feature: A list chracterizing an independent feature
    :return: A list of string counts
    """
    feature_list=[]
    for i in range(len(datalist)):
        feature=[0]*(len(features)+1)
        for j in range(len(features)):
            #if features[j] in datalist[i]:
            feature[j]=datalist[i].count(features[j])
        print("checklist",[x for x in independent_feature if x in datalist[i]])
        if len([x for x in independent_feature if x in datalist[i]])>=4:
            feature[len(features)]=1
        else:
            feature[len(features)]=0
        print(feature)
        feature_list=feature_list+[feature]
    return feature_list


def get_testcross (feature_list):
    test_set=[]
    cross_set=[]
    number=int(len(feature_list))
    print(number)
    test=random.sample(range(len(feature_list)), int(np.rint(0.2*number)))
    print(test)
    print(feature_list[1399])
    #print(feature_list[1400])
    for i in test:
        test_set=test_set+[feature_list[i]]
    for i in sorted(test, reverse=True):
        del feature_list[i]
    print(len(feature_list))
    cross=random.sample(range(len(feature_list)), int(np.rint(0.2*number)))
    for i in cross:
        cross_set=cross_set+[feature_list[i]]
    for i in sorted(cross, reverse=True):
        del feature_list[i]
    print(len(feature_list))
    testcrossdata=[feature_list]+[test_set]+[cross_set]
    print("datalength", len(testcrossdata[0]), "testlength", len(testcrossdata[1]), "crosslength", len(testcrossdata[2]))
    return testcrossdata

# result=get_testcross(datalist)
# arithmetic_result=[prepare_data(result[0], arithmetic_contexts, cantorian_contexts)]+[prepare_data(result[1], arithmetic_contexts, cantorian_contexts)]+[prepare_data(result[2], arithmetic_contexts, cantorian_contexts)]
#
# geometric_result=[prepare_data(result[0], geometric_contexts, cantorian_contexts)]+[prepare_data(result[1], geometric_contexts, cantorian_contexts)]+[prepare_data(result[2], geometric_contexts, cantorian_contexts)]
#
# result=[prepare_data(result[0], arithmetic_contexts+geometric_contexts, cantorian_contexts)]+[prepare_data(result[1], arithmetic_contexts+geometric_contexts, cantorian_contexts)]+[prepare_data(result[2], arithmetic_contexts+geometric_contexts, cantorian_contexts)]
# print(len(result))

#print("datalength", len(result[0]), "testlength", len(result[1]), "crosslength", len(result[2]))
#result=[np.array(x) for x in result]
#arithmetic_result=[np.array(x) for x in arithmetic_result]
#geometric_result=[np.array(x) for x in geometric_result]

#np.save("result", result)
#np.save("arithmetic_result", arithmetic_result)
#np.save("geometric_result", geometric_result)
#print("array", result)
#print(prepare_data(datalist, arithmetic_contexts+geometric_contexts, cantorian_contexts))

result=np.load("result.npy")
arithmetic_result=np.load("arithmetic_result.npy")
geometric_result=np.load("geometric_result.npy")



predictors=mv.get_predictors(result[0])
predictors=mv.scale(predictors)
independents=mv.get_independents(result[0])
independents=mv.multiclass_independents(independents)

arithmetic_predictors=mv.get_predictors(arithmetic_result[0])
arithmetic_predictors=mv.scale(arithmetic_predictors)
arithmetic_independents=mv.get_independents(arithmetic_result[0])
arithmetic_independents=mv.multiclass_independents(arithmetic_independents)

geometric_predictors=mv.get_predictors(geometric_result[0])
geometric_predictors=mv.scale(geometric_predictors)
geometric_independents=mv.get_independents(geometric_result[0])
geometric_independents=mv.multiclass_independents(geometric_independents)

test_predictors=mv.get_predictors(result[1])
test_predictors=mv.scale(test_predictors)
test_independents=mv.get_independents(result[1])
test_independents=mv.multiclass_independents(test_independents)

arithmetic_test_predictors=mv.get_predictors(arithmetic_result[1])
arithmetic_test_predictors=mv.scale(arithmetic_test_predictors)
arithmetic_test_independents=mv.get_independents(arithmetic_result[1])
arithmetic_test_independents=mv.multiclass_independents(arithmetic_test_independents)

geometric_test_predictors=mv.get_predictors(geometric_result[1])
geometric_test_predictors=mv.scale(geometric_test_predictors)
geometric_test_independents=mv.get_independents(geometric_result[1])
geometric_test_independents=mv.multiclass_independents(geometric_test_independents)

cross_predictors=mv.get_predictors(result[2])
cross_predictors=mv.scale(cross_predictors)
cross_independents=mv.get_independents(result[2])
cross_independents=mv.multiclass_independents(cross_independents)

arithmetic_cross_predictors=mv.get_predictors(arithmetic_result[2])
arithmetic_cross_predictors=mv.scale(arithmetic_cross_predictors)
arithmetic_cross_independents=mv.get_independents(arithmetic_result[2])
arithmetic_cross_independents=mv.multiclass_independents(arithmetic_cross_independents)

geometric_cross_predictors=mv.get_predictors(geometric_result[2])
geometric_cross_predictors=mv.scale(geometric_cross_predictors)
geometric_cross_independents=mv.get_independents(geometric_result[2])
geometric_cross_independents=mv.multiclass_independents(geometric_cross_independents)


architecture=mv.neural_architecture(2, [predictors.shape[1]-1,40, independents.shape[1]])

#trained=mv.train_neural_network(predictors, architecture, independents)
# #with open("parameters", "w") as parameters:
#np.save("parameters840reg", trained)

# arithmetic_architecture=mv.neural_architecture(2, [arithmetic_predictors.shape[1]-1,40, independents.shape[1]])
#
# arithmetic_trained=mv.train_neural_network(arithmetic_predictors, arithmetic_architecture, arithmetic_independents)
# np.save("arithmetic_parameters840", arithmetic_trained)
#
# geometric_architecture=mv.neural_architecture(2, [geometric_predictors.shape[1]-1,40, independents.shape[1]])
#
# geometric_trained=mv.train_neural_network(geometric_predictors, geometric_architecture, geometric_independents)
# np.save("geometric_parameters840", geometric_trained)
architecture1=np.load("parameters4.npy")
architecture2=np.load("parameters10000.npy")
architecture3=np.load("parameters840.npy")
#predict1=mv.forward_propagation(np.array(predictors[-1]), np.load("parameters4.npy"))
#predict2=mv.forward_propagation(np.array(predictors[0]), np.load("parameters4.npy"))
#print("indepentent", independents[0])
#print("predict1", predict1[-1])
#print("predict2", predict2[-1])
#print(independents)
#print(np.load("parameters3.npy"))
#print(np.load("parameters2.npy"))

error1=0
error2=0
predictions=[]
for i in range(predictors.shape[0]):
    predict1=mv.forward_propagation(np.array(predictors[i]), architecture3)
    #predict2=mv.forward_propagation(np.array(predictors[i]), architecture2)
    error1+=np.sum(np.absolute(np.rint(np.array(predict1[-1]))-independents[i]))
    if np.sum(np.absolute(np.rint(np.array(predict1[-1]))-independents[i]))!=0:
        print("errorcheck",predict1[-1], independents[i])
    #error2+=np.sum(np.absolute(np.rint(np.array(predict2[-1]))-independents[i]))
    #print("Predict1",predict1[-1], independents[i])
    #print("Predict2", predict2[-1], independents[i])
    predictions=predictions+[predict1[-1]]
#print("cantorpredslist", [x for x in predictions if np.rint(x[1])==1])
print("cantorpreds", len([x for x in predictions if np.rint(x[1])==1]))
#print("noncantorpreds", len([x for x in predictions if np.rint(x[1])==0]))
print("error3", error1)
print("error10000", error2)

test_error1=0
test_error2=0
predictions=[]
for i in range(test_predictors.shape[0]):
    test_predict1=mv.forward_propagation(np.array(test_predictors[i]), architecture3)
    #test_predict2=mv.forward_propagation(np.array(test_predictors[i]), architecture2)
    test_error1+=np.sum(np.absolute(np.rint(np.array(test_predict1[-1]))-test_independents[i]))
    if np.sum(np.absolute(np.rint(np.array(test_predict1[-1]))-test_independents[i]))!=0:
        print("errorcheck",test_predict1[-1], test_independents[i])
    #test_error2+=np.sum(np.absolute(np.rint(np.array(test_predict2[-1]))-test_independents[i]))
    #print("testPredict1",test_predict1[-1], test_independents[i])
    #print("testPredict2", test_predict2[-1], test_independents[i])
    predictions=predictions+[test_predict1[-1]]
#print("cantorpredslist", [x for x in predictions if np.rint(x[1])==1])
print("cantorpreds", len([x for x in predictions if np.rint(x[1])==1]))
#print("noncantorpreds", len([x for x in predictions if np.rint(x[1])==0]))
print("testerror3", test_error1)
print("testerror10000", test_error2)

cross_error1=0
cross_error2=0

#print("cross_predictors", cross_predictors)
predictions=[]

for i in range(cross_predictors.shape[0]):
    #print("cross_pred", np.array(cross_predictors[i]))
    cross_predict1=mv.forward_propagation(np.array(cross_predictors[i]), architecture3)
    #cross_predict2=mv.forward_propagation(np.array(cross_predictors[i]), architecture2)
    cross_error1+=np.sum(np.absolute(np.rint(np.array(cross_predict1[-1]))-cross_independents[i]))
    if np.sum(np.absolute(np.rint(np.array(cross_predict1[-1]))-cross_independents[i]))!=0:
        print("errorcheck",cross_predict1[-1], cross_independents[i])
    #cross_error2+=np.sum(np.absolute(np.rint(np.array(cross_predict2[-1]))-cross_independents[i]))
    #print("crossPredict1",cross_predict1[-1], cross_independents[i])
    #print("crossPredict2", cross_predict2[-1], cross_independents[i])
    predictions=predictions+[cross_predict1[-1]]
#print("cantorpredslist", [x for x in predictions if np.rint(x[1])==1])
print("cantorpreds", len([x for x in predictions if np.rint(x[1])==1]))
#print("noncantorpreds", len([x for x in predictions if np.rint(x[1])==0]))
print("crosserror3", cross_error1)
#print("crosserror10000", cross_error2)



#print("Parameters 20000", np.load("parameters20000.npy"))
#print("Parameters 4", np.load("parameters4.npy"))


def olden_algorithm(architecture):
    architecture=[np.delete(x,0, axis=1) for x in architecture]
    connection_weights=np.dot(np.transpose(architecture[0]), np.transpose(architecture[1]))
    #print("connection_weights", connection_weights)
    overall_weight1=np.sum(connection_weights[:,0])
    overall_weight2=np.sum(connection_weights[:,1])
    #print("overall_weight", overall_weight1)
    #print("overall_weight2", overall_weight2)
    relative_importance1=connection_weights[:,0]/overall_weight1
    relative_importance2=connection_weights[:,1]/overall_weight2
    relative_importances=np.c_[relative_importance1, relative_importance2]
    #print("relative_importances", relative_importances)
    #relative_importances=connection_weights/overall_weight
    return connection_weights

relative_importances=olden_algorithm(architecture2)
relative_importances=np.c_[relative_importances, np.array(arithmetic_contexts+geometric_contexts)]
print(relative_importances)


relative_importances=olden_algorithm(architecture3)
relative_importances=np.c_[relative_importances, np.array(arithmetic_contexts+geometric_contexts)]
print(relative_importances)

print("number of cantor", len([x for x in independents if x[1]==1]))
print("number of cantor", len([x for x in test_independents if x[1]==1]))
print("number of cantor", len([x for x in cross_independents if x[1]==1]))


