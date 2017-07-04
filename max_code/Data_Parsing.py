import os
import codecs
import itertools
import numpy as np
from ML4Humanities.max_code import multivariate_linear_regression as mv
import pickle

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
arithmetic_contexts=[ "ziffer", "arithmet", "zahl", "unbenannt", "zählbar", "konkret", "vielheit", "buchstabe", "reihe"]
geometric_contexts=["geomet", "gerade", "linie", "ebene", "körper", "raum", "entfernung", "entfernt", "punkt"]
euclidian_context=["teil","theil", "in sich faßt", "euklid", "ganz"]
cantorian_contexts=["paar", "zählbar", "weite", "bestimmungsgründe", "enstehungsweise", "gleich", "verbindung", "regel"]
arith_combs=set(itertools.combinations(arithmetic_contexts, 3))
geo_combs=set(itertools.combinations(geometric_contexts, 3))
eucl_combs=set(itertools.combinations(euclidian_context, 3))
cantor_combs=set(itertools.combinations(cantorian_contexts, 3))


datalist=[x for x in datalist if "unendlich" in x]
print(len(datalist))


for i in range(len(datalist)):
    for x in arith_combs:
        if len([y for y in x if y in datalist[i]])==3 and datalist[i] not in arithmetic_list:
            arithmetic_list=arithmetic_list+[datalist[i]]
            countera+=1
            print("countera",countera)
        #if datalist[i]:
            #arithmetic_list=arithmetic_list+[datalist[i]]
            #countera+=1
for i in range(len(datalist)):
    for x in geo_combs:
        if len([y for y in x if y in datalist[i]])==3 and datalist[i] not in geometric_list:
            geometric_list=geometric_list+[datalist[i]]
            counterg+=1
            print("counterg", counterg)
for i in range(len(datalist)):
    for x in eucl_combs:
        if len([y for y in x if y in datalist[i]])==3 and datalist[i] not in euclid_list:
            euclid_list=euclid_list+[datalist[i]]
            countere+=1
            print("countere", countere)
for i in range(len(datalist)):
    for x in cantor_combs:
        if len([y for y in x if y in datalist[i]])==3 and datalist[i] not in cantor_list:
            cantor_list=cantor_list+[datalist[i]]
            counterc+=1
            print("counterc", counterc)

print("Arithmetic", countera)
print("Geometric",counterg)
print("Euclid",countere)
print("Cantor",counterc)
print("Arith Geo",len([x for x in arithmetic_list if x in geometric_list]))
print("Euclid Cantor", len([x for x in euclid_list if x in cantor_list]))
print("Arith Cantor",len([x for x in arithmetic_list if x in cantor_list]))
print("Arith Euclid",len([x for x in arithmetic_list if x in euclid_list]))
print("Geo Cantor",len([x for x in geometric_list if x in cantor_list]))
print("Geo Euclid",len([x for x in geometric_list if x in euclid_list]))
#print(len([x for "Arithmetik" in datalist[i]]))

def prepare_data(datalist, features, independent_feature):
    """

    :param datalist: A list of strings
    :param features: A list of features to be counted
    :param independent_feature: A list chracterizing an independent feature
    :return: A dictionary of word counts
    """
    feature_list=[]
    for i in range(len(datalist)):
        feature=[0]*(len(features)+1)
        for j in range(len(features)):
            #if features[j] in datalist[i]:
            feature[j]=datalist[i].count(features[j])
        print("checklist",[x for x in independent_feature if x in datalist[i]])
        if len([x for x in independent_feature if x in datalist[i]])>=3:
            feature[len(features)]=1
        else:
            feature[len(features)]=0
        print(feature)
        feature_list=feature_list+[feature]
    return feature_list

result=prepare_data(datalist, arithmetic_contexts+geometric_contexts, cantorian_contexts)
#result=np.array(result)
#print("array", result)
#print(prepare_data(datalist, arithmetic_contexts+geometric_contexts, cantorian_contexts))

predictors=mv.get_predictors(result)
independents=mv.get_independents(result)
independents=mv.multiclass_independents(independents)

architecture=mv.neural_architecture(2, [predictors.shape[1]-1,30, independents.shape[1]])

trained=mv.train_neural_network(predictors, architecture, independents)
#with open("parameters", "w") as parameters:
np.save("parameters", trained)

predict=mv.forward_propagation(np.array(predictors[300]), np.load("parameters.npy"))
print("predict", predict)
