import os

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
            with open(os.path.join(dirpath, file), errors="ignore") as f:#The encoding takes care of non-ASCII symbols
                dataset=dataset+f.read()+paragraph
    output=open("c:\\users\\maxgr\\Downloads\\Machine Learning for the Humanities\\bolzanostring.txt", "w")#put here the path of the output .txt-file
    output.write(dataset)
    output.close()
    return dataset

def open_datastring(path):
    """

    :param path:
    :return:
    """
    with open(path) as f:
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
datalist=datalist(datastring)
print(datalist[1])
print(len(datalist))
