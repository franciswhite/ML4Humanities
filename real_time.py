import numpy as np
import string
from word_indexing import word_indexer
from k_means import k_means
from word_count_to_real_time import real_time
import numpy as np
import string
from word_indexing import word_indexer

#Corpus to statistic
#Writes all filenames in list
import glob
filenames = glob.glob("/home/sh/Desktop/june_project/data_quine/all_texts/*.txt")

#Orders filenames according to date of mentioned in their name
import re
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

filenames.sort(key=natural_keys)
data_points = word_indexer(filenames)
#Corpus to statistic
data_points = word_indexer(filenames)
print(data_points)
real = real_time(data_points)
print(real)



#Plot word_count - real
import matplotlib.pyplot as plt

plt.plot(data_points, real, 'ro')
plt.axis([0, 500000, 1935, 1965])
plt.xlabel('Word Count Time')
plt.ylabel('Real Time')
plt.title('Plot: Word Count vs. Real Time')
plt.show()