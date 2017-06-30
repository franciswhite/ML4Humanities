#Indexing "conceptual scheme" by its word-count position
import numpy as np
import string

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

def word_indexer(filenames):
    '''Takes file of text and returns word-positions where "conceptual scheme" occurs.
    :param data_path: List of filenames, chronologically ordered.
    :return time_points: Array with positions.'''
    word_count = 0  #init word counter
    occurrence_tracker = [] #init empty list to note word count where bigram occurs
    first_part = "nope" #init first element of bigram
    count = 0
    count2 = 0
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                temp0 = line.translate(None, string.punctuation)
                temp1 = temp0.split(" ")
                for word in temp1:
                    word_count += 1
                    if word.lower() == "conceptual":
                        first_part = "conceptual"
                    else:
                        pass
                    if ((word.lower() == "scheme") or (word.lower() == "schemata") or (word.lower() == "schemes")) and (first_part == "conceptual"):
                        occurrence_tracker.append(word_count)
                        first_part =word #setting first word of bigram to not-"conceptual" again
                    else:
                        first_part = word

    time_points = np.array(occurrence_tracker)[np.newaxis].T
    return time_points, len(time_points)

#word_indexer(filenames)
print(word_indexer(filenames))