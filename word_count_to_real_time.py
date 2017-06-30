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
import numpy as np



# word_count_at_start = []
# number_files = 0
# number_words = 0
# with open('/home//sh/Desktop/june_project/data_quine/all_texts/text_allreasdfsdf.txt', 'w') as outfile:
#     for fname in filenames:
#         number_files += 1
#         with open(fname) as infile:
#             word_count_at_start.append(number_words)
#             word_count_at_start.append(fname)
#
#             for line in infile:
#                 line = line.split()
#                 for words in line:
#                     number_words += 1

def mapping(word_count):
    '''Maps word count to real time.'''
    if 0 < word_count < 11642:
        return 1936
    if 11642 <= word_count < 24123:
        return 1937
    if 24123 <= word_count < 25936:
        return 1938
    if 25936 <= word_count < 43371:
        return 1939
    if 43371 <= word_count < 135084:
        return 1940
    if 135084 <= word_count < 189959:
        return 1941
    if 189959 <= word_count < 191010:
        return 1942
    if 191010 <= word_count < 196714:
        return 1943
    if 196714 <= word_count < 203009:
        return 1945
    if 203009 <= word_count < 207659:
        return 1946
    if 207659 <= word_count < 212786:
        return 1947
    if 212786 <= word_count < 219440:
        return 1948
    if 219440 <= word_count < 221768:
        return 1949
    if 221768 <= word_count < 307308:
        return 1950
    if 307308 <= word_count < 333472:
        return 1951
    if 333472 <= word_count < 345728:
        return 1952
    if 345728 <= word_count < 394939:
        return 1953
    if 394939 <= word_count < 398942:
        return 1954
    if 398942 <= word_count < 409712:
        return 1955
    if 409712 <= word_count < 419290:
        return 1956
    if 419290 <= word_count < 426417:
        return 1957
    if 426417 <= word_count < 434039:
        return 1958
    if 434039 <= word_count < 437241:
        return 1959
    if 437241 <= word_count < 548677:
        return 1960


def real_time(list):
    '''Takes list/array of word counts and turns it into list of real time.'''
    real_time = []
    for i in range(0,len(list)):
        temp0 = mapping(list[i])
        real_time.append(temp0)
    return real_time



