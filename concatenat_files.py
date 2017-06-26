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



with open('/home//sh/Desktop/june_project/data_quine/all_texts/text_all', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)