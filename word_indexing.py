#indexing "conceptual scheme" by its word-count position
import numpy as np
import string
collective_corpus = []
word_count = 0  #init word counter
occurrence_tracker = [] #init empty list to note word count where bigram occurs
first_part = "nope" #init first element of bigram
with open("toy_corpus.txt", 'r') as temp0:
    for line in temp0:
        #collective_corpus.append(line.replace('.', " ", 1).replace('?'," ", 1).replace(',', " ", 1).replace(':', " ", 1).replace(';', " ", 1).replace('!', " ", 1).replace("'", " ", 1).split()) #removes points etc
        collective_corpus.append(line.translate(None, string.punctuation))
#upper-case lower-case should not matter!!!
    for line in collective_corpus:
        line = line.split(" ")
        for word in line:
            word_count += 1
            print(word, word_count)
            if word.lower() == "conceptual":
                first_part = "conceptual"
            else:
                pass
            if ((word.lower() == "scheme") or (word.lower() == "schemata") or (word.lower() == "schemes")) and (first_part.lower() == "conceptual"):
                occurrence_tracker.append(word_count)
                first_part = word #setting first word of bigram to not-"conceptual" again
            else:
                first_part = word
                pass

time_points = np.array(occurrence_tracker)
print(time_points)

###Weirdly does not replace all commas.
