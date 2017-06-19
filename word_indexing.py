#indexing "conceptual scheme" by its word-count position
import numpy as np

collective_corpus = []
word_count = 0  #init word counter
occurrence_tracker = [] #init empty list to note word count where bigram occurs
first_part = "nope" #init first element of bigram
with open("toy_corpus.txt", 'r') as temp0:
    for line in temp0:
        collective_corpus.append(line.split())
#upper-case lower-case should not matter!!!
    for line in collective_corpus:
        for word in line:
            word_count += 1
            if word == "conceptual":
                print("here1")
                first_part = "conceptual"
            else:
                pass
            if (word == "scheme" or word == "schemata" or word == "schemes") and (first_part == "conceptual"):
                occurrence_tracker.append(word_count)
                first_part = "nope" #setting first word of bigram to not-"conceptual" again
            else:
                first_part = "nope"
                pass

time_points = np.array(occurrence_tracker)
print(time_points)

