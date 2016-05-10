import csv
import numpy as np
from gensim import corpora, models

class YoutubeCorpus(corpora.TextCorpus):
    stoplist = set('for a of the and to in i it you this that is was'.split()) # https://radimrehurek.com/gensim/tut1.html

    def get_texts(self):
        for line in csv.reader(open(self.input), delimiter=";"):
            yield [x for x in line[4].lower().split() if x not in self.stoplist]


def tokenise(s):
    stoplist = set('for a of the and to in i it you this that is was'.split())
    return [x for x in s.lower().split() if x not in stoplist]

file_path = "data/comment_10000.csv"

corpus = YoutubeCorpus(file_path)

lsi = models.LsiModel(corpus, num_topics=10)  # https://radimrehurek.com/gensim/models/lsimodel.html

new_doc = "who is te poor boy???plzz im just asking..."

for topic_id in range(0, 10):
    print("---- TOPIC %d ----" % topic_id)
    topic_components = lsi.show_topic(topic_id, topn=10)
    for (word, weight) in topic_components:
        print("%s => %.3f" % (corpus.dictionary[int(word)], weight))

print(lsi[corpus.dictionary.doc2bow(tokenise("who is te poor boy???plzz im just asking..."))])





