import csv
import numpy as np
import sklearn.cluster
from gensim import corpora, models, matutils


class YoutubeCorpus(corpora.TextCorpus):
    def tokenise(self, s):
        stoplist = set('for a of the and to in i it you this that is was'.split())
        return [x for x in s.lower().split() if x not in stoplist]

    def get_texts(self):
        for line in csv.reader(open(self.input), delimiter=";"):
            yield self.tokenise(line[4])

class CommentClusterer:

    comments = dict()
    corpus = None
    model = None
    num_topics = None

    def __init__(self, file_path, num_topics=10):
        self.num_topics = num_topics
        self.read_comments_file(file_path)
        self.build_model()

    def read_comments_file(self, file_path):
        """Read the comments file and build necessary data structures."""
        self.corpus = YoutubeCorpus(file_path)

        with open(file_path) as f:
            reader = csv.reader(f, delimiter=";")

            for line in reader:
                if line[0] == "ID":     # ignore header
                    continue
                else:
                    video_id = line[1]
                    content = line[4]

                    if video_id not in self.comments:
                        self.comments[video_id] = [content]
                    else:
                        self.comments[video_id].append(content)

    def build_model(self):
        """Build the model that maps comments to a vector space"""
        self.model = models.LsiModel(self.corpus, num_topics=self.num_topics)

    def cluster_comments(self, video_id, num_clusters, report_filename="cluster_summary.txt"):
        """Return clusters of comments for given video ID."""
        comment_texts = list(set(self.comments[video_id]))
        comment_features = [self.model[self.corpus.dictionary.doc2bow(self.corpus.tokenise(x))] for x in comment_texts]
        comment_features_array = np.asarray([matutils.sparse2full(row, self.num_topics) for row in comment_features])
        print(comment_features_array.shape)

        clus = sklearn.cluster.KMeans(n_clusters=num_clusters)
        clus.fit(comment_features_array)
        labels = clus.predict(comment_features_array)
        distances = np.min(clus.transform(comment_features_array), axis=1)

        self.summarise_clusters(labels, comment_texts, distances, num_clusters, filename=report_filename)

    def summarise_clusters(self, assignments, texts, distances, num_clusters, filename):
        zipped = list(zip(assignments, texts, distances))

        with open(filename, 'w') as f:

            for cluster_index in range(num_clusters):
                comments_in_cluster = list(filter(lambda x: x[0] == cluster_index, zipped))
                comments_in_cluster.sort(key=lambda x: x[2])

                f.write("===== CLUSTER {} =====\n".format(cluster_index))
                f.write("======================\n")
                f.write("{}\n".format(comments_in_cluster[0][1]))
                f.write("======================\n")
                for c in comments_in_cluster[1:5]:
                    pass
                    f.write("{:.2f}\t{}\n".format(c[2], c[1]))
                f.write("\n")


cc = CommentClusterer("data/comment_100000.csv", num_topics=30)

cc.cluster_comments("tN3iNxr2bhk", num_clusters=10)

