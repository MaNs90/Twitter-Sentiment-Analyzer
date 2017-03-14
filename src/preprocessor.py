import csv
import numpy as np
from src.tokenizer import Tokenizer
from sklearn.preprocessing import normalize


class Feature:
    # Parent class for features

    def __init__(self, trainingPath):
        self.trainingPath = trainingPath
        self.data = []
        self.parse()
        self.myTweets = [tweet['Tweet'] for tweet in self.data]
        self.vector = None

    def labels(self):
        myLabels = np.empty((len(self.myTweets)),dtype=np.int16)
        index = 0
        for sentiment in self.data:
            myLabels[index] = sentiment["Sentiment"]
            index += 1
        return myLabels

    def parse(self):
        tk = Tokenizer(preserve_case=False)
        with open(self.trainingPath) as training:
            tsvRead = csv.reader(training, delimiter="\t")
            enum = {'positive': 2, 'neutral': 1, 'negative': 0}
            tweet_dict = {}
            for line in tsvRead:
                self.data.append({'Sentiment' : enum[line[0]], 'Tweet' : tk.tokenize(line[1])})


class WordEmbeddings(Feature):

    def __init__(self, trainingPath, glovePath, dim):
        Feature.__init__(self, trainingPath)
        self.glovePath = glovePath
        self.dim = dim
        self.vector = np.empty([len(self.myTweets), 1])

    def glove(self, flag=False):
        glove_embedding = {}

        all_words = set()
        for tweet in self.myTweets:
            all_words = all_words.union(set(w for w in tweet))
        glove_count = 0
        with open(self.glovePath, 'r', encoding='utf-8') as infile:
            for line in infile:
                parts = line.split()
                word = parts[0]
                nums = list(map(float, parts[1:]))
                if word in all_words:
                    glove_count += 1
                    glove_embedding[word] = np.array(nums)
        index = 0
        if flag:
            features = []
        else:
            features = np.empty([len(self.myTweets), self.dim])
        for words in self.myTweets:
            if flag:
                temp = np.empty([100, self.dim])
            else:
                temp = np.empty([len(words), self.dim])
            count = 0
            for w in words:
                if w in glove_embedding.keys():
                    temp[count] = glove_embedding[w]
                else:
                    temp[count] = np.zeros(self.dim)
                count += 1
            if flag:
                features.append(temp)
            else:
                features[index] = np.mean(temp, axis=0)
            index += 1

        self.vector = features
        return features


class Lexicon(WordEmbeddings):

    def __init__(self, trainingPath, glovePath, dim, sentimentPath):
        WordEmbeddings.__init__(self, trainingPath, glovePath, dim)
        self.sentimentPath = sentimentPath

    def lexicon(self):
        features = self.glove()
        all_words = set(w for words in self.myTweets for w in words)
        unigramLexicon = {}
        count = 0
        print("Finding lexical sentiments...")
        with open(self.sentimentPath, 'r', encoding='utf-8') as infile:
            for line in infile:
                sentence = line.split()
                word = sentence[0]
                num = sentence[1]
                if word in all_words:
                    count += 1
                    unigramLexicon[word] = num
        print("The function found {} in the unigram sentiment lexicon out of {} total unique words".format(count, len(
            all_words)))

        index = 0
        sentimentFeatures = np.empty([len(self.myTweets), 1])
        for words in self.myTweets:
            temp = np.empty([len(words), 1])
            count = 0
            for w in words:
                if w in unigramLexicon.keys():
                    # print("I found ", w)
                    # print("Its embedding is ", glove_small[w])
                    temp[count] = unigramLexicon[w]
                else:
                    # print("I didn't find ", w)
                    temp[count] = 0
                count += 1

            sentimentFeatures[index] = np.mean(temp, axis=0)
            index += 1

        sentimentFeatures = normalize(sentimentFeatures, axis=0)
        new_features = []
        for i, f in enumerate(features):
            new_features.append(np.append(f, sentimentFeatures[i]))

        print(sentimentFeatures)
        features = np.array(new_features)

        self.vector = features
        return features

