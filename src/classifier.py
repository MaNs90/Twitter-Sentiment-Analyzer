from src.preprocessor import Postag, Lexicon, WordEmbeddings
from sklearn.metrics import f1_score
from sklearn import svm
from keras.layers import LSTM, Masking
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.models import model_from_json
import os

PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))


class Classifiers:
    # Contains methods for structuring data and applying classifiers

    def __init__(self, trainingPath, testPath):
        self.trainingPath = trainingPath
        self.testPath = testPath
        # feature object parses twitter data and puts it in feature vector form for a classifier
        self.feature = None
        # test object should always be in the same form
        self.test = None

    def processLexicon(self):
        # trainLexicon uses GloVe word embeddings and sentiment lexicon datasets to process/weight twitter data
        # the normalised feature vector can be used in classifiers such as SVM

        print("Embedding Training Data...")
        self.feature = Postag(
            os.path.join(PATH, "data-clean", self.trainingPath),
            os.path.join(PATH, "data", "glove", "glove.twitter.27B.100d.txt"),
            100,
            os.path.join(PATH, "data", "sentiment", "unigrams-pmilexicon.txt")
        )
        self.feature.getvecs()

        print("Embedding Test Data...")
        self.test = Postag(
            os.path.join(PATH, "data-clean", self.testPath),
            os.path.join(PATH, "data", "glove", "glove.twitter.27B.100d.txt"),
            100,
            os.path.join(PATH, "data", "sentiment", "unigrams-pmilexicon.txt")
        )
        self.test.getvecs()

    def processNN(self):
        print("Embedding RNN Training Data...")
        self.feature = WordEmbeddings(
            os.path.join(PATH, "data-clean", self.trainingPath),
            os.path.join(PATH, "data", "glove", "glove.twitter.27B.100d.txt"),
            100
        )

        print("Embedding RNN Test Data...")
        self.test = WordEmbeddings(
            os.path.join(PATH, "data-clean", self.testPath),
            os.path.join(PATH, "data", "glove", "glove.twitter.27B.100d.txt"),
            100
        )
        self.feature.glove(flag=True)
        self.test.glove(flag=True)

    def svm(self):
        # simply applies an SVM classifier to the feature vector
        # prints F1 score

        support = svm.SVC(kernel="linear", C=100)
        support = support.fit(self.feature.vector, self.feature.labels())
        result = support.predict(self.test.vector)
        print("My F1 score is ", f1_score(self.test.labels(), result, labels=[2, 0], average="macro"))

    def rnn(self):
        data = np.array(self.feature.vector)
        test = np.array(self.test.vector)
        labels = []
        val_dict = {2:[0,0,1], 1:[0,1,0], 0:[1,0,0]}
        for l in self.feature.labels():
            labels.append(val_dict[l])

        print("CREATING RNN")
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=(None, 100)))
        model.add(LSTM(100))
        model.add(Dense(3, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        model.fit(data, labels, nb_epoch=5, batch_size=64)

        predictions = model.predict(test, batch_size=32, verbose=0)
        result = []
        for j, r in enumerate(predictions):
            max_value = 0
            max_index = 1
            for i, val in enumerate(r):
                if val > max_value:
                    max_index = i
                    max_value = val
            result.append(max_index)

        result=np.array(result)

        print("My F1 score is ", f1_score(self.test.labels(), result, labels=[2, 0], average="macro"))





