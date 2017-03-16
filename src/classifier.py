from src.preprocessor import Postag, Lexicon, WordEmbeddings
from sklearn.metrics import f1_score
from sklearn import svm
from keras.layers import LSTM, Masking
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.metrics import classification_report

PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))


class Classifiers:
    def __init__(self, trainingPath, testPath):
        """
        Class that runs the classification process on training and testing file.
        :param trainingPath: Path to cleaned training file.
        :param testPath: Path to cleaned testing file.
        """
        self.trainingPath = trainingPath
        self.testPath = testPath
        self.feature = None
        self.test = None
        self.clfresults = []

    def processLexicon(self):
        """
        Construct feature vector containing word embeddings, sentiment lexicons and POs tags.
        """

        print("Embedding Training Data...")
        self.feature = Postag(
            os.path.join(PATH, "data-clean", self.trainingPath),
            os.path.join(PATH, "data", "glove", "glove.twitter.27B.100d.txt"),
            100,
            os.path.join(PATH, "data", "sentiment", "unigrams-pmilexicon.txt"),
            os.path.join(PATH, "data", "sentiment", "bigrams-pmilexicon.txt")
        )
        self.feature.getvecs() # Training features

        print("Embedding Test Data...")
        self.test = Postag(
            os.path.join(PATH, "data-clean", self.testPath),
            os.path.join(PATH, "data", "glove", "glove.twitter.27B.100d.txt"),
            100,
            os.path.join(PATH, "data", "sentiment", "unigrams-pmilexicon.txt"),
            os.path.join(PATH, "data", "sentiment", "bigrams-pmilexicon.txt")
        )
        self.test.getvecs() # Testing features

    def processNN(self):
        """
        Process the 2D features specific for the neural network.
        """
        print("Embedding RNN Training Data...")
        self.feature = Lexicon(
            os.path.join(PATH, "data-clean", self.trainingPath),
            os.path.join(PATH, "data", "glove", "glove.twitter.27B.100d.txt"),
            100,
            os.path.join(PATH, "data", "sentiment", "unigrams-pmilexicon.txt"),
            os.path.join(PATH, "data", "sentiment", "bigrams-pmilexicon.txt")
        )

        print("Embedding RNN Test Data...")
        self.test = Lexicon(
            os.path.join(PATH, "data-clean", self.testPath),
            os.path.join(PATH, "data", "glove", "glove.twitter.27B.100d.txt"),
            100,
            os.path.join(PATH, "data", "sentiment", "unigrams-pmilexicon.txt"),
            os.path.join(PATH, "data", "sentiment", "bigrams-pmilexicon.txt")
        )
        self.feature.lexicon(flag=True)
        self.test.lexicon(flag=True)

    def svm(self, param):
        """
        Applies a support vector machine classifier to the training and testing data.
        :param param: C paramater for the support vector machine.
        """

        support = svm.SVC(kernel="linear", C=param)
        support = support.fit(self.feature.vector, self.feature.labels())
        result = support.predict(self.test.vector)
        self.clfresults.append(result)
        print("Detailed classification report:")
        print()
        print("The model is trained on the full training set.")
        print("The scores are computed on the full development set.")
        print()
        y_true, y_pred = self.test.labels(), result
        print(classification_report(y_true, y_pred))
        print()
        print("The (F1_pos+F1_neg)/2 score is ", f1_score(y_true, y_pred, labels=[2, 0], average="macro"))

    def rforest(self, num_est):
        """
        Applies a random forest classifier to the training and testing data.
        :param num_est: Number of estimators paramater for the classifier.
        """
        forest = RandomForestClassifier(n_estimators=num_est)
        forest = forest.fit(self.feature.vector, self.feature.labels())
        result = forest.predict(self.test.vector)
        self.clfresults.append(result)
        print("Detailed classification report:")
        print()
        print("The model is trained on the full training set.")
        print("The scores are computed on the full development set.")
        print()
        y_true, y_pred = self.test.labels(), result
        print(classification_report(y_true, y_pred))
        print()
        print("The (F1_pos+F1_neg)/2 score is ", f1_score(y_true, y_pred, labels=[2, 0], average="macro"))

    def rnn(self, num_epochs):
        """
        Applies a reccurrent neural network to the training and testing data.
        :param num_epochs: Number of epochs paramater for the network.
        """
        data = np.array(self.feature.vector)
        test = np.array(self.test.vector)
        labels = []
        val_dict = {2:[0,0,1], 1:[0,1,0], 0:[1,0,0]}
        for l in self.feature.labels():
            labels.append(val_dict[l])

        print("CREATING RNN")
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=(75, 102)))
        model.add(LSTM(100))
        model.add(Dense(3, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        model.fit(data, labels, nb_epoch=num_epochs, batch_size=64)

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

        self.clfresults.append(result)
        print("Detailed classification report:")
        print()
        print("The model is trained on the full training set.")
        print("The scores are computed on the full development set.")
        print()
        y_true, y_pred = self.test.labels(), result
        print(classification_report(y_true, y_pred))
        print()
        print("The (F1_pos+F1_neg)/2 score is ", f1_score(y_true, y_pred, labels=[2, 0], average="macro"))

    def hybrid(self, weights):
        """
        A hybrid classifier function that combines the predicted outputs of the separate classifiers.
        Each classifier votes for an output giving and the highest weighted vote is used as the predicted class.
        :param weights: Weighting to give to each classifier results.
        """
        output = []
        for i, p in enumerate(self.clfresults[0]):
            votes = [0,0,0]
            votes[p] = weights[0]
            for j, altpr in enumerate(self.clfresults):
                if j!= 0:
                    votes[altpr[i]] += weights[j]
            max_value = 0
            max_index = 0
            for i, val in enumerate(votes):
                if val > max_value:
                    max_index = i
                    max_value = val
            output.append(max_index)

        print("Detailed classification report:")
        print()
        print("The model is trained on the full training set.")
        print("The scores are computed on the full development set.")
        print()
        y_true, y_pred = self.test.labels(), output
        print(classification_report(y_true, y_pred))
        print()
        print("The (F1_pos+F1_neg)/2 score is ", f1_score(self.test.labels(), output, labels=[2, 0], average="macro"))





