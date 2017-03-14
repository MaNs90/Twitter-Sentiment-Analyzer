from src.preprocessor import Lexicon
from sklearn.metrics import f1_score
from sklearn import svm


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
        self.feature = Lexicon(
            "data-clean/" + self.trainingPath,
            "data/glove/glove.twitter.27B.100d.txt",
            100,
            "data/sentiment/unigrams-pmilexicon.txt"
        )
        self.feature.lexicon()

        print("Embedding Test Data...")
        self.test = Lexicon(
            "data-clean/" + self.testPath,
            "data/glove/glove.twitter.27B.100d.txt",
            100,
            "data/sentiment/unigrams-pmilexicon.txt"
        )
        self.test.lexicon()

    def svm(self):
        # simply applies an SVM classifier to the feature vector
        # prints F1 score

        support = svm.SVC(kernel="linear", C=100)
        support = support.fit(self.feature.vector, self.feature.labels())
        result = support.predict(self.test.vector)
        print("My F1 score is ", f1_score(self.test.labels(), result, labels=[2, 0], average="macro"))