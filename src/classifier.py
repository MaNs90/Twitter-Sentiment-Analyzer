from src.preprocessor import Postag, Lexicon, WordEmbeddings
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.ensemble import VotingClassifier
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
            os.path.join(PATH, "data", "sentiment", "unigrams-pmilexicon.txt"),
            os.path.join(PATH, "data", "sentiment", "bigrams-pmilexicon.txt")
        )
        self.feature.getvecs()

        print("Embedding Test Data...")
        self.test = Postag(
            os.path.join(PATH, "data-clean", self.testPath),
            os.path.join(PATH, "data", "glove", "glove.twitter.27B.100d.txt"),
            100,
            os.path.join(PATH, "data", "sentiment", "unigrams-pmilexicon.txt"),
            os.path.join(PATH, "data", "sentiment", "bigrams-pmilexicon.txt")

        )
        self.test.getvecs()

    def processNN(self):
        self.feature = WordEmbeddings(
            os.path.join(PATH, "data-clean", "self.trainingPath"),
            os.path.join(PATH, "data", "glove", "glove.twitter.27B.100d.txt"),
            100
        )
        self.feature.glove(flag=True)

        #self.test = Object
        self.test.run()

    def svm(self):
        # simply applies an SVM classifier to the feature vector
        # prints F1 score

        support = svm.SVC(kernel="linear", C=100)
        support = support.fit(self.feature.vector, self.feature.labels())
        result = support.predict(self.test.vector)
        print("My F1 score is ", f1_score(self.test.labels(), result, labels=[2, 0], average="macro"))

    def nn(self):
        return
        #fit
        #predict
        #result

    #def hybrid(self, predicts, weights):
    #    eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3], weights=[1, 1, 1])



