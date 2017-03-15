import src.cleaner as cleaner
import src.classifier as classifier


class Runner:

    def __init__(self, pathA, pathB, testA, testB):
        self.cleaner = cleaner.Cleaner(pathA, pathB)
        self.cleanerTest = cleaner.Cleaner(testA, testB)
        self.taskA = classifier.Classifiers(pathA, testA)
        self.taskB = classifier.Classifiers(pathB, testB)

    def run(self, clean):

        if clean:
            print("Cleaning...")
            self.cleaner.clean()
            self.cleanerTest.clean()
        print()
        print("###################### Processing Task A... ##############################")
        self.taskA.processLexicon()
        print("Training and Classifying using Support Vector Machines...")
        self.taskA.svm(10)
        print("Training and Classifying using Random Forest...")
        self.taskA.rforest()
        print("Processing RNN 2D feature set...")
        self.taskA.processNN()
        print("Training and Classifying RNN...")
        self.taskA.rnn()
        print("Hybrid Result...")
        self.taskA.hybrid(weights=[0.4,0.3,0.3])


        print()
        print("###################### Processing Task B... ##############################")
        self.taskB.processLexicon()
        print("Training and Classifying using Support Vector Machines...")
        self.taskB.svm(100)
        print("Training and Classifying using Random Forest...")
        self.taskB.rforest()
        print("Processing RNN 2D feature set...")
        self.taskB.processNN()
        print("Training and Classifying RNN...")
        self.taskB.rnn()
        print("Hybrid Result...")
        self.taskB.hybrid(weights=[0.4,0.3,0.3])


if __name__ == "__main__":
    pathA = "twitter-train-cleansed-A.tsv"
    pathB = "twitter-train-cleansed-B.tsv"
    testA = "twitter-dev-gold-A.tsv"
    testB = "twitter-dev-gold-B.tsv"

    runner = Runner(pathA, pathB, testA, testB)

    runner.run(clean=False)
