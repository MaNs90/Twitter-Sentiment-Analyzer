import src.cleaner as cleaner
import src.classifier as classifier
import csv


class Runner:

    def __init__(self, pathA, pathB, testA, testB):
        """
        Runner class for the classification process.
        :param pathA: Path of training file for task A
        :param pathB: Path of training file for task B
        :param testA: Path of testing file for task A
        :param testB: Path of testing file for task B
        """
        self.cleaner = cleaner.Cleaner(pathA, pathB)
        self.cleanerTest = cleaner.Cleaner(testA, testB)
        self.taskA = classifier.Classifiers(pathA, testA)
        self.taskB = classifier.Classifiers(pathB, testB)

    def run(self, clean):
        """
        Begin the sentiment classification process.
        :param clean: Flag as to whether you want to clean the dataset first.
        """

        if clean:
            print("Cleaning...")
            self.cleaner.clean()
            self.cleanerTest.clean()
            ida = self.cleanerTest.tweetIDA
            idb = self.cleanerTest.tweetIDB
        print()
        print("###################### Processing Task A... ##############################")
        self.taskA.processLexicon()
        print("Training and Classifying using Support Vector Machines...")
        self.taskA.svm(10)
        print("Training and Classifying using Random Forest...")
        self.taskA.rforest(num_est=200)
        print("Processing RNN 2D feature set...")
        self.taskA.processNN(False)
        print("Training and Classifying RNN...")
        self.taskA.rnn(num_epochs=5)
        print("Hybrid Result...")
        resA = self.taskA.hybrid(weights=[4,3,3])


        print()
        print("###################### Processing Task B... ##############################")
        self.taskB.processLexicon()
        print("Training and Classifying using Support Vector Machines...")
        self.taskB.svm(50)
        print("Training and Classifying using Random Forest...")
        self.taskB.rforest(num_est=200)
        print("Processing RNN 2D feature set...")
        self.taskB.bayes()
        print("Processing RNN 2D feature set...")
        self.taskB.processNN(True)
        print("Training and Classifying RNN...")
        self.taskB.rnn(num_epochs=7)
        print("Hybrid Result...")
        resB = self.taskB.hybrid(weights=[3,2,2,6])

        enum = {2:'positive', 1:'neutral', 0:'negative', 3:'unknown'}

        with open("resultsA.csv", 'w') as outfile:
            spamwriter = csv.writer(outfile, delimiter=',')
            for i,pred in enumerate(resA):
                spamwriter.writerow([ida[i], enum[pred]])


        with open("resultsB.csv", 'w') as outfile:
            spamwriter = csv.writer(outfile, delimiter=',')
            for i, pred in enumerate(resB):
                spamwriter.writerow([idb[i], enum[pred]])


if __name__ == "__main__":
    pathA = "twitter-train-cleansed-A.tsv"
    pathB = "twitter-train-cleansed-B.tsv"
    testA = "twitter-test-A.tsv"
    testB = "twitter-test-B.tsv"

    runner = Runner(pathA, pathB, testA, testB)

    # CHANGE TO TRUE BEFORE SUBMISSION!
    runner.run(clean=True)
