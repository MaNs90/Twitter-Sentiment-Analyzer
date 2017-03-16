import src.cleaner as cleaner
import src.classifier as classifier


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
        print()
        print("###################### Processing Task A... ##############################")
        self.taskA.processLexicon()
        print("Training and Classifying using Support Vector Machines...")
        self.taskA.svm(10)
        print("Training and Classifying using Random Forest...")
        self.taskA.rforest(num_est=200)
        print("Processing RNN 2D feature set...")
        self.taskA.processNN()
        print("Training and Classifying RNN...")
        self.taskA.rnn(num_epochs=5)
        print("Hybrid Result...")
        self.taskA.hybrid(weights=[0.3,0.3,0.4])


        print()
        print("###################### Processing Task B... ##############################")
        self.taskB.processLexicon()
        print("Training and Classifying using Support Vector Machines...")
        self.taskB.svm(50)
        print("Training and Classifying using Random Forest...")
        self.taskB.rforest(num_est=200)
        print("Processing RNN 2D feature set...")
        self.taskB.processNN()
        print("Training and Classifying RNN...")
        self.taskB.rnn(num_epochs=7)
        print("Hybrid Result...")
        self.taskB.hybrid(weights=[0.3,0.3,0.4])


if __name__ == "__main__":
    pathA = "twitter-train-cleansed-A.tsv"
    pathB = "twitter-train-cleansed-B.tsv"
    testA = "twitter-dev-gold-A.tsv"
    testB = "twitter-dev-gold-B.tsv"

    runner = Runner(pathA, pathB, testA, testB)

    runner.run(clean=False)
