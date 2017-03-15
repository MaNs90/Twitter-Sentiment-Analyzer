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
        
        print()    
        print("###################### Processing Task B... ##############################")
        self.taskB.processLexicon()
        print("Training and Classifying using Support Vector Machines...")
        self.taskB.svm(100)


if __name__ == "__main__":
    pathA = "twitter-train-cleansed-A.tsv"
    pathB = "twitter-train-cleansed-B.tsv"
    testA = "twitter-dev-gold-A.tsv"
    testB = "twitter-dev-gold-B.tsv"

    runner = Runner(pathA, pathB, testA, testB)

    runner.run(clean=True)
