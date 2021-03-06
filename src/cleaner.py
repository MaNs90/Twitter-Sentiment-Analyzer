import csv
import re
import os
import pandas as pd


class Cleaner:
    def __init__(self, pathA, pathB):
        """
        Cleaner class to remove noisy data.
        :param pathA: Path to file for task A to clean.
        :param pathB: Path to file for task B to clean.
        """
        back1 = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))

        filePathRoot = os.path.join(back1, "data")

        self.pathA = os.path.join(filePathRoot, pathA)
        self.pathB = os.path.join(filePathRoot, pathB)

        cleanedPathRoot = os.path.join(back1, "data-clean")

        self.cleanedPathA = os.path.join(cleanedPathRoot, pathA)
        self.cleanedPathB = os.path.join(cleanedPathRoot, pathB)

        self._setUpEmoticonRegularExpressions()

        self.regexCorrections = {
            re.compile(r"http\S+"): "<url>",
            re.compile(r"/"): " / ",
            re.compile(r"[@][A-z0-9]+"): "<user>",
            re.compile(self.happyEmoticonRegex): "<smile>",
            re.compile(r"[8:=;]['`\-][0-9A-z]"): "<lolface>",
            re.compile(self.sadEmoticonRegex): "<sadface>",
            re.compile(self.neutralEmoticonRegex): "<neutralface>",
            re.compile(r"<3"): "<heart>",
            re.compile(r"[-+]?[0-9]+"): "<number>",
            # re.compile(r'\#?[A-Za-z\']*[\.\,]'): " ",
            # re.compile(r'\#?[A-Za-z]*\'[A-Za-z]*'): "",
        }

        self.functionCorrections = {
            re.compile(r"[#][A-z0-9]+"): self._hashtagConverter,
            re.compile(r"[!?.]{2,}"): lambda tweetWord: tweetWord[0] + " <repeat>",
            re.compile(r"(\\w+)\\1$"): lambda tweetWord: tweetWord[len(tweetWord) - 1] + " <elong>",
            re.compile(r"^[A-Z]+$"): lambda tweetWord: tweetWord.lower() + " <allcaps>"
        }

        self.multipleDotsRegex = re.compile(r"[.]{2,}")

        self.tweetIDA = []
        self.tweetIDB = []

    def _hashtagConverter(self, hashtag):
        """
        Convert hashtags to <hashtag>
        :param hashtag: The hashtag to convert.
        :return: Formatted hashtag.
        """
        hashtagBody = hashtag[1:len(hashtag)]

        return "<hashtag> {}{}".format(hashtagBody, (" <allcaps>"
                                                     if hashtagBody.upper() == hashtagBody else ""))

    def _setUpEmoticonRegularExpressions(self):
        """
        Instantiate emoticon regexes.
        """
        NormalEyes = r'[:=]'
        Wink = r'[;]'

        NoseArea = r'(|o|O|-)'

        HappyMouths = r'[D\)\]]'
        SadMouths = r'[\(\[]'
        Tongue = r'[pP]'
        OtherMouths = r'[doO/\\]'

        self.happyEmoticonRegex = (
            "(" + NormalEyes + "|" + Wink + ")" +
            NoseArea +
            "(" + Tongue + "|" + HappyMouths + ")"
        )

        self.sadEmoticonRegex = (
            "(" + NormalEyes + "|" + Wink + ")" +
            NoseArea +
            "(" + SadMouths + ")"
        )

        self.neutralEmoticonRegex = (
            "(" + NormalEyes + ")" +
            NoseArea +
            "(" + OtherMouths + ")"
        )

    def clean(self):
        """
        Begin cleaning the specified files.
        """
        dataFrameA = pd.read_csv(self.pathA, header=None, sep='\t')
        dataFrameB = pd.read_csv(self.pathB, header=None, sep='\t')

        if os._exists(self.cleanedPathA):
            os.remove(self.cleanedPathA)
        if os._exists(self.cleanedPathB):
            os.remove(self.cleanedPathB)

        with open(self.cleanedPathA, "w", newline="\n") as clean:
            output = csv.writer(clean, delimiter="\t")
            for index, row in dataFrameA.iterrows():
                self.tweetIDA.append([row[0],row[1]])
                sentiment, tweetWordList = row[4], row[5].split()
                phrase = tweetWordList[int(row[2]):int(row[3]) + 1]
                cleanPhrase = self._deepClean(phrase)
                cleanTweetWordList = self._deepClean(tweetWordList)
                output.writerow([sentiment, " ".join(cleanPhrase), " ".join(cleanTweetWordList)])

        print("Cleaned: '{}', clean file path: '{}'".format(self.pathA, self.cleanedPathA))

        with open(self.cleanedPathB, "w", newline="\n") as clean:
            output = csv.writer(clean, delimiter="\t")
            for index, row in dataFrameB.iterrows():
                self.tweetIDB.append([row[0],row[1]])
                sentiment, tweetWordList = row[2], row[3].split()
                cleanTweetWordList = self._deepClean(tweetWordList)
                output.writerow([sentiment, " ".join(cleanTweetWordList)])

        print("Cleaned: '{}', clean file path: '{}'".format(self.pathB, self.cleanedPathB))

    def _deepClean(self, tweetWordList):
        """
        Perform the cleaning process.
        :param tweetWordList: List of tweet words.
        :return: Run the cleaning process for each tweet word in the list.
        """
        return [self._runDeepCleanProcess(tweetWord) for tweetWord in tweetWordList]

    def _runDeepCleanProcess(self, tweetWord):
        """
        Run the cleaning process.
        :param tweetWord: Run cleaning on the tweet word.
        :return: Return the new tweet word.
        """
        tweetWord = re.sub(self.multipleDotsRegex, "", tweetWord)

        #tweetWord = tweetWord.replace(",", " ")

        for regex, replacementWord in self.regexCorrections.items():
            originalWord = tweetWord

            if regex.match(tweetWord):
                tweetWord = replacementWord

            if originalWord != tweetWord:
                return tweetWord

        for regex, function in self.functionCorrections.items():
            originalWord = tweetWord

            if regex.match(tweetWord):
                tweetWord = function(tweetWord)

            if originalWord != tweetWord:
                break

        return tweetWord
