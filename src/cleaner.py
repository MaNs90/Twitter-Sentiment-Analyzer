import csv
import re
import os
import pandas as pd

class Cleaner:
    def __init__(self, pathA, pathB):
        filePathRoot = "data/"

        self.pathA = filePathRoot + pathA
        self.pathB = filePathRoot + pathB

        cleanedPathRoot = "data-clean/"

        self.cleanedPathA = cleanedPathRoot + pathA
        self.cleanedPathB = cleanedPathRoot + pathB

        self._setUpEmoticonRegularExpressions()

        self.regexCorrections = {
            re.compile(r"http\S+"): "<URL>",
            re.compile(r"/"): " / ",
            re.compile(r"[@][A-z0-9]+"): "<USER>",
            re.compile(self.happyEmoticonRegex): "<SMILE>",
            re.compile(r"[8:=;]['`\-][0-9A-z]"): "<LOLFACE>",
            re.compile(self.sadEmoticonRegex): "<SADFACE>",
            re.compile(self.neutralEmoticonRegex): "<NEUTRALFACE>",
            re.compile(r"<3"): "<HEART>",
            re.compile(r"[-+]?[0-9]+"): "<NUMBER>",
            re.compile(r'\#?[A-Za-z\']*[\.\,]'): " ",
            # re.compile(r'\#?[A-Za-z]*\'[A-Za-z]*'): "",
        }

        self.functionCorrections = {
            re.compile(r"[#][A-z0-9]+"): self._hashtagConverter,
            re.compile(r"[!?.]{2,}"): lambda tweetWord: tweetWord[0] + " <REPEAT>",
            re.compile(r"(\\w+)\\1$"): lambda tweetWord: tweetWord[len(tweetWord) - 1] + " <ELONG>",
            re.compile(r"^[A-Z]+$"): lambda tweetWord: tweetWord.lower() + " <ALLCAPS>"
        }

        self.multipleDotsRegex = re.compile(r"[.]{2,}")

    def _hashtagConverter(self, hashtag):
        hashtagBody = hashtag[1:len(hashtag)]

        return "<HASHTAG> {}{}".format(hashtagBody, (" <ALLCAPS>"
                                                     if hashtagBody.upper() == hashtagBody else ""))

    def _setUpEmoticonRegularExpressions(self):
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
        dataFrameA = pd.read_csv(self.pathA, header=None, sep='\t')
        dataFrameB = pd.read_csv(self.pathB, header=None, sep='\t')

        if os._exists(self.cleanedPathA):
            os.remove(self.cleanedPathA)
        if os._exists(self.cleanedPathB):
            os.remove(self.cleanedPathB)

        with open(self.cleanedPathA, "w") as clean:
            output = csv.writer(clean, delimiter="\t")
            for index, row in dataFrameA.iterrows():
                sentiment, tweetWordList = row[4], row[5].split()
                phrase = tweetWordList[int(row[2]):int(row[3]) + 1]
                cleanPhrase = self._deepClean(phrase)
                cleanTweetWordList = self._deepClean(tweetWordList)
                output.writerow([sentiment, " ".join(cleanPhrase), " ".join(cleanTweetWordList)])

        print("Cleaned: '{}', clean file path: '{}'".format(self.pathA, self.cleanedPathA))

        with open(self.cleanedPathB, "w") as clean:
            output = csv.writer(clean, delimiter="\t")
            for index, row in dataFrameB.iterrows():
                # 263034334720716800
                # 261794390148804608
                sentiment, tweetWordList = row[2], row[3].split()
                cleanTweetWordList = self._deepClean(tweetWordList)
                output.writerow([sentiment, " ".join(cleanTweetWordList)])

        print("Cleaned: '{}', clean file path: '{}'".format(self.pathB, self.cleanedPathB))

    def _deepClean(self, tweetWordList):
        return [self._runDeepCleanProcess(tweetWord) for tweetWord in tweetWordList]

    def _runDeepCleanProcess(self, tweetWord):
        tweetWord = re.sub(self.multipleDotsRegex, "", tweetWord)

        tweetWord = tweetWord.replace(",", " ")

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
