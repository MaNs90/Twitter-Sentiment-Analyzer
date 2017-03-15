import csv
import numpy as np
from src.tokenizer import Tokenizer
from sklearn.preprocessing import normalize
import re
import nltk


class Feature:
    # Parent class for features

    def __init__(self, trainingPath):
        self.trainingPath = trainingPath
        self.data = []
        self.parse()
        self.myTweets = [tweet['Tweet'] for tweet in self.data]
        self.vector = None

    def labels(self):
        myLabels = np.empty((len(self.myTweets)),dtype=np.int16)
        index = 0
        for sentiment in self.data:
            myLabels[index] = sentiment["Sentiment"]
            index += 1
        return myLabels

    def parse(self):
        tk = Tokenizer(preserve_case=False)
        with open(self.trainingPath) as training:
            tsvRead = csv.reader(training, delimiter="\t")
            enum = {'positive': 2, 'neutral': 1, 'negative': 0}
            tweet_dict = {}
            for line in tsvRead:
                self.data.append({'Sentiment' : enum[line[0]], 'Tweet' : tk.tokenize(line[1])})


class WordEmbeddings(Feature):

    def __init__(self, trainingPath, glovePath, dim):
        Feature.__init__(self, trainingPath)
        self.glovePath = glovePath
        self.dim = dim
        self.vector = np.empty([len(self.myTweets), 1])

    def glove(self, flag=False):
        glove_embedding = {}

        all_words = set()
        for tweet in self.myTweets:
            all_words = all_words.union(set(w for w in tweet))
        glove_count = 0
        with open(self.glovePath, 'r', encoding='utf-8') as infile:
            for line in infile:
                parts = line.split()
                word = parts[0]
                nums = list(map(float, parts[1:]))
                if word in all_words:
                    glove_count += 1
                    glove_embedding[word] = np.array(nums)
        index = 0
        if flag:
            features = []
        else:
            features = np.empty([len(self.myTweets), self.dim])
        for words in self.myTweets:
            if flag:
                temp = np.zeros([75, self.dim])
            else:
                temp = np.empty([len(words), self.dim])
            count = 0
            for w in words:
                if w in glove_embedding.keys():
                    temp[count] = glove_embedding[w]
                else:
                    temp[count] = np.zeros(self.dim)
                count += 1
            if flag:
                features.append(temp)
            else:
                features[index] = np.mean(temp, axis=0)
            index += 1

        self.vector = features
        return features


class Lexicon(WordEmbeddings):

    def __init__(self, trainingPath, glovePath, dim, sentimentPath):
        WordEmbeddings.__init__(self, trainingPath, glovePath, dim)
        self.sentimentPath = sentimentPath

    def lexicon(self):
        features = self.glove()
        all_words = set(w for words in self.myTweets for w in words)
        unigramLexicon = {}
        count = 0
        print("Finding lexical sentiments...")
        with open(self.sentimentPath, 'r', encoding='utf-8') as infile:
            for line in infile:
                sentence = line.split()
                word = sentence[0]
                num = sentence[1]
                if word in all_words:
                    count += 1
                    unigramLexicon[word] = num
        print("The function found {} in the unigram sentiment lexicon out of {} total unique words".format(count, len(
            all_words)))

        index = 0
        sentimentFeatures = np.empty([len(self.myTweets), 1])
        for words in self.myTweets:
            temp = np.empty([len(words), 1])
            count = 0
            for w in words:
                if w in unigramLexicon.keys():
                    # print("I found ", w)
                    # print("Its embedding is ", glove_small[w])
                    temp[count] = unigramLexicon[w]
                else:
                    # print("I didn't find ", w)
                    temp[count] = 0
                count += 1

            sentimentFeatures[index] = np.mean(temp, axis=0)
            index += 1

        sentimentFeatures = normalize(sentimentFeatures, axis=0)
        new_features = []
        for i, f in enumerate(features):
            new_features.append(np.append(f, sentimentFeatures[i]))

        print(sentimentFeatures)
        features = np.array(new_features)

        self.vector = features
        return features


class Postag(Lexicon):
    def __init__(self, trainingPath, glovePath, dim, sentimentPath):
        Lexicon.__init__(self, trainingPath, glovePath, dim, sentimentPath)


    def regex_or(self, *items):
        """
        Combines multiple regex expressions using OR.
        :param items: Expressions to combine.
        :return: Combined expressions.
        """
        return '(?:' + '|'.join(items) + ')'

    def get_emoticon_regex(self):
        """
        Gets the emoticon regex expression.
        :return: Emoticon regex expression.
        """
        mycompile = lambda pat: re.compile(pat, re.UNICODE)

        normalEyes = "[:=]"
        wink = "[;]"
        noseArea = "(?:|-|[^a-zA-Z0-9 ])"
        happyMouths = r"[D\)\]\}]+"
        sadMouths = r"[\(\[\{]+"
        tongue = "[pPd3]+"
        otherMouths = r"(?:[oO]+|[/\\]+|[vV]+|[Ss]+|[|]+)"

        bfLeft = r"(â™¥|0|[oO]|Â°|[vV]|\\$|[tT]|[xX]|;|\u0ca0|@|Ê˜|â€¢|ãƒ»|â—•|\\^|Â¬|\\*)"
        bfCenter = r"(?:[\.]|[_-]+)"
        bfRight = r"\1"
        s3 = r"(?:--['\"])"
        s4 = r"(?:<|&lt;|>|&gt;)[\._-]+(?:<|&lt;|>|&gt;)"
        s5 = "(?:[.][_]+[.])"
        basicface = "(?:" + bfLeft + bfCenter + bfRight + ")|" + s3 + "|" + s4 + "|" + s5

        oOEmote = r"(?:[oO]" + bfCenter + r"[oO])"

        emoticon = "^" + self.regex_or(
            "(?:>|&gt;)?" + self.regex_or(normalEyes, wink) + self.regex_or(noseArea, "[Oo]") + self.regex_or(
                tongue + r"(?=\W|$|RT|rt|Rt)", otherMouths + r"(?=\W|$|RT|rt|Rt)", sadMouths, happyMouths),
            self.regex_or("(?<=(?: ))", "(?<=(?:^))") + self.regex_or(sadMouths, happyMouths, otherMouths) + noseArea +
            self.regex_or(normalEyes, wink) + "(?:<|&lt;)?",
            basicface,
            oOEmote
        ) + "$"

        return mycompile(emoticon)

    def getRegex(self):
        """
        Calculates Twitter regex.
        :return: List containing Twitter regex and pos tags associated with them.
        """

        mycompile = lambda pat: re.compile(pat, re.UNICODE)

        # Hearts regex 37 'H'
        Hearts = mycompile("^(?:<+/?3+)+$")

        # URL regex 38 'U'
        url = mycompile("^<url>$")

        # Email regex 39 'E'
        Bound = r"(?:\W|^|$)"
        Email = mycompile(Bound + self.regex_or("(?<=(?:\W))",
                                           "(?<=(?:^))") + r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}(?=" + Bound + ")")

        # Emoticon regex 40 'E'
        Emoji_re_test = self.get_emoticon_regex()

        # Arrows regex 41 'A'
        Arrows = mycompile(
            "^" + self.regex_or(r"(?:<*[-â€•â€”=]*>+|<+[-â€•â€”=]*>*)", u"[\u2190-\u21ff]+") + "$")  # Arrow regex

        # Entity regex (HTML entity) 42 'EN'
        entity_bf = r"&(?:amp|lt|gt|quot);"
        entity = mycompile(entity_bf)

        # arbitraryAbbrev regex (captures abbreviations) 43 'AA'
        boundaryNotDot = self.regex_or("$", r"\s", r"[â€œ\"?!,:;]", entity_bf)
        aa1 = r"(?:[A-Za-z]\.){2,}(?=" + boundaryNotDot + ")"
        aa2 = r"[^A-Za-z](?:[A-Za-z]\.){1,}[A-Za-z](?=" + boundaryNotDot + ")"
        standardAbbreviations = r"\b(?:[Mm]r|[Mm]rs|[Mm]s|[Dd]r|[Ss]r|[Jj]r|[Rr]ep|[Ss]en|[Ss]t)\."
        arbitraryAbbrev = mycompile("^" + self.regex_or(aa1, aa2, standardAbbreviations) + "$")

        # decorations regex (notes, stars, etc) 44 'D'
        decorations = mycompile(u"(?:[â™«â™ª]+|[â˜…â˜†]+|[â™¥â�¤â™¡]+|[\u2639-\u263b]+|[\ue001-\uebbb]+)")

        # HashTag regex 45 'HT'
        Hashtag = mycompile("^#[a-zA-Z0-9_]+$")

        # AtMention regex (user mentions) 46 'AM'
        AtMention = mycompile("^<user>$")

        regex_codes = [[Hearts, 'H'],
                       [url, 'U'],
                       [Email, 'EMA'],
                       [Emoji_re_test, 'E'],
                       [Arrows, 'A'],
                       [entity, 'EN'],
                       [arbitraryAbbrev, 'AA'],
                       [decorations, 'D'],
                       [Hashtag, 'HT'],
                       [AtMention, 'AM']]

        return regex_codes

    def posTag(self, part):
        """
        Tag the words with nltk pos_tagging or twitter specific tags.
        :param data: List of dictionaries containing information on the tweets.
        :return: List of dictionaries containing information on tweets and POS tags.
        """
        r_codes = self.getRegex()

        pos_data = self.data
        for id, tweet_info in enumerate(self.data):
            tweet = tweet_info['Tweet']
            new_sent = []
            for word in tweet:
                match = False
                for rc in r_codes:
                    if re.match(rc[0], word):
                        new_sent.append([(word, rc[1])])
                        match = True
                        break
                if not match:
                    new_sent.append(nltk.pos_tag([word]))

            new_dict = tweet_info
            new_dict['POS'] = new_sent

            if part == 'A':
                marked_tweet = tweet_info['Marked instance']
                new_marked = []
                for m_word in marked_tweet:
                    match = False
                    for rc in r_codes:
                        if re.match(rc[0], m_word):
                            new_marked.append([(m_word, rc[1])])
                            match = True
                            break
                    if not match:
                        new_marked.append(nltk.pos_tag([m_word]))

                new_dict['POS marked'] = new_marked

            pos_data[id] = new_dict

        return pos_data

    def pos_vectors(self, dt):
        """
        Occurrence vector test
        :param data:
        :return:
        """

        cats = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
                'POS', 'PRP', 'PRP$', 'RB', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP',
                'WP$',
                'WRB', 'H', 'U', 'EMA', 'E', 'A', 'EN', 'AA', 'D', 'HT', 'AM']

        all_vecs = []
        tweet_vec = np.zeros(46)
        for line in dt:
            pos_tags = line['POS']
            tweet_vec = np.zeros(46)
            for word in pos_tags:
                tag = word[0][1]
                for i, ct_tag in enumerate(cats):
                    if tag == ct_tag:
                        tweet_vec[i] += 1

            all_vecs.append(tweet_vec)

        return all_vecs

    def getvecs(self):
        print("Getting POS tag vectors...")
        features = self.lexicon()
        pt = self.posTag('B')
        vec = self.pos_vectors(pt)
        pos = normalize(vec)
        new_features = []
        for i, f in enumerate(features):
            new_features.append(np.append(f, pos[i]))

        Features = np.array(new_features)
        return Features

