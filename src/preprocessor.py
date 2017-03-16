from src.tokenizer import Tokenizer
from sklearn.preprocessing import normalize
import csv
import nltk
import numpy as np
import re


class Feature:

    def __init__(self, trainingPath):
        """
        Parent class for features.
        :param trainingPath: The path to the file to extract features from.
        """
        self.trainingPath = trainingPath
        self.data = []
        self.parse()
        self.myTweets = [tweet['Tweet'] for tweet in self.data]
        self.vector = None

    def labels(self):
        """
        Finding labels for tweets.
        :return: Labels for the tweets.
        """
        myLabels = np.empty((len(self.myTweets)),dtype=np.int16)
        index = 0
        for sentiment in self.data:
            myLabels[index] = sentiment["Sentiment"]
            index += 1
        return myLabels

    def parse(self):
        """
        Replace sentiment with 2: Positive, 1: Neutral, 0: Negative
        """
        tk = Tokenizer(preserve_case=False)
        with open(self.trainingPath) as training:
            tsvRead = csv.reader(training, delimiter="\t")
            enum = {'positive': 2, 'neutral': 1, 'negative': 0, 'unknown':3}
            tweet_dict = {}
            for line in tsvRead:
                if tk.tokenize(line[1]):
                    phrase = tk.tokenize(line[1])
                    for i,word in enumerate(phrase):
                        if i>50 and word in ["neutral","positive","negative","unknown"]:
                            phrase = phrase[:i]
                            break
                    self.data.append({'Sentiment' : enum[line[0]], 'Tweet' : phrase})


class WordEmbeddings(Feature):

    def __init__(self, trainingPath, glovePath, dim):
        """
        Class to construct word embeddings for features.
        :param trainingPath: Path of file to get training files from.
        :param glovePath: Path to glove word embeddings.
        :param dim: Dimension of glove embedding.
        """
        Feature.__init__(self, trainingPath)
        self.glovePath = glovePath
        self.dim = dim
        self.vector = np.empty([len(self.myTweets), 1])

    def glove(self, flag=False):
        """
        Construct glove word embedding feature vectors.
        :param flag: Whether we want to keep 2D data or not (feature for each word or average for tweet)
        :return: New feature vector.
        """
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
        
        print("The function found {} in the Glove embeddings out of {} total unique words".format(glove_count, len(
            all_words)))
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

        if not flag:
            features = normalize(features, axis=1)
        self.vector = features
        return features


class Lexicon(WordEmbeddings):

    def __init__(self, trainingPath, glovePath, dim, sentimentPath1,sentimentPath2):
        """
        Construct lexicons to add to word embedding feature, extends from word embeddings.
        :param trainingPath: Path of training file.
        :param glovePath: Path to glove file.
        :param dim: Dimensions of the glove feature vector.
        :param sentimentPath1: Path to unigram sentiment lexicon.
        :param sentimentPath2: Path to bigram sentiment lexicon.
        """
        WordEmbeddings.__init__(self, trainingPath, glovePath, dim)
        self.sentimentPath1 = sentimentPath1
        self.sentimentPath2 = sentimentPath2
        
    def lexicon(self, flag=False):
        """
        Begin adding lexicon features to the word embedding feature.
        :return: New feature with word embedding appended.
        """
        features = self.glove(flag=flag)
        all_words = set(w for words in self.myTweets for w in words)
        unigramLexicon = {}
        count = 0
        print("Finding lexical sentiments...")
        with open(self.sentimentPath1, 'r', encoding='utf-8') as infile:
            for line in infile:
                sentence = line.split()
                word = sentence[0]
                num = sentence[1]
                if word in all_words:
                    count += 1
                    unigramLexicon[word] = num
        print("The function found {} in the unigram sentiment lexicon out of {} total unique words".format(count, len(
            all_words)))

        bigramDict={}
        for tweet in self.myTweets:
            for i in range(1,len(tweet)):
                bigramDict[(tweet[i-1],tweet[i])] = True
              
        bigramLexicon = {}
        count = 0
        with open(self.sentimentPath2, 'r',encoding='utf-8') as infile:
            for line in infile:
                sentence =  line.split() 
                word0 = sentence[0]
                word1 = sentence[1]
                num =  sentence[2]
                if (word0,word1) in bigramDict:
                    count = count+1
                    bigramLexicon[(word0,word1)] = num
        
        print("The function found {} in the bigram sentiment lexicon out of {} total unique words".format(count,len(bigramDict.keys())))
        
        index = 0
        if flag:
            sentimentFeatures1 = []
        else:
            sentimentFeatures1 = np.empty([len(self.myTweets), 1])
        for words in self.myTweets:
            if flag:
                temp = np.zeros([75, 1])
            else:
                temp = np.empty([len(words), 1])
            count = 0
            for w in words:
                if w in unigramLexicon.keys():
                    # print("I found ", w)
                    # print("Its embedding is ", glove_small[w])
                    if flag:
                        temp[count] = float(unigramLexicon[w])/6
                    else:
                        temp[count] = unigramLexicon[w]
                else:
                    # print("I didn't find ", w)
                    temp[count] = 0
                count += 1

            if flag:
                sentimentFeatures1.append(temp)
            else:
                sentimentFeatures1[index] = np.mean(temp, axis=0)

            index += 1

        if not flag:
            sentimentFeatures1 = normalize(sentimentFeatures1, axis=0)
        
        index = 0
        if flag:
            sentimentFeatures2 = []
        else:
            sentimentFeatures2 = np.empty([len(self.myTweets), 1])
        for words in self.myTweets:
            if len(words)==0:
                continue
            if flag:
                temp = np.zeros([75, 1])
            else:
                temp = np.empty([len(words), 1])
            count = 0
            for w in range(1,len(words)):
                if (words[w-1],words[w]) in bigramLexicon:
                    if flag:
                        temp[count] = float(bigramLexicon[(words[w - 1], words[w])])/6
                    else:
                        temp[count] = bigramLexicon[(words[w-1],words[w])]
                else:
                    #print("I didn't find ", w)
                    temp[count] = 0
                count = count + 1  

            if flag:
                sentimentFeatures2.append(temp)
            else:
                sentimentFeatures2[index] = np.mean(temp, axis=0)
            index = index + 1

        if not flag:
            sentimentFeatures2 = normalize(sentimentFeatures2,axis=0)
            new_features = []
            for i, f in enumerate(features):
                new_features.append(np.append(f, sentimentFeatures1[i]))

            features = np.array(new_features)

            new_features = []
            for i, f in enumerate(features):
                new_features.append(np.append(f, sentimentFeatures2[i]))
        else:
            new_features = []
            for i, f in enumerate(features):
                n_f = []
                for j, fe in enumerate(f):
                    n_f.append(np.append(fe, sentimentFeatures1[i][j]))
                new_features.append(n_f)

            features = np.array(new_features)

            new_features = []
            for i, f in enumerate(features):
                n_f = []
                for j, fe in enumerate(f):
                    n_f.append(np.append(fe, sentimentFeatures2[i][j]))
                new_features.append(n_f)


           
        features = np.array(new_features)
        self.vector = features
        return features


class Postag(Lexicon):
    def __init__(self, trainingPath, glovePath, dim, sentimentPath1, sentimentPath2):
        """
        Add POS tags to lexicon and word embedding feature vector, extends from lexicon class.
        :param trainingPath: Path to file to get features from.
        :param glovePath: Path to glove file.
        :param dim: Number of dimensions for glove data.
        :param sentimentPath1: Path to unigram sentiment lexicon file.
        :param sentimentPath2: Path to bigram sentiment lexicon file.
        """
        Lexicon.__init__(self, trainingPath, glovePath, dim, sentimentPath1, sentimentPath2)


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

        bfLeft = r"(Ã¢â„¢Â¥|0|[oO]|Ã‚Â°|[vV]|\\$|[tT]|[xX]|;|\u0ca0|@|ÃŠËœ|Ã¢â‚¬Â¢|Ã£Æ’Â»|Ã¢â€”â€¢|\\^|Ã‚Â¬|\\*)"
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
        ) + "|(<smile>|<lolface>|<sadface>|<neutralface>)$"

        return mycompile(emoticon)

    def getRegex(self):
        """
        Calculates Twitter regex.
        :return: List containing Twitter regex and pos tags associated with them.
        """

        mycompile = lambda pat: re.compile(pat, re.UNICODE)

        # Hearts regex 37 'H'
        Hearts = mycompile("^((?:<+/?3+)+)|(<heart>)$")

        # URL regex 38 'U'
        url = mycompile("^<url>$")

        # Email regex 39 'E'
        Bound = r"(?:\W|^|$)"
        Email = mycompile(Bound + self.regex_or("(?<=(?:\W))",
                                           "(?<=(?:^))") + r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}(?=" + Bound + ")")

        # Emoticon regex 40 'E'
        Emoji_re = self.get_emoticon_regex()

        # Arrows regex 41 'A'
        Arrows = mycompile(
            "^" + self.regex_or(r"(?:<*[-Ã¢â‚¬â€¢Ã¢â‚¬â€�=]*>+|<+[-Ã¢â‚¬â€¢Ã¢â‚¬â€�=]*>*)", u"[\u2190-\u21ff]+") + "$")  # Arrow regex

        # Entity regex (HTML entity) 42 'EN'
        entity_bf = r"&(?:amp|lt|gt|quot);"
        entity = mycompile(entity_bf)

        # arbitraryAbbrev regex (captures abbreviations) 43 'AA'
        boundaryNotDot = self.regex_or("$", r"\s", r"[Ã¢â‚¬Å“\"?!,:;]", entity_bf)
        aa1 = r"(?:[A-Za-z]\.){2,}(?=" + boundaryNotDot + ")"
        aa2 = r"[^A-Za-z](?:[A-Za-z]\.){1,}[A-Za-z](?=" + boundaryNotDot + ")"
        standardAbbreviations = r"\b(?:[Mm]r|[Mm]rs|[Mm]s|[Dd]r|[Ss]r|[Jj]r|[Rr]ep|[Ss]en|[Ss]t)\."
        arbitraryAbbrev = mycompile("^" + self.regex_or(aa1, aa2, standardAbbreviations) + "$")

        # decorations regex (notes, stars, etc) 44 'D'
        decorations = mycompile(u"(?:[Ã¢â„¢Â«Ã¢â„¢Âª]+|[Ã¢Ëœâ€¦Ã¢Ëœâ€ ]+|[Ã¢â„¢Â¥Ã¢ï¿½Â¤Ã¢â„¢Â¡]+|[\u2639-\u263b]+|[\ue001-\uebbb]+)")

        # HashTag regex 45 'HT'
        Hashtag = mycompile("^<hashtag>$")

        # AtMention regex (user mentions) 46 'AM'
        AtMention = mycompile("^<user>$")

        regex_codes = [[Hearts, 'H'],
                       [url, 'U'],
                       [Email, 'EMA'],
                       [Emoji_re, 'E'],
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
        Construct a POS vector for the data to append to the lexicon and word embedding feature vector.
        :param dt: List of POS tagged dictionaries for the tweets.
        :return: POS vectors for the tweets.
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

        all_vecs = normalize(all_vecs,axis=1)
        return all_vecs

    def getvecs(self):
        """
        Calculate POS vectors and append to the word embedding and lexicon feature.
        :return: New feature vector containing all 3 features.
        """
        print("Getting POS tag vectors...")
        features = self.lexicon()
        pt = self.posTag('B')
        vec = self.pos_vectors(pt)
        pos = normalize(vec)
        new_features = []
        for i, f in enumerate(features):
            new_features.append(np.append(f, pos[i]))

        Features = np.array(new_features)
        self.vector = Features
        return Features

