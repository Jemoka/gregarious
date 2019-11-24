
import time
import string
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
# from nltk.corpus import stopwords


class CompareEngine(object):
    @staticmethod
    def __getSynsetSimilarity(worda, wordb):
        if worda == wordb:
            return 1
        # Wu-Palmer Similarity (Wu and Palmer 1994)
        maxSim = 0
        wordaSynsets = wn.synsets(worda)
        for synseta in wordaSynsets:
            wordbSynsets = wn.synsets(wordb)
            for synsetb in wordbSynsets:
                similarity = wn.wup_similarity(synseta, synsetb)
                if similarity:
                    if similarity == 1:
                        return 1
                    if maxSim < similarity:
                        maxSim = similarity
        return maxSim

    @staticmethod
    def __POSDictify(sentence):
        assert type(sentence)==str, "Supply string setnence to _CompareEngine__POSDictify"
        posDict = {}
        taggedText = pos_tag(word_tokenize(sentence))
        for word, pos in taggedText:
            # if word.lower() not in stopwords.words('english') and word.lower() not in string.punctuation:
            if word.lower() not in string.punctuation:
                oldPOS = posDict.get(pos)
                if oldPOS:
                    posDict[pos] = oldPOS+[word.lower()]
                else:
                    posDict[pos] = [word.lower()]
        return posDict

    def __evalSimilarityByPOS(self, sentenceA, sentenceB):
        posDictA = self.__POSDictify(sentenceA)
        posDictB = self.__POSDictify(sentenceB)

        posScoreDict = {}

        for pos, wordsA in posDictA.items():
            wordsB = posDictB.get(pos)
            scores_sum = 0
            word_count = 0
            if wordsB:
                for wordB in wordsB:
                    for wordA in wordsA:
                        scores_sum += self.__getSynsetSimilarity(wordA, wordB)
                        word_count += 1
            if word_count is not 0:
                posScoreDict[pos] = scores_sum/word_count
        
        return posScoreDict

    def eval(self, sentenceA, sentenceB, type="s"):
        posDict = self.__evalSimilarityByPOS(sentenceA, sentenceB)
        if type[0].lower() == "p":
            return posDict
        scores_sum = 0
        pos_count = 0
        for _, score in posDict.items():
            scores_sum += score
            pos_count += 1
        if pos_count == 0:
            return 0
        return scores_sum/pos_count
