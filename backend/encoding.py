"""
Bada Bing, Bada Boom
The thoughts, ideas, and opinions both spoken and internalized
are the forces, that traverse all platforms, all standards
and all norms to form their unique class, unique style, and a
unique kind of ideology that future generations will look up to.

We are #!/Shabang. Copyright 2019/2020 Shabang Systems, LLC.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import time
import tensorflow as tf
import keras
import numpy as np
from keras.utils import to_categorical
from nltk import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem import WordNetLemmatizer

class SentenceOneHotEncoder(object):
        def __init__(self, minval=2):
                self.encoding_dict = {"__$UNDEF$__": 1}
                self.encoding_dict_rev = {1: "__$UNDEF$__"}
                self.occurence = {}
                self.trained = False
                self.__currentID = 2
                self.__minval = minval
                self.__maxWords = 0
                self.__lemma = WordNetLemmatizer()

        @property
        def vocabSize(self):
                return self.__currentID

        @property
        def sentSize(self):
                return self.__maxWords

        def train(self, sentences):
                assert not self.trained, "Please call SentenceOneHotEncoder.untrain to reset training."
                assert type(sentences)==list or type(sentences)==np.ndarray, "Please supply an *array* of string sentences."
                sents_tokenized = []
                for item in sentences:
                        assert type(item)==str, "Please supply an array of *string* sentences."
                        for sent in sent_tokenize(item):
                                sents_tokenized.append(sent)
                for sent in sents_tokenized:
                        words = word_tokenize(sent)
                        if len(words) > self.__maxWords:
                                self.__maxWords = len(words)
                        for word in words:
                                preppedWord = self.__lemma.lemmatize(self.__lemma.lemmatize(word.lower(), "v"), "n")
                                count = self.occurence.get(preppedWord)
                                if not count:
                                        self.occurence[preppedWord] = 1
                                else:
                                        self.occurence[preppedWord] = count+1
                                        if count+1>=self.__minval:
                                                id = self.encoding_dict.get(preppedWord)
                                                if not id:
                                                        self.encoding_dict[preppedWord] = self.__currentID 
                                                        self.encoding_dict_rev[self.__currentID] = preppedWord
                                                        self.__currentID += 1
                self.trained = True

        def untrain(self):
                print("DANGER AHEAD: you are resetting the training of this vectorizer and ALL DATA WILL BE LOST!")
                print("You have 5 seconds to kill this...")
                time.sleep(5)
                print("Welp. Your training weights is going now.")
                self.encoding_dict = {"__$UNDEF$__": 1}
                self.encoding_dict_rev = {1: "__$UNDEF$__"}
                self.trained = False
                self.occurence = {}
                self.__currentID = 2
                print("Done.")

        def encode(self, sentences):
                assert self.trained, "Model not trained! Please call SentenceOneHotEncoder.train to train."
                assert type(sentences)==list or type(sentences)==np.ndarray or type(sentences)==str, "Please supply an *array* of string sentences or a string of sentences."
                if type(sentences) == str:
                        sentences = [sentences]
                sents = []   
                for item in sentences:
                        assert type(item)==str, "Please supply an array of *string* sentences."
                        for sent in sent_tokenize(item):
                                sents.append(sent)
                sents_encoded = []
                for sent in sents:
                        words = word_tokenize(sent)
                        if len(words) > self.__maxWords:
                                self.__maxWords = len(words)
                for sent in sents:
                        words = word_tokenize(sent)
                        word_vectors = []
                        for word in words:
                                id = self.encoding_dict.get(self.__lemma.lemmatize(self.__lemma.lemmatize(word.lower(), "v"), "n"), 1)
                                word_vectors.append(id)
                        while len(word_vectors) < self.__maxWords:
                                word_vectors.append(0)
                        cats = to_categorical(np.array(word_vectors), num_classes=self.__currentID).tolist()
                        cats_n = [0]*len(cats)
                        for i, item in enumerate(cats):
                                if item[0] == 1:
                                        item = [0]*self.__currentID
                                cats_n[i] = item
                        sents_encoded.append(cats_n)
                
                return np.asarray(sents_encoded)
        
        def decode(self, sentences):
                assert self.trained, "Model not trained! Please call SentenceVectorizer.train to train."
                assert type(sentences)==list or type(sentences)==np.ndarray, "Please supply an *array* of string sentences."
                detokenizer = TreebankWordDetokenizer()
                sents_decoded = []
                for sent in sentences:
                        assert type(sent)==list or type(sent)==np.ndarray, "Please supply an array of array vector sentences."
                        if type(sent) == np.ndarray:
                                sent = sent.tolist()
                        words_decoded = []
                        for w in sent:
                                word = w.index(1)
                                if word == 0:
                                        continue
                                word_decoded = self.encoding_dict_rev[word]
                                words_decoded.append(word_decoded)
                        sents_decoded.append(detokenizer.detokenize(words_decoded))
                return sents_decoded



class SentenceVectorizer(object):
        def __init__(self, pad=False, minval=2):
                self.encoding_dict = {"__$UNDEF$__": 1}
                self.encoding_dict_rev = {1: "__$UNDEF$__"}
                self.occurence = {}
                self.trained = False
                self.pad = pad
                self.__currentID = 2
                self.__minval = minval

        @property
        def sequenceLength(self):
                return self.__currentID

        def train(self, sentences):
                assert not self.trained, "Please call SentenceVectorizer.untrain to reset training."
                assert type(sentences)==list or type(sentences)==np.ndarray, "Please supply an *array* of string sentences."
                sents_tokenized = []
                for item in sentences:
                        assert type(item)==str, "Please supply an array of *string* sentences."
                        for sent in sent_tokenize(item):
                                sents_tokenized.append(sent)
                for sent in sents_tokenized:
                        words = word_tokenize(sent)
                        for word in words:
                                count = self.occurence.get(word.lower())
                                if not count:
                                        self.occurence[word.lower()] = 1
                                else:
                                        self.occurence[word.lower()] = count+1
                                        if count+1>=self.__minval:
                                                id = self.encoding_dict.get(word.lower())
                                                if not id:
                                                        self.encoding_dict[word.lower()] = self.__currentID 
                                                        self.encoding_dict_rev[self.__currentID] = word.lower()
                                                        self.__currentID += 1
                self.trained = True

        def untrain(self):
                print("DANGER AHEAD: you are resetting the training of this vectorizer and ALL DATA WILL BE LOST!")
                print("You have 5 seconds to kill this...")
                time.sleep(5)
                print("Welp. Your training weights is going now.")
                self.encoding_dict = {"__$UNDEF$__": 1}
                self.encoding_dict_rev = {1: "__$UNDEF$__"}
                self.trained = False
                self.occurence = {}
                self.__currentID = 2
                print("Done.")

        def encode(self, sentences):
                assert self.trained, "Model not trained! Please call SentenceVectorizer.train to train."
                assert type(sentences)==list or type(sentences)==np.ndarray or type(sentences)==str, "Please supply an *array* of string sentences or a string of sentences."
                if type(sentences) == str:
                        sentences = [sentences]
                sents = []    
                for item in sentences:
                        assert type(item)==str, "Please supply an array of *string* sentences."
                        for sent in sent_tokenize(item):
                                sents.append(sent)
                sents_encoded = []
                for sent in sents:
                        words = word_tokenize(sent)
                        word_vectors = []
                        for word in words:
                                id = self.encoding_dict.get(word.lower(), 1)
                                word_vectors.append(id)
                        if self.pad:
                                while len(word_vectors)<self.__currentID:
                                        word_vectors.append(0)
                        sents_encoded.append(word_vectors)
                
                return sents_encoded
        
        def decode(self, sentences):
                assert self.trained, "Model not trained! Please call SentenceVectorizer.train to train."
                assert type(sentences)==list or type(sentences)==np.ndarray, "Please supply an *array* of string sentences."
                detokenizer = TreebankWordDetokenizer()
                sents_decoded = []
                for sent in sentences:
                        assert type(sent)==list or type(sent)==np.ndarray, "Please supply an array of array vector sentences."
                        if type(sent) == np.ndarray:
                                sent = sent.tolist()
                        words_decoded = []
                        for word in sent:
                                if word == 0:
                                        continue
                                word_decoded = self.encoding_dict_rev[word]
                                words_decoded.append(word_decoded)
                        sents_decoded.append(detokenizer.detokenize(words_decoded))
                return sents_decoded
