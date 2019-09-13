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

class SentenceOneHotEncoder(object):
        def __init__(self):
                self.encoding_dict = {"__$UND$__": 1}
                self.encoding_dict_rev = {1: "__$UND$__"}
                self.__currentID = 2
                self.trained = False
        @property
        def sequenceLength(self):
                return self.__currentID

        def train(self, sentences):
                if not self.trained:
                        assert type(sentences)==list or type(sentences)==np.ndarray, "Please supply an *array* of string sentences."
                        sents = []
                        for item in sentences:
                                assert type(item)==str, "Please supply an array of *string* sentences."
                                for sent in sent_tokenize(item):
                                        sents.append(sent)
                        for sent in sents:
                                words = word_tokenize(sent)
                                for word in words:
                                        id = self.encoding_dict.get(word.lower())
                                        if not id:
                                                self.encoding_dict[word.lower()] = self.__currentID 
                                                self.encoding_dict_rev[self.__currentID] = word.lower()
                                                self.__currentID += 1
                        self.trained = True
                else:
                        raise Exception("Please call SentenceOneHotEncoder.untrain to reset training.")
        
        def untrain(self):
                print("DANGER AHEAD: you are resetting the training of this vectorizer and ALL DATA WILL BE LOST!")
                print("You have 5 seconds to kill this...")
                time.sleep(5)
                print("Welp. Your training weights is going now.")
                self.encoding_dict = {"__$UNDEF$__": 1}
                self.encoding_dict_rev = {1: "__$UNDEF$__"}
                self.trained = False
                self.__currentID = 2
                print("Done.")

        def encode(self, sentences):
                if self.trained:
                        assert type(sentences)==list or type(sentences)==np.ndarray, "Please supply an *array* of string sentences."
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
                                        id = self.encoding_dict.get(word.lower())
                                        if id:
                                                word_vectors.append(id)
                                        else:
                                                word_vectors.append(1)
                                sents_encoded.append(to_categorical(np.array(word_vectors), num_classes=self.__currentID))
                        newarr =  np.array([])
                        for sent_encoded in sents_encoded:
                                newarr = np.concatenate(newarr,sent_encoded, axis=2)
                        return newarr
                else:
                        raise Exception("Model not trained! Please call SentenceOneHotEncoder.train to train.")
        
        def decode(self, sentences):
                assert type(sentences)==list or type(sentences)==np.ndarray, "Please supply an *array* of string sentences."
                detokenizer = TreebankWordDetokenizer()
                sents_decoded = []
                for sent in sentences:
                        assert type(sent)==list or type(sent)==np.ndarray, "Please supply an array of array one-hot vector sentences."
                        if type(sent) == np.ndarray:
                                sent = sent.tolist()
                        words_decoded = []
                        for word in sent:
                                indx = word.index(1)
                                if indx == 0:
                                        continue
                                word_decoded = self.encoding_dict_rev[indx]
                                words_decoded.append(word_decoded)
                        sents_decoded.append(detokenizer.detokenize(words_decoded))
                return sents_decoded

class SentenceVectorizer(object):
        def __init__(self, pad=False):
                self.encoding_dict = {"__$UNDEF$__": 1}
                self.encoding_dict_rev = {1: "__$UNDEF$__"}
                self.trained = False
                self.pad = pad
                self.__currentID = 2

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
