#pylint=disable(maybe-no-member)

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

import tensorflow as tf
import numpy as np

import keras
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, LSTM

from nltk import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

from backend import encoding

toyDataset = ["chickpeas, eats honeybees."]



encoder = encoding.SentenceOneHotEncoder()
encoder.train(toyDataset)
enc = encoder.encode(["chickpeas, honeybees."])

enc_data = enc

# inp = Input(shape=(None, encoder.currentID))
# lstm = LSTM(encoder.currentID, input_shape=(None, encoder.currentID))
# lstm_built = lstm(inp)

# model = Model(inputs=[inp], outputs=[lstm_built])
# model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit([enc_data], [enc_data], epochs=10)


inp = Input(shape=(3, 6))
lstm = LSTM(3, batch_input_shape=(None, 3, 6))
lstm_built = lstm(inp)

model = Model(inputs=[inp], outputs=[lstm_built])
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
step = np.array([1, 2, 3, 4, 5, 6])
dataItem = np.array([step, step, step])
batch = np.array([dataItem, dataItem])
model.fit(batch, batch, epochs=10)


# sentences = []
# for item in toyDataset:
#     for sent in sent_tokenize(item):
#         sentences.append(sent)
# sents_tokenized = []
# for sentence in sentences:
#     sents_tokenized.append(word_tokenize(sentence))
# sent_vectors = []
# encoding_dict = {}
# currentID = 0
# for sent in sents_tokenized:
#         word_vectors = []
#         for word in sent:
#                 id = encoding_dict.get(word.lower())
#                 if id:
#                         word_vectors.append(id)
#                 else:
#                         encoding_dict[word.lower()] = currentID 
#                         word_vectors.append(currentID)
#                         currentID += 1
#         sent_vectors.append(word_vectors)
# sents_encoded = []
# for sent in sent_vectors:
#         sents_encoded.append(to_categorical(sent, num_classes=currentID+1)) #extra one for something that is not trained for
# print(sents_encoded)

