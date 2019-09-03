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
from keras.layers import Input, Dense, Embedding, Flatten, Add

from nltk import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

from backend import encoding

toyDataset = ["chickpeas, eats honeybees."]



encoder = encoding.SentenceVectorizer(pad=True)
encoder.train(toyDataset)
data = encoder.encode(["chickpeas, honeybees.", "eat!", "eats some chickpeas."])

# Inputs
sentenceAInput = Input(shape=(None, ))
sentenceBInput = Input(shape=(None, ))

# Sentence A embedding+processing
embeddingA = Embedding(encoder.sequenceLength, 64, input_length=encoder.sequenceLength)
embeddingA_built = embeddingA(sentenceAInput)
flatteningA = Flatten()
flatteningA_built = flatteningA(embeddingA_built)
sentenceAEmbedded = Dense(60)
sentenceAEmbedded_built = sentenceAEmbedded(flatteningA_built)

# Sentence B embedding+processing
embeddingB = Embedding(encoder.sequenceLength, 64, input_length=encoder.sequenceLength)
embeddingB_built = embeddingB(sentenceBInput)
flatteningB = Flatten()
flatteningB_built = flatteningB(embeddingB_built)
sentenceBEmbedded = Dense(60)
sentenceBEmbedded_built = sentenceBEmbedded(flatteningB_built)

# Combining/Output
adder = Add()
added = adder([sentenceAEmbedded_built, sentenceBEmbedded_built])
score = Dense(1)
score_built = score(added)

trainer = Model(inputs=[sentenceAInput, sentenceBInput], outputs=score_built)
trainer.compile('rmsprop', 'mse')

sentenceAEmbedder = Model(inputs=sentenceAInput, outputs=sentenceAEmbedded_built)
sentenceBEmbedder = Model(inputs=sentenceBInput, outputs=sentenceBEmbedded_built)

# dummyOutput = [list(range(10)), list(range(10)), list(range(10))]

trainer.fit(x=[[data[0]], [data[1]]], y=[[0]])
