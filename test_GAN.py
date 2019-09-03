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
from keras.layers import Input, Dense, Embedding, Flatten

from nltk import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

from backend import encoding

toyDataset = ["chickpeas, eats honeybees."]



encoder = encoding.SentenceVectorizer(pad=True)
encoder.train(toyDataset)
data = encoder.encode(["chickpeas, honeybees.", "eat!", "eats some chickpeas."])

inp = Input(shape=(None, ))
embedding = Embedding(encoder.sequenceLength, 64, input_length=encoder.sequenceLength)
embedding_built = embedding(inp)
flattening = Flatten()
flattening_built = flattening(embedding_built)
testDense = Dense(10)
testDense_built = testDense(flattening_built)

model = Model(inputs=inp, outputs=testDense_built)
model.compile('rmsprop', 'mse')

dummyOutput = [list(range(10)), list(range(10)), list(range(10))]

model.fit(x=[data], y=[dummyOutput])
