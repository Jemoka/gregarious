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
from backend.embedding import SemanticEmbedEngine

toyDataset = ["chickpeas, honeybees.", "eat!", "eats some chickpeas."]

encoder = encoding.SentenceVectorizer(pad=True)
encoder.train(toyDataset)
data = encoder.encode("chickpeas! honeybees. Eat! some groups of chickpeas.")
sents1 = [data[0], data[1]]
sents2 = [data[2], data[3]]
outs = [0, 1]

engine = SemanticEmbedEngine.create(60, encoder.sequenceLength)
engine.fit(sents1, sents2, outs)


# Inputs
