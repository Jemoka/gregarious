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

# import keras
# from keras.utils import to_categorical
# from keras.models import Model
# from keras.layers import Input, Dense, Embedding, Flatten, Add

# from nltk import sent_tokenize, word_tokenize
# from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import gutenberg as gt
from nltk.corpus import reuters
from nltk.corpus import :

from backend import encoding
from backend.utils import CompareEngine
from backend.encoding import SentenceVectorizer
from backend.embedding.engines import SemanticEmbedEngine
from backend.io import CorpusManager

import random

# # from tqdm import tqdm56
# enc = encoding.SentenceOneHotEncoder()
# enc.train(["This is a dummy dataset. A dummy dataset is dumb. Dumb and dummer. Quicks!"])

# input_a_fake = enc.encode(["This is a long, dummy sentence. This is another. long."])
# input_b_fake = enc.encode(["dummy is a. SENTENCE long. Quicks!"])
# outputs_fake = [0.1, 0.8, 1]

# print(input_a_fake)

# embedEngine = SemanticEmbedEngine_V2.create(128, enc.sequenceLength)
# embedEngine.fit(input_a_fake, input_b_fake, outputs_fake, epochs=10, batch_size=16)


# training_data_raw = ""
# for fileid in gt.fileids():
#     training_data_raw = training_data_raw+gt.raw(fileid)

# raw = reuters.raw()
# raw_n = ''
# for i, l in enumerate(raw):
    # raw_n = raw_n+l
    # if i>=999999:
        # break
# # print(raw_n)    
# manager = CorpusManager(raw_n)
# # enc = encoding.SentenceOneHotEncoder()
# manager = CorpusManager(gt.raw("austen-sense.txt"))
# manager.compile(dup_factor=0.1, save_dir="corpora/reuters-toy", workers=50)

# encoder = SentenceVectorizer(pad=True, minval=3)
# print("Seatbelts please! Loading a database file...")
manager = CorpusManager.load("corpora/reuters-toy/CM_compdata_a9df2.cpmgr")
# # manager = CorpusManager.load("gutenberg/CM_compdata_35c3f.cpmgr")
# manager = CorpusManager.load("austen-sense/CM_compdata_d4530.cpmgr")
# manager = CorpusManager.load("corpora/reuters-toy/CM_compdata_4c401.cpmgr")
# manager.compile(10000, save_dir="corpora/austen-sense-toy")
# manager.compile(20)
# manager = CorpusManager.load("austen-sense-new/CM_compdata_078dc.cpmgr")
print("Done.")
# manager = CorpusManager("This is a terrable sentence. \n Whatever! \n The string marks a silly sentence. \n Bleh, chick peas, honey bees. \n Groups of chickpeas. Honeies. Bees.")
# manager.compile(size=100)
# manager.encoder.untrain()
input_a, input_b, outputs = manager.dump(False)
test_a, test_b, test_out = manager.sample(10, False)

# Training
embedEngine = SemanticEmbedEngine.create(256, manager.sequenceLength, manager.sentSize, recurrentSize=256)
embedEngine.fit(input_a, input_b, outputs, epochs=50, batch_size=32, validation_split=0.02)
diffs = embedEngine.predict_diff(test_a, test_b)

# print(diffs)
# print(test_out)


# manager.compile(size=1000, save_dir="austen-sense")
# print(manager.generate(100, True))
# compEngine = CompareEnsgine()

# encoder = encoding.SentenceVectorizer(pad=True)
# encoder.train(austen)
# input_a = []
# input_b = []
# outputs = []

# for _ in tqdm(range(1000)):
#     linea = austen[random.randint(0, len(austen))]
#     lineb = austen[random.randint(0, len(austen))]``
#     while linea.strip() == "":
#         linea = austen[random.randint(0, len(austen))]
#     while lineb.strip() == "":
#         lineb = austen[random.randint(0, len(austen))]
#     similarity = compEngine.eval(linea, lineb)
#     la_enc = encoder.encode(linea)
#     lb_enc = encoder.encode(lineb)
#     input_a = input_a + [la_enc[0]]
#     input_b = input_b + [lb_enc[0]]
#     outputs.append(similarity)


print("")
