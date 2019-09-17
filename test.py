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

from backend import encoding

enc = encoding.SentenceOneHotEncoder()
enc.train(["This is an ice cone mechelle fifer that white gold. This is an ice girl say good girl say mastpice. Friday, icy, girl ice city."])
res = enc.encode("This. is. an ice. cone. mechelle. white gold.")
print()


        # # Sentence A embedding+processing
        # embeddingA = Embedding(vocabSize, matrixEmbedSize, input_length=vocabSize, mask_zero=True)
        # embeddingA_built = embeddingA(sentenceAInput)
        # recurrentA = GRU(recurrentSize)
        # recurrentA_built = recurrentA(embeddingA_built)
        # sentenceAEmbedded = Dense(embedSize, kernel_initializer=normal)
        # sentenceAEmbedded_built = sentenceAEmbedded(recurrentA_built)

        # # Sentence B embedding+processing
        # embeddingB = Embedding(vocabSize, matrixEmbedSize, input_length=vocabSize)
        # embeddingB_built = embeddingB(sentenceBInput)
        # recurrentB = GRU(recurrentSize)
        # recurrentB_built = recurrentB(embeddingB_built)
        # sentenceBEmbedded = Dense(embedSize, kernel_initializer=normal)
        # sentenceBEmbedded_built = sentenceBEmbedded(recurrentB_built)
# Sentence A embedding+processing
        # flattenA = Flatten()
        # flattenA_built = flattenA(sentenceAInput)
        # embeddingA = Embedding(vocabSize, matrixEmbedSize, input_length=vocabSize)
        # embeddingA_built = embeddingA(flattenA_built)
        # recurrentA = GRU(recurrentSize)
        # recurrentA_built = recurrentA(embeddingA_built)
        # sentenceAEmbedded = Dense(embedSize, kernel_initializer=normal)
        # sentenceAEmbedded_built = sentenceAEmbedded(recurrentA_built)

        # # Sentence B embedding+processing
        # flattenB = Flatten()
        # flattenB_built = flattenB(sentenceBInput)
        # embeddingB = Embedding(vocabSize, matrixEmbedSize, input_length=vocabSize)
        # embeddingB_built = embeddingB(flattenB_built)
        # recurrentB = GRU(recurrentSize)
        # recurrentB_built = recurrentB(embeddingB_built)
        # sentenceBEmbedded = Dense(embedSize, kernel_initializer=normal)
        # sentenceBEmbedded_built = sentenceBEmbedded(recurrentB_built)