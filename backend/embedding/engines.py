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
from keras.layers import Input, Dense, Embedding, Flatten, Add, LSTM, Masking
from keras.optimizers import RMSprop, Adam

from keras import backend as K

class SemanticEmbedEngine(object):
    def __init__(self):
        self.trainer = None
        self.embedder_a = None
        self.embedder_b = None

    @classmethod
    def create(cls, embedSize, vocabSize, recurrentSize=None):
        if not recurrentSize:
            recurrentSize = embedSize
    
        sentenceAInput = Input(shape=(None, vocabSize))
        maskA = Masking(mask_value=0.0)(sentenceAInput)
        sentenceBInput = Input(shape=(None, vocabSize))
        maskB = Masking(mask_value=0.0)(sentenceBInput)

        normal = keras.initializers.glorot_normal()
        
        recurrentA_a = LSTM(recurrentSize, return_sequences=True)
        recurrentA_a_built = recurrentA_a(maskA)
        recurrentA_b = LSTM(recurrentSize)
        recurrentA_b_built = recurrentA_b(recurrentA_a_built)
        sentenceAEmbedded = Dense(embedSize, kernel_initializer=normal)
        sentenceAEmbedded_built = sentenceAEmbedded(recurrentA_b_built)

        recurrentB_a = LSTM(recurrentSize, return_sequences=True)
        recurrentB_a_built = recurrentB_a(maskB)
        recurrentB_b = LSTM(recurrentSize)
        recurrentB_b_built = recurrentB_b(recurrentB_a_built)
        sentenceBEmbedded = Dense(embedSize, kernel_initializer=normal)
        sentenceBEmbedded_built = sentenceBEmbedded(recurrentB_b_built)

        # Combining/Output
        adder = Add()
        added = adder([sentenceAEmbedded_built, sentenceBEmbedded_built])
        combineEmbedded = Dense(embedSize, kernel_initializer=normal)
        combineEmbedded_built = combineEmbedded(added)
        score = Dense(1, kernel_initializer=normal, activation="sigmoid")
        score_built = score(combineEmbedded_built)

        trainer = Model(inputs=[sentenceAInput, sentenceBInput], outputs=score_built)
        optimizer = Adam(lr=5e-3)
        trainer.compile(optimizer, 'logcosh')
        
        sentenceAEmbedder = Model(inputs=sentenceAInput, outputs=sentenceAEmbedded_built)
        sentenceBEmbedder = Model(inputs=sentenceBInput, outputs=sentenceBEmbedded_built)

        engine = cls()
        engine.trainer = trainer
        engine.embedder_a = sentenceAEmbedder
        engine.embedder_b = sentenceBEmbedder

        return engine

    def fit(self, sentenceVectors_a, sentenceVectors_b, similarities, batch_size=10, epochs=10, validation_split=0.0):
        self.trainer.fit(x=[sentenceVectors_a, sentenceVectors_b], y=[similarities], batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    
    def predict_diff(self, sentenceVectors_a, sentenceVectors_b):
        return self.trainer.predict([sentenceVectors_a, sentenceVectors_b])
    
    def encode(self, sentenceVector, flavor="a"):
        if flavor == "a":
            return self.embedder_a.predict([sentenceVector])
        elif flavor == "b":
            return self.embedder_b.predict([sentenceVector])
        else:
            raise Exception(str(flavor)+" is not a valid flavor. Choose between a or b.")

