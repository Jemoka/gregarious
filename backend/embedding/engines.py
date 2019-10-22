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
from keras.layers import Input, Dense, Embedding, Flatten, Add, LSTM, Masking, Concatenate, Conv1D, MaxPooling1D
from keras.optimizers import RMSprop, Adam, SGD, Adagrad

from keras import backend as K

class SemanticEmbedEngine(object):
    def __init__(self):
        self.trainer = None
        self.embedder_a = None
        self.embedder_b = None

    @classmethod
    def create(cls, embedSize, vocabSize, paddedSentSize, recurrentSize=None):
        if not recurrentSize:
            recurrentSize = embedSize
    
        sentenceAInput = Input(shape=(paddedSentSize, vocabSize))
        # maskA = Masking(mask_value=0.0)(sentenceAInput)
        sentenceBInput = Input(shape=(paddedSentSize, vocabSize))
        # maskB = Masking(mask_value=0.0)(sentenceBInput)

        normal = keras.initializers.glorot_normal()

        conv_A_a = Conv1D(recurrentSize, 5)
        conv_A_a_built = conv_A_a(sentenceAInput)
        conv_A_b = Conv1D(recurrentSize, 5)
        conv_A_b_built = conv_A_b(conv_A_a_built)
        conv_A_c = Conv1D(recurrentSize, 5)
        conv_A_c_built = conv_A_c(MaxPooling1D()(conv_A_b_built))
        # conv_A_flat = Flatten()(conv_A_c_built)
        dense_A_a = Dense(embedSize, kernel_initializer=normal, activation="relu")
        dense_A_a_built = dense_A_a(conv_A_c_built)
        dense_A_b = Dense(embedSize, kernel_initializer=normal, activation="relu")
        dense_A_b_built = dense_A_b(dense_A_a_built)
        sentenceAEmbedded = Dense(embedSize, kernel_initializer=normal, activation="relu")
        sentenceAEmbedded_built = sentenceAEmbedded(dense_A_b_built)
        

        conv_B_a = Conv1D(recurrentSize, 5)
        conv_B_a_built = conv_B_a(sentenceBInput)
        conv_B_b = Conv1D(recurrentSize, 5)
        conv_B_b_built = conv_B_b(conv_B_a_built)
        conv_B_c = Conv1D(recurrentSize, 5)
        conv_B_c_built = conv_B_c(conv_B_b_built)
        # conv_B_flat = Flatten()(conv_B_c_built)
        dense_B_a = Dense(embedSize, kernel_initializer=normal, activation="relu")
        dense_B_a_built = dense_B_a(conv_B_c_built)
        dense_B_b = Dense(embedSize, kernel_initializer=normal, activation="relu")
        dense_B_b_built = dense_B_b(dense_B_a_built)
        sentenceBEmbedded = Dense(embedSize, kernel_initializer=normal, activation="relu")
        sentenceBEmbedded_built = sentenceBEmbedded(dense_B_b_built)

        # Combining/Output
        adder = Concatenate(axis=1)
        added = adder([sentenceAEmbedded_built, sentenceBEmbedded_built])
        recurrentA = LSTM(recurrentSize*2, return_sequences=True)
        recurrentA_built = recurrentA(added)
        recurrentB = LSTM(recurrentSize*2)
        recurrentB_built = recurrentB(recurrentA_built)
        combineEmbedded = Dense(embedSize, kernel_initializer=normal, activation="relu")
        combineEmbedded_built = combineEmbedded(recurrentB_built)
        score = Dense(1, kernel_initializer=normal, activation="relu")
        score_built = score(combineEmbedded_built)

        trainer = Model(inputs=[sentenceAInput, sentenceBInput], outputs=score_built)
        optimizer = Adam(lr=1e-3)
        trainer.compile(optimizer, 'mae')
        
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

