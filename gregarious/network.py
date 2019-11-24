import uuid
from .data.io import DataFile, CorpusManager
import tensorflow as tf
from keras.layers import Dense, LSTM, Input, Masking, concatenate, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Lambda, Dropout
from keras import regularizers
from keras import backend as K
from keras.models import Model, load_model

class Gregarious(object):
    def __init__(self, df, seed_model=None, optimizer="SGD", loss="binary_crossentropy", metrics=['mae', 'acc']):
        self.dataFile = df
        self.manager = CorpusManager(df)
        self.__computed_data = self.manager.compute()
        if seed_model:
            self.model = load_model(seed_model)
        else:
            self.model = self.__build(self.__computed_data["meta"]["lengths"])
            self.model.compile(optimizer, loss, metrics)

    def recompile(self, optimizer="SGD", loss="binary_crossentropy", metrics=['mae', 'acc']):
        self.model.compile(optimizer, loss, metrics)

    def train(self, epochs=10, batch_size=10, validation_split=0.1, callbacks=None, save=None):
        self.model.fit(x=self.__computed_data["ins"], y=self.__computed_data["out"], epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=callbacks)
        if save:
            self.model.save(save)

    def  __build(self, lengths):
        handles_len = lengths[0]
        names_len = lengths[1]
        desc_len = lengths[2]
        status_len = lengths[3]
        desc_len = lengths[0]
        status_len = lengths[1]
        # fnf_len = lengths[4]

        handles = Input(shape=(handles_len,))
        names = Input(shape=(names_len,))
        desc = Input(shape=(desc_len,))
        status = Input(shape=(status_len,))
        # fnf = Input(shape=(2,))

        handles_masked = Masking()(handles) 
        names_masked = Masking()(names) 
        desc_masked = Masking()(desc) 
        status_masked = Masking()(status) 
        
        handles_rec = LSTM(36)(handles_masked)
        names_rec = LSTM(36)(names_masked)
        desc_rec = LSTM(36)(desc_masked)
        status_rec = LSTM(36)(status_masked)

        handles_expanded = Lambda(lambda x: K.expand_dims(x, axis=-1))(handles)

        handles_conv_bigram = Conv1D(32, 2, padding="same")(handles_expanded)
        handles_conv_trigram = Conv1D(32, 3, padding="same")(handles_expanded)
        handles_conv_quadgram = Conv1D(32, 4, padding="same")(handles_expanded)

        handles_conv = concatenate([handles_conv_bigram, handles_conv_trigram, handles_conv_quadgram])

        names_expanded = Lambda(lambda x: K.expand_dims(x, axis=-1))(names)

        names_conv_bigram = Conv1D(32, 2, padding="same")(names_expanded)
        names_conv_trigram = Conv1D(32, 3, padding="same")(names_expanded)
        names_conv_quadgram = Conv1D(32, 4, padding="same")(names_expanded)

        names_conv = MaxPooling1D()(concatenate([names_conv_bigram, names_conv_trigram, names_conv_quadgram]))

        desc_expanded = Lambda(lambda x: K.expand_dims(x, axis=-1))(desc)

        desc_conv_bigram = Conv1D(32, 2, padding="same")(desc_expanded)
        desc_conv_trigram = Conv1D(32, 3, padding="same")(desc_expanded)
        desc_conv_quadgram = Conv1D(32, 4, padding="same")(desc_expanded)

        desc_conv = MaxPooling1D()(concatenate([desc_conv_bigram, desc_conv_trigram, desc_conv_quadgram]))

        status_expanded = Lambda(lambda x: K.expand_dims(x, axis=-1))(status)

        status_conv_bigram = Conv1D(32, 2, padding="same")(status_expanded)
        status_conv_trigram = Conv1D(32, 3, padding="same")(status_expanded)
        status_conv_quadgram = Conv1D(32, 4, padding="same")(status_expanded)

        status_conv = MaxPooling1D()(concatenate([status_conv_bigram, status_conv_trigram, status_conv_quadgram]))

        network_cat = concatenate([handles_conv, names_conv, desc_conv, status_conv], axis=1)
        # network_cat = concatenate([desc_conv, status_conv], axis=1)

        net = self.__conv_unit(network_cat)
        net = self.__conv_unit(net)
        net = self.__conv_unit(net)
        net = MaxPooling1D()(net)

        net = self.__conv_unit(net)
        net = self.__conv_unit(net)
        net = self.__conv_unit(net)
        net = MaxPooling1D()(net)

        net = self.__conv_unit(net)
        net = self.__conv_unit(net)
        net = self.__conv_unit(net)
        net = MaxPooling1D()(net)

        net = GlobalMaxPooling1D()(net)

        net = Dense(128, activation="relu")(net)
        net = Dense(128, activation="relu")(net)
        net = Dense(64, activation="relu")(net)

        net = Dropout(0.2)(net)

        net = Dense(32, activation="relu")(net)
        net = Dense(2, name="result", activation="sigmoid", kernel_regularizer=regularizers.l1_l2(0.01))(net)

        return Model(inputs=[handles, names, desc, status], outputs=net)
        # return Model(inputs=[desc, status], outputs=net)

    def __conv_unit(self, in_layer):
        bigram = Conv1D(32, 2, padding="same")(in_layer)
        trigram = Conv1D(32, 3, padding="same")(in_layer)
        quadgram = Conv1D(32, 4, padding="same")(in_layer)
        return concatenate([bigram, trigram, quadgram])


