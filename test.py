#pylint=disable(maybe-no-member) 

import pickle
import keras
from gregarious.data import io
from gregarious.network import Gregarious

from keras.optimizers import Adam

dd = io.DataDescription()
# df = io.DataFile('corpora/datasets/training_data_2_csv_UTF.csv', dd)
# df.compile()

# df.save()
# # print(df.imported_data)

# breakpoint()

with open("2fa59588.gregariousdata", "rb") as data:
    df = pickle.load(data)

# net = Gregarious(df, optimizer=Adam(lr=1e-4))
net = Gregarious(df, "trained_networks/Test_Run-3.h5")
net.recompile("adam", "binary_crossentropy", ["mae", "acc"])
net.train(epochs=150, batch_size=32, validation_split=0.05, callbacks=[keras.callbacks.TensorBoard(log_dir="./training_tb_logs/R4", update_freq="batch")], save="trained_networks/Test_Run-3-1.h5")

# from gregarious.data.encoding import BytePairEncoder

# ec = BytePairEncoder()

# tokens = ec.encode(["Reader reads the reading read to him by another reader.", "Ron reads a reading as well, for he is loving his kissings of the reading."], factor=10)
# print(tokens)

# breakpoint()
# # # print(ec.combine(tokens[0], ('r', 'a')))
# bp_a = ec._BytePairEncoder__make_bytepair(tokens[0])
# bp_b = ec._BytePairEncoder__make_bytepair(tokens[1])

# bpct_a = ec._BytePairEncoder__return_bp_count(bp_a)
# bpct_b = ec._BytePairEncoder__return_bp_count(bp_b)
# print(ec._BytePairEncoder__two_counting_dicts_to_one(bpct_a, bpct_b))
# breakpoint()
