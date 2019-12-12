# pylint=disable(maybe-no-member)

import pickle
import keras
from gregarious.data import io
from gregarious.network import Gregarious

from keras.optimizers import Adam

# dd = io.DataDescription()
# df = io.DataFile('corpora/datasets/cresci-rtbust-2019.csv', dd, name="rtbust-1000")
# df.compile()

# df.save()
# # # print(df.imported_data)

# breakpoint()

with open("rtbust-1000.gregariousdata", "rb") as data:
    df = pickle.load(data)

# isbots = df.importedData["isBot"]
# humans = 0
# bots = 0
# for i in isbots:
    # if i == [0, 1]
        # bots+=1
    # elif i == [1, 0]:
        # humans+=1

# breakpoint()
# net = Gregarious(df, optimizer=Adam(lr=1e-4))
# net = Gregarious(df, optimizer=Adam(lr=3e-4))
# net = Gregarious(df, optimizer=Adam(lr=1e-3))
# net = Gregarious(df, optimizer=Adam(lr=2e-3))
# net = Gregarious(df, optimizer=Adam(lr=1e-3))
# net.train(epochs=150, batch_size=32, validation_split=0.1, callbacks=[keras.callbacks.TensorBoard(log_dir="./training_tb_logs/CRESTI-test-6", update_freq="batch"), keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)], save="trained_networks/CRESTI-test-6.h5")
net = Gregarious(df, seed_model="trained_networks/CRESTI-test-6.h5")
# net = Gregarious(df, optimizer=Adam(lr=1e-2))
# net = Gregarious(df, optimizer=Adam(lr=2e-2))
# net = Gregarious(df, optimizer=Adam(lr=0.1))
# breakpoint()

cm = io.CorpusManager(df)
cmDat = cm.compute()
data = cmDat["ins"]
res = net.predict(data[0], data[1], data[2], data[3])


def checkycheck(i):
    r = res[i]
    val = [1, 0] if r[0] > r[1] else [0, 1]
    print("True:", cmDat["out"][0][i], "Pred:", val)


breakpoint()

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
