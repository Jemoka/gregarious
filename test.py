#pylint=disable(maybe-no-member) 

# from gregarious.data import io

# dd = io.DataDescription(header_keys_special={"handle":"screen_name", "isBot":"bot"})
# df = io.DataFile('corpora/datasets/training_data_2_csv_UTF.csv', dd)
# # print(df.imported_data)
# breakpoint()

from gregarious.data.encoding import BytePairEncoder

ec = BytePairEncoder()

tokens = ec._BytePairEncoder__init_charlist_generation(["Rameo ramen reads the review of radical rare ra", "Let's break this city, make it all reeeeee ra hall!!"])
breakpoint()
bp_a = ec._BytePairEncoder__make_bytepair(tokens[0])
bp_b = ec._BytePairEncoder__make_bytepair(tokens[1])

bpct_a = ec._BytePairEncoder__return_bp_count(bp_a)
bpct_b = ec._BytePairEncoder__return_bp_count(bp_b)
print(ec._BytePairEncoder__two_counting_dicts_to_one(bpct_a, bpct_b))
breakpoint()

