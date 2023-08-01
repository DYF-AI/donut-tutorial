import pickle

ticket = "/mnt/j/dataset/document-intelligence/EATEN数据集/dataset_trainticket/real_1920.pkl"

f = open(ticket, "rb")
data1 = pickle.load(f)

print(data1)


passport = "/mnt/j/dataset/document-intelligence/EATEN数据集/dataset_passport/10w_line_text_synth.pkl"

f = open(passport, "rb")
data2 = pickle.load(f)

print(data2)