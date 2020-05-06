import pickle
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x))

with open(PATH+'datafile/attention_map_'+args[1]+'_'+args[2]+'.pkl','rb') as f:
    attention = pickle.load(f)
with open(PATH+'datafile/test_performance.pkl','rb') as f:
    performance = pickle.load(f)

good_attention = np.average(attention[performance[1:]==1],axis = 0)
bad_attention = np.average(attention[performance[1:] < 0.528],axis = 0)
mean = np.average(performance[1:])
stdev = np.std(performance[1:])
print(mean-2.626*stdev)
print(attention[performance[1:]==1].shape)
print(attention[performance[1:]<0.528].shape)

const_type = ["DO","PD"]
sample_sentence = ["[CLS] Linda took the man something . [SEP]","[CLS] Linda took something to the man . [SEP]"]

fig, axis = plt.subplots(1,2)
axis[0].imshow(bad_attention)
axis[1].imshow(good_attention)
plt.show()
map = good_attention - bad_attention
plt.imshow(map)
plt.xticks([])
plt.yticks([])
plt.xticks(np.arange(map.shape[0]), sample_sentence[const_type.index(args[1])].split(" "), rotation = 90)
plt.show()

fig, axis = plt.subplots(4,5)
for i in range(4):
    for j in range(5):
        map = attention[performance[1:]==1][5*i+j]
        axis[i,j].imshow(map)
        axis[i,j].set_xticks([])
        axis[i,j].set_yticks([])
        plt.xticks(np.arange(map.shape[0]), sample_sentence[const_type.index(args[1])].split(" "), rotation = 90)
plt.show()
