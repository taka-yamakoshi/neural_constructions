import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

#Load the model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

#Load the verbs
file_name = args[1] + '_' + args[2]
file = open(PATH + 'textfile/'+file_name+'.txt')
verb_list = file.read().split('\n')[:-1]
file.close()

for verb in verb_list:
    #if tokenizer.encode(verb)[1] == 100:
        #print(verb)
    ids = tokenizer.encode(verb)
    tokens = []
    for id in ids[1:-1]:
        tokens.append(tokenizer.decode(id))
    print(tokens)

with torch.no_grad():
    output = model(torch.tensor([tokenizer.encode(verb_list[-1])]))
print(output[0].shape)
