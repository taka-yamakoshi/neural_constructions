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

def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x))

#Load sentences
sentence_file = open(PATH+'textfile/good_' + args[1] +'_corpus.txt')
sentence_str = sentence_file.read()
sentence_file.close()
sentence_str = sentence_str.split('\n')[:-1]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

verb_list = ["showed", "told", "guaranteed", "lent", "offered", "loaned", "left", "promised", "slipped", "wrote", "taught", "gave", "fed", "paid", "voted", "handed", "served", "tossed", "sent", "sold"]
verb_vocab = len(verb_list)

hidden_states = []
for i, verb in enumerate(verb_list):
    for j in range(10):
        sentence = sentence_str[10*i+j]
        tokenized_sentence = tokenizer.encode(sentence)
        verb_pos = tokenized_sentence.index(tokenizer.encode(verb)[1:-1][0])
        tokenized_sentence[verb_pos] = tokenizer.encode("[MASK]")[1:-1][0]
        input_tensor = torch.tensor([tokenized_sentence])
        with torch.no_grad():
            outputs = model(input_tensor)
            state = []
        for layer in outputs[1]:
            state.append(layer[0][verb_pos])
        hidden_states.append(state)

with open(PATH + 'datafile/hidden_states_' + args[1] + '.pkl','wb') as f:
    pickle.dump(hidden_states,f)
