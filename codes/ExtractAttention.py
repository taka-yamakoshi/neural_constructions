import pickle
import numpy as np
import csv
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
with open(PATH+'textfile/generated_pairs.csv') as f:
    reader = csv.reader(f)
    file = [row for row in reader]
    head = file[0]
    corpus = file[1:]
sent_list = [[pair[head.index('DOsentence')],pair[head.index('PDsentence')]] for pair in corpus if pair[head.index('recipient_id')] == args[2]]
verb_list = [corpus[25*i][head.index('DOsentence')].split(" ")[1] for i in range(200)]

#Load the model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

const_type = ["DO","PD"]

##TODO: Account for varying length of tokenized_sentence##
attention_list = []
mask_id = tokenizer.encode("[MASK]")[1:-1][0]
for i, verb in enumerate(verb_list):
    verb_id = tokenizer.encode(verb)[1:-1]
    if len(verb_id) == 1:
        for sentence in sent_list[5*i:5*(i+1)]:
            words = sentence[const_type.index(args[1])].split(" ")
            words.append(".")
            tokenized_sentence = tokenizer.encode(" ".join(words))
            if len(tokenized_sentence) == 8:
                verb_pos = tokenized_sentence.index(verb_id[0])
                tokenized_sentence[verb_pos] = mask_id
                input_tensor = torch.tensor([tokenized_sentence])
                with torch.no_grad():
                    outputs = model(input_tensor)
                    attention = []
                    for layer in outputs[2]:
                        attention.append(list(layer[0].numpy()))
                    attention_list.append(np.array(attention))
ave_attention = np.average(np.array(attention_list),axis = 0)
with open(PATH+'datafile/attention_map.pkl','wb') as f:
    pickle.dump(ave_attention,f)
fig, axis = plt.subplots(12,12)
for i in range(12):
    for j in range(12):
        axis[i,j].imshow(ave_attention[i][j])
plt.show()
