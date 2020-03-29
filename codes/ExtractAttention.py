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
sentence_file = open(PATH+'textfile/'+ args[1] + '_' + args[2] +'.txt')
sentence_str = sentence_file.read()
sentence_file.close()
sentence_str = sentence_str.split('\n')[:-1]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

attention_list = []
for sentence in sentence_str:
    tokenized_text = sentence.split(" ")
    masked_index = 3
    masked_text = tokenized_text.copy()
    masked_text[masked_index] = '[MASK]'
    indexed_tokens = tokenizer.convert_tokens_to_ids(masked_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    with torch.no_grad():
        outputs = model(tokens_tensor)
        attention = []
        for layer in outputs[2]:
            attention.append(list(layer[0].numpy()))
        attention_list.append(np.array(attention))
ave_attention = np.average(np.array(attention_list),axis = 0)
fig, axis = plt.subplots(12,12)
for i in range(12):
    for j in range(12):
        axis[i,j].imshow(ave_attention[i][j])
plt.show()
