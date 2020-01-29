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


#Load the model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()


prob_list = []

for sentence in sentence_str:
    tokenized_text = sentence.split(" ")
    masked_index = 2
    masked_text = tokenized_text.copy()
    masked_text[masked_index] = '[MASK]'
    indexed_tokens = tokenizer.convert_tokens_to_ids(masked_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]
    probs = predictions[0, masked_index]
    log_probs = torch.log(softmax(probs))
    prob = log_probs[tokenizer.convert_tokens_to_ids(tokenized_text[masked_index])].item()
    prob_list.append(prob)

#Save the data
with open(PATH + 'datafile/log_probs_' + args[1] + '_' + args[2] + '.pkl','wb') as f:
    pickle.dump(np.array(prob_list),f)
