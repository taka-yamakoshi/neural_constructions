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
sentence_str = "[SEP] the man [MASK] it [MASK] them . [SEP]"


#Load the model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

tokenized_text = sentence_str.split(" ")
masked_text = tokenized_text.copy()
indexed_tokens = tokenizer.convert_tokens_to_ids(masked_text)
tokens_tensor = torch.tensor([indexed_tokens])
with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]
first_probs = predictions[0, 3]
first_log_probs = torch.log(softmax(first_probs)).numpy()
first_id = np.argsort(first_log_probs)[::-1][:50]
second_probs = predictions[0, 5]
second_log_probs = torch.log(softmax(second_probs)).numpy()
second_id = np.argsort(second_log_probs)[::-1][:50]
with open(PATH + 'textfile/double_mask_' + args[1] + '.txt','w') as f:
    f.write(sentence_str)
    f.writelines('\n')
    for first_num, second_num in zip(first_id,second_id):
        f.write(tokenizer.convert_ids_to_tokens([first_num])[0])
        f.writelines('\t')
        f.write(tokenizer.convert_ids_to_tokens([second_num])[0])
        f.writelines('\n')
