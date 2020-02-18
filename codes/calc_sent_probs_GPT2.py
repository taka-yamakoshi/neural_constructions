import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
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
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()


prob_list = []

for sentence in sentence_str:
    tokenized_text = sentence.split(" ")
    sent_prob = 0
    sent_len = len(sentence.split(" "))
    for timestep in range(2,sent_len-1):
        str_input = " ".join(tokenized_text[1:timestep])
        indexed_tokens = tokenizer.encode(str_input)
        tokens_tensor = torch.tensor([indexed_tokens])
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]
        log_probs = predictions[0,-1,:]
        prob = log_probs[tokenizer.encode([tokenized_text[timestep]])].item()
        sent_prob += prob
    prob_list.append(sent_prob)
    print(sent_prob)
#Save the data
with open(PATH + 'datafile/GPT2_sent_log_probs_' + args[1] + '_' + args[2] + '.pkl','wb') as f:
    pickle.dump(np.array(prob_list),f)
