import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import csv
import logging
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x))
def calc_prob(tokenized_sentence,timestep):
    tokens_tensor = torch.tensor([tokenized_sentence[:timestep]])
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]
        log_probs = predictions[0,-1,:]
        prob = log_probs[tokenized_sentence[timestep]].item()
    return prob
def calc_sent_prob(sentence):
    words = sentence.split(" ")
    words.append(".")
    tokenized_sentence = tokenizer.encode(" ".join(words))
    prob_list = [calc_prob(tokenized_sentence,timestep) for timestep in range(1,len(tokenized_sentence))]
    sent_prob = np.sum(np.array(prob_list))
    return sent_prob

#Load sentences
with open(PATH+'textfile/generated_pairs.csv') as f:
    reader = csv.reader(f)
    file = [row for row in reader]
    head = file[0]
    corpus = file[1:]
if args[1] == "recipient":
    sent_list_alt = [[pair[head.index('DOsentence')],pair[head.index('PDsentence')]] for pair in corpus if pair[head.index('recipient_id')] == args[2] and pair[head.index('classification')] == "alternating"]
    sent_list_nonalt = [[pair[head.index('DOsentence')],pair[head.index('PDsentence')]] for pair in corpus if pair[head.index('recipient_id')] == args[2] and pair[head.index('classification')] == "non-alternating"]
elif args[1] == "theme":
    sent_list_alt = [[pair[head.index('DOsentence')],pair[head.index('PDsentence')]] for pair in corpus if pair[head.index('theme_type')] == args[2] and pair[head.index('classification')] == "alternating"]
    sent_list_nonalt = [[pair[head.index('DOsentence')],pair[head.index('PDsentence')]] for pair in corpus if pair[head.index('theme_type')] == args[2] and pair[head.index('classification')] == "non-alternating"]
else:
    logging.error('Invalid comparison category: should be either "recipient" or "theme"')


#Load the model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

#Calculate probability
DO_alt_prob = [calc_sent_prob(sentence[0]) for sentence in sent_list_alt]
PD_alt_prob = [calc_sent_prob(sentence[1]) for sentence in sent_list_alt]
DO_nonalt_prob = [calc_sent_prob(sentence[0]) for sentence in sent_list_nonalt]
PD_nonalt_prob = [calc_sent_prob(sentence[1]) for sentence in sent_list_nonalt]

alt_ratio = np.array(DO_alt_prob) - np.array(PD_alt_prob)
nonalt_ratio = np.array(DO_nonalt_prob) - np.array(PD_nonalt_prob)

#Save the data
with open(PATH + 'datafile/GPT2_log_ratio_' + args[2] + '.pkl','wb') as f:
    pickle.dump([alt_ratio,nonalt_ratio],f)
