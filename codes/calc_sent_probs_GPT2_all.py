import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import csv
import logging
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

def calc_sent_prob(sentence):
    words = sentence.split(" ")
    words.append(".")
    tokenized_sentence = tokenizer.encode(" ".join(words), add_special_tokens=True)
    input_ids = torch.tensor(tokenized_sentence).unsqueeze(0)
    outputs = model(input_ids, labels=input_ids)
    predictions = outputs[1][0]
    prob_list = np.array([torch.log(F.softmax(predictions[i], dim=0))[tokenized_sentence[i+1]].item() for i in range(len(tokenized_sentence)-1)])
    sent_prob = np.sum(prob_list)
    return sent_prob

#Load sentences
print("Loading sentences")
with open(PATH+'textfile/generated_pairs_xlnet.csv') as f:
    reader = csv.reader(f)
    file = [row for row in reader]
    head = file[0]
    corpus = file[1:]

sent_list = [[pair[head.index('DOsentence')],pair[head.index('PDsentence')]] for pair in corpus]


#Load the model
print("Loading the model")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large')
model.eval()

#Calculate probability
print("Calculating DO")
DO_prob = np.array([calc_sent_prob(sentence[0]) for sentence in sent_list])
print("Calculating PD")
PD_prob = np.array([calc_sent_prob(sentence[1]) for sentence in sent_list])
ratio = DO_prob - PD_prob

#Dump the data
print("Dumping data")
with open(PATH+'datafile/gpt2_large_DO_test.pkl','wb') as f:
    pickle.dump(DO_prob,f)
with open(PATH+'datafile/gpt2_large_PD_test.pkl','wb') as f:
    pickle.dump(PD_prob,f)
with open(PATH+'datafile/gpt2_large_log_ratio_test.pkl','wb') as f:
    pickle.dump(ratio,f)

with open(PATH+'textfile/generated_pairs_gpt2_large_test.csv','w') as f:
    writer = csv.writer(f)
    head.extend(['GPT2_large_ratio'])
    writer.writerow(head)
    for i,row in enumerate(corpus):
        row.extend([ratio[i]])
        writer.writerow(row)
