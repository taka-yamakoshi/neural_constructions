import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import csv
import logging
from transformers import XLNetTokenizer, XLNetLMHeadModel
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

def calc_prob(tokenized_sentence, timestep):
    input_sentence = tokenized_sentence[:timestep]
    input_sentence.append(mask_id)
    input_ids = torch.tensor(input_sentence).unsqueeze(0)
    perm_mask = torch.zeros((1, (timestep+1), (timestep+1)), dtype=torch.float)
    perm_mask[:, :, -1] = 1.0
    target_mapping = torch.zeros((1, 1, (timestep+1)), dtype=torch.float)
    target_mapping[0, 0, -1] = 1.0
    outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
    predictions = outputs[0][0][0]
    log_probs = torch.log(F.softmax(predictions, dim=0))
    return log_probs[tokenized_sentence[timestep]].item()

def calc_sent_prob(sentence):
    words = sentence.split(" ")
    words.append(".")
    tokenized_sentence = tokenizer.encode(" ".join(words), add_special_tokens=False)
    prob_list = np.array([calc_prob(tokenized_sentence, timestep) for timestep in range(1,len(tokenized_sentence))])
    return np.sum(prob_list)

#Load sentences
print("Loading sentences")
with open(PATH+'textfile/generated_pairs_new_LSTM.csv') as f:
    reader = csv.reader(f)
    file = [row for row in reader]
    head = file[0]
    corpus = file[1:]
sent_list = [[pair[head.index('DOsentence')],pair[head.index('PDsentence')]] for pair in corpus]


#Load the model
print("Loading the model")
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased')
model.eval()

#Calculate probability
mask_id = tokenizer.encode("<mask>",add_special_tokens=False)[0]
print("Calculating DO")
DO_prob = np.array([calc_sent_prob(sentence[0]) for sentence in sent_list[:30]])
print("Calculating PD")
PD_prob = np.array([calc_sent_prob(sentence[1]) for sentence in sent_list[:30]])
ratio = DO_prob - PD_prob

#Dump the data
print("Dumping data")
with open(PATH+'datafile/xlnet_DO_test.pkl','wb') as f:
    pickle.dump(DO_prob,f)
with open(PATH+'datafile/xlnet_PD_test.pkl','wb') as f:
    pickle.dump(PD_prob,f)
with open(PATH+'datafile/xlnet_log_ratio_test.pkl','wb') as f:
    pickle.dump(ratio,f)

with open(PATH+'textfile/generated_pairs_xlnet_test.csv','w') as f:
    writer = csv.writer(f)
    head.extend(['XLNet_ratio_new'])
    writer.writerow(head)
    for i,row in enumerate(corpus[:30]):
        row.extend([ratio[i]])
        writer.writerow(row)
