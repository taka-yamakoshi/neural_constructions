import pickle
import numpy as np
import torch
import torch.nn.functional as F
import csv
import logging
import model
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

with open(PATH+'datafile/hidden650_batch128_dropout0.2_lr20.0.pt', 'rb') as f:
    model = torch.load(f, map_location=torch.device('cpu'))
model.eval()

with open(PATH+'datafile/word2id.pkl', 'rb') as f:
    word2id = pickle.load(f)
with open(PATH+'datafile/id2word.pkl', 'rb') as f:
    id2word = pickle.load(f)


#Load sentences
with open(PATH+'textfile/generated_pairs_latest.csv') as f:
    reader = csv.reader(f)
    file = [row for row in reader]
    head = file[0]
    corpus = file[1:]
sent_list = [[pair[head.index('DOsentence')],pair[head.index('PDsentence')]] for pair in corpus]

sentence_data_DO = [torch.tensor([word2id[word] if word in word2id else word2id['<unk>'] for word in (['<eos>'] + sentences[0].split(" ")+['.'])]) for sentences in sent_list]
sentence_data_PD = [torch.tensor([word2id[word] if word in word2id else word2id['<unk>'] for word in (['<eos>'] + sentences[1].split(" ")+['.'])]) for sentences in sent_list]


def calculate_probs_shared_hidden(sentence_data):
    hid_state = [init_c0,init_h0,init_c1,init_h1]
    total_probs = []
    for i,sentence in enumerate(sentence_data):
        hid_state, probs =  my_model.forward(sentence,hid_state)
        total_probs.append(probs)
        if i%100 == 0:
            print(str(i) + " sentences done")
    return np.array(total_probs)

def calculate_sent_prob(sentence,init_state):
    prob_list = np.zeros((len(sentence)-1))
    hid_state = init_state
    for i, word in enumerate(sentence[:-1]):
        output,hid_state =  model(word.unsqueeze(0).unsqueeze(0),hid_state)
        log_probs = torch.log(F.softmax(output[0][0],dim=0))
        prob_list[i] = log_probs[sentence[i+1].item()]
    return np.sum(prob_list)

def calculate_probs_indiv(sentence_data):
    total_probs = np.zeros((len(sentence_data)))
    for i,sentence in enumerate(sentence_data):
        init_state = model.init_hidden(1)
        total_probs[i] = calculate_sent_prob(sentence,init_state)
        if i%100 == 0:
            print(str(i) + " sentences done")
    return total_probs

DO_prob = calculate_probs_indiv(sentence_data_DO[:10])
PD_prob = calculate_probs_indiv(sentence_data_PD[:10])
ratio = DO_prob - PD_prob
#Save the data
with open(PATH+'datafile/LSTM_DO_new_test.pkl','wb') as f:
    pickle.dump(DO_prob,f)
with open(PATH+'datafile/LSTM_PD_new_test.pkl','wb') as f:
    pickle.dump(PD_prob,f)
with open(PATH+'datafile/LSTM_log_ratio_new_test.pkl','wb') as f:
    pickle.dump(ratio,f)

with open(PATH+'textfile/generated_pairs_latest_test.csv','w') as f:
    writer = csv.writer(f)
    writer.writerow(head)
    for i,row in enumerate(corpus[:10]):
        row[head.index('LSTM_ratio')] = ratio[i]
        writer.writerow(row)
