import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
import numpy as np
import csv
import pickle
import random
from multiprocessing import Pool
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()
#Load the hidden states
with open('../datafile/hidden_states_dais_gpt2_'+args[1]+'.pkl','rb') as f:
    hidden_states = pickle.load(f)

hidden_DO = hidden_states[0]
hidden_PD = hidden_states[1]

del hidden_states
#Load labels
with open('../textfile/generated_pairs_latest.csv') as f:
    reader = csv.reader(f)
    file = [row for row in reader]
    head = file[0]
    corpus = file[1:]
label = np.array([[float(row[head.index('BehavDOpreference')])/100] for row in corpus])
hidden_DO_pronoun = np.zeros((1000,13,768))
for i in range(1000):
    hidden_DO_pronoun[i] = hidden_DO[5*i+2]
hidden_PD_pronoun = np.zeros((1000,13,768))
for i in range(1000):
    hidden_PD_pronoun[i] = hidden_PD[5*i+2]
label_pronoun = np.zeros((1000,1))
for i in range(1000):
    label_pronoun[i] = label[5*i+2]


#Separate the sentences for training and test
random_seq_alt = np.random.permutation(np.arange(100))
random_seq_non_alt = np.random.permutation(np.arange(100))+100
train_verb_pick = list(random_seq_alt[:80]) + list(random_seq_non_alt[:80])
test_verb_pick = list(random_seq_alt[80:]) + list(random_seq_non_alt[80:])

train_DO = np.zeros((len(train_verb_pick)*5,13,768))
for i, id in enumerate(train_verb_pick):
    train_DO[5*i:5*(i+1)] = hidden_DO_pronoun[5*id:5*(id+1)]
train_PD = np.zeros((len(train_verb_pick)*5,13,768))
for i, id in enumerate(train_verb_pick):
    train_PD[5*i:5*(i+1)] = hidden_PD_pronoun[5*id:5*(id+1)]
train_label = np.zeros((len(train_verb_pick)*5,1))
for i, id in enumerate(train_verb_pick):
    train_label[5*i:5*(i+1)] = label_pronoun[5*id:5*(id+1)]

test_DO = np.zeros((len(test_verb_pick)*5,13,768))
for i, id in enumerate(test_verb_pick):
    test_DO[5*i:5*(i+1)] = hidden_DO_pronoun[5*id:5*(id+1)]
test_PD = np.zeros((len(test_verb_pick)*5,13,768))
for i, id in enumerate(test_verb_pick):
    test_PD[5*i:5*(i+1)] = hidden_PD_pronoun[5*id:5*(id+1)]
test_label = np.zeros((len(test_verb_pick)*5,1))
for i, id in enumerate(test_verb_pick):
    test_label[5*i:5*(i+1)] = label_pronoun[5*id:5*(id+1)]


def model_run(train_input,val_input,alpha):
    DO_data,PD_data,train_label = train_input
    train_data = np.array([list(DO)+list(PD) for DO, PD in zip(DO_data,PD_data)])
    clf = Ridge(alpha)
    clf.fit(train_data,train_label)
    DO_val_data,PD_val_data,test_label = val_input
    test_data = np.array([list(DO)+list(PD) for DO, PD in zip(DO_val_data,PD_val_data)])
    return clf.score(test_data,test_label)


#Train the classfier
performance = np.zeros(13)
alpha_list = np.zeros(13)
for layer_num in range(13):
    DO_data = np.array([train_DO[sent_num][layer_num] for sent_num in range(len(train_DO))])
    PD_data = np.array([train_PD[sent_num][layer_num] for sent_num in range(len(train_PD))])
    train_input = (DO_data,PD_data,train_label)
    
    DO_val_data = np.array([test_DO[sent_num][layer_num] for sent_num in range(len(test_DO))])
    PD_val_data = np.array([test_PD[sent_num][layer_num] for sent_num in range(len(test_PD))])
    val_input = (DO_val_data,PD_val_data,test_label)
    grid_range = np.array([5**i for i in range(11)])
    arg = [(train_input,val_input,alpha) for alpha in grid_range]
    with Pool(processes=8) as p:
        result_list = p.starmap(model_run,arg)

    performance[layer_num] = np.max(result_list)
    alpha_list[layer_num] = grid_range[np.argmax(result_list)]
    print(alpha_list[layer_num],performance[layer_num])


#Dump the train performance and parameters
with open('../datafile/test_performance_dais_gpt2_behav_pronoun_'+args[1]+'_'+args[2]+'.pkl','wb') as f:
    pickle.dump(performance,f)
#with open('../datafile/alpha_dais_gpt2_behav_pronoun_'+args[1]+'_'+args[2]+'.pkl','wb') as f:
#pickle.dump(alpha_list,f)
