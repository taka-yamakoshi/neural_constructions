import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
import pickle
import matplotlib.pyplot as plt
import random
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

def batchfy(train_data,batch_size):
    batch_num = int(len(train_data)/batch_size)
    return [torch.tensor(list(train_data[batch_size*i:batch_size*(i+1)])) for i in range(batch_num)]


#Load the hidden states
with open(PATH + 'datafile/hidden_states_dais_gpt2_'+args[1]+'.pkl','rb') as f:
    hidden_states = pickle.load(f)

hidden_DO = hidden_states[0]
hidden_PD = hidden_states[1]

del hidden_states

#Separate the sentences for training and test
seed = 1999
random_seq_alt = np.random.permutation(np.arange(100))
seed = 2019
random_seq_non_alt = np.random.permutation(np.arange(100))+100
train_verb_pick = list(random_seq_alt[:20]) + list(random_seq_non_alt[:20])
test_verb_pick = list(random_seq_alt[20:]) + list(random_seq_non_alt[20:])

DO_train_verb = np.zeros((len(train_verb_pick)*25,13,768))
for i, id in enumerate(train_verb_pick):
    DO_train_verb[25*i:25*(i+1)] = hidden_DO[25*id:25*(id+1)]
DO_pronoun_train_verb = [DO_train_verb[5*id+2] for id in range(200)]
DO_shortDefinite_train_verb = [DO_train_verb[5*id+3] for id in range(200)]
train_DO = np.array(DO_pronoun_train_verb + DO_shortDefinite_train_verb)
del DO_pronoun_train_verb, DO_shortDefinite_train_verb
DO_longDefinite_train_verb = [DO_train_verb[5*id] for id in range(200)]
DO_longIndefinite_train_verb = [DO_train_verb[5*id+1] for id in range(200)]
DO_shortIndefinite_train_verb = [DO_train_verb[5*id+4] for id in range(200)]
test_DO_recipient = np.array(DO_longDefinite_train_verb + DO_longIndefinite_train_verb + DO_shortIndefinite_train_verb)
del DO_longDefinite_train_verb, DO_longIndefinite_train_verb, DO_shortIndefinite_train_verb
del DO_train_verb

PD_train_verb = np.zeros((len(train_verb_pick)*25,13,768))
for i, id in enumerate(train_verb_pick):
    PD_train_verb[25*i:25*(i+1)] = hidden_PD[25*id:25*(id+1)]
PD_pronoun_train_verb = [PD_train_verb[5*id+2] for id in range(200)]
PD_shortDefinite_train_verb = [PD_train_verb[5*id+3] for id in range(200)]
train_PD = np.array(PD_pronoun_train_verb + PD_shortDefinite_train_verb)
del PD_pronoun_train_verb, PD_shortDefinite_train_verb
PD_longDefinite_train_verb = [PD_train_verb[5*id] for id in range(200)]
PD_longIndefinite_train_verb = [PD_train_verb[5*id+1] for id in range(200)]
PD_shortIndefinite_train_verb = [PD_train_verb[5*id+4] for id in range(200)]
test_PD_recipient = np.array(PD_longDefinite_train_verb + PD_longIndefinite_train_verb + PD_shortIndefinite_train_verb)
del PD_longDefinite_train_verb, PD_longIndefinite_train_verb, PD_shortIndefinite_train_verb
del PD_train_verb

test_DO_verb = np.zeros((len(test_verb_pick)*25,13,768))
for i, id in enumerate(test_verb_pick):
    test_DO_verb[25*i:25*(i+1)] = hidden_DO[25*id:25*(id+1)]
test_PD_verb = np.zeros((len(test_verb_pick)*25,13,768))
for i, id in enumerate(test_verb_pick):
    test_PD_verb[25*i:25*(i+1)] = hidden_PD[25*id:25*(id+1)]

print("Number of sentences for training: " + str(len(train_DO)+len(train_PD)))
print("Number of sentences for recipient validation: " + str(len(test_DO_recipient)+len(test_PD_recipient)))
print("Number of sentences for verb validation: " + str(len(test_DO_verb)+len(test_PD_verb)))

with open(PATH + 'datafile/params_dais_gpt2_'+args[1]+'.pkl','rb') as f:
    params = pickle.load(f)

#Calculate performance on the test set
print("Validating")
test_performance_recipient = np.zeros((13,12))
print("Recipient Variation")
for layer_num in range(13):
    for head_num in range(12):
        DO_data = np.array([test_DO_recipient[sent_num][layer_num][64*head_num:64*(head_num+1)] for sent_num in range(len(test_DO_recipient))])
        PD_data = np.array([test_PD_recipient[sent_num][layer_num][64*head_num:64*(head_num+1)] for sent_num in range(len(test_PD_recipient))])
        stored_param = params[layer_num][head_num]
        weight = stored_param['layer.weight'][0]
        bias = stored_param['layer.bias'][0]
        DO_performance = np.array([(torch.dot(weight,torch.tensor(list(DO_vec)))+bias).item() for DO_vec in DO_data]) > 0
        PD_performance = np.array([(torch.dot(weight,torch.tensor(list(PD_vec)))+bias).item() for PD_vec in PD_data]) < 0
        test_performance_recipient[layer_num][head_num] = (np.sum(DO_performance)+np.sum(PD_performance))/(len(DO_data)+len(PD_data))
        print("layer_num: " + str(layer_num) + " head_num: " + str(head_num) + " " + str(test_performance_recipient[layer_num][head_num]))

#Dump the test performance
with open(PATH + 'datafile/test_performance_recipient_dais_'+args[1]+'.pkl','wb') as f:
    pickle.dump(test_performance_recipient,f)


test_performance_verb = np.zeros((13,12))
print("Verb Variation")
for layer_num in range(13):
    for head_num in range(12):
        DO_data = np.array([test_DO_verb[sent_num][layer_num][64*head_num:64*(head_num+1)] for sent_num in range(len(test_DO_verb))])
        PD_data = np.array([test_PD_verb[sent_num][layer_num][64*head_num:64*(head_num+1)] for sent_num in range(len(test_PD_verb))])
        stored_param = params[layer_num][head_num]
        weight = stored_param['layer.weight'][0]
        bias = stored_param['layer.bias'][0]
        DO_performance = np.array([(torch.dot(weight,torch.tensor(list(DO_vec)))+bias).item() for DO_vec in DO_data]) > 0
        PD_performance = np.array([(torch.dot(weight,torch.tensor(list(PD_vec)))+bias).item() for PD_vec in PD_data]) < 0
        test_performance_verb[layer_num][head_num] = (np.sum(DO_performance)+np.sum(PD_performance))/(len(DO_data)+len(PD_data))
        print("layer_num: " + str(layer_num) + " head_num: " + str(head_num) + " " + str(test_performance_verb[layer_num][head_num]))

#Dump the test performance
with open(PATH + 'datafile/test_performance_verb_dais_'+args[1]+'.pkl','wb') as f:
    pickle.dump(test_performance_verb,f)
