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
    return [torch.tensor(list(train_data[batch_size*i:batch_size*(i+1)]),dtype=torch.float) for i in range(batch_num)]


#Load the hidden states
with open(PATH + 'datafile/hidden_states_dais_gpt2_'+args[1]+'.pkl','rb') as f:
    hidden_states = pickle.load(f)

hidden_DO = hidden_states[0]
hidden_PD = hidden_states[1]

del hidden_states

#Classifier
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.layer(x)
        activation = nn.Tanh()
        return activation(x)

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

#Train the classfier
batch_size = 40
performance = np.zeros((13,12))
params = {}
for layer_num in range(13):
    param = {}
    for head_num in range(12):
        model = Net()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        train_data = np.array([list(train_DO[sent_num][layer_num][64*head_num:64*(head_num+1)]) for sent_num in range(len(train_DO))] + [list(train_PD[sent_num][layer_num][64*head_num:64*(head_num+1)]) for sent_num in range(len(train_PD))])
        train_label = np.array([[1.0] for i in range(len(train_DO))] + [[-1.0] for i in range(len(train_PD))])
        prev_test_loss = 400
        epoch = 0
        random_seq = np.random.permutation(np.arange(len(train_data)))
        x_train= np.array([train_data[random_id] for random_id in random_seq[:600]])
        x_test= Variable(torch.tensor([list(train_data[random_id]) for random_id in random_seq[600:]]))
        t_train = np.array([train_label[random_id] for random_id in random_seq[:600]])
        t_test = Variable(torch.tensor([list(train_label[random_id]) for random_id in random_seq[600:]]))
        while epoch < 50:
            random_seq = np.random.permutation(np.arange(len(x_train)))
            x_batch = batchfy([x_train[random_id] for random_id in random_seq], batch_size)
            t_batch = batchfy([t_train[random_id] for random_id in random_seq], batch_size)
            for x, t in zip(x_batch,t_batch):
                x, t = Variable(x), Variable(t)
                optimizer.zero_grad()
                output = model(x)
                loss = loss_fn(output,t)
                loss.backward()
                optimizer.step()
            ##Test for early stopping
            optimizer.zero_grad()
            output = model(x_test)
            test_loss = loss_fn(output,t_test)
            test_loss = test_loss.item()
            prev_test_loss = np.min(np.array([test_loss,prev_test_loss]))
            epoch += 1
            if test_loss/prev_test_loss > 1.1 and epoch > 30:
                break
        new_performance = torch.sum(output*t_test > 0).item()/len(t_test)
        param[head_num]= model.state_dict()
        print("layer_num: " + str(layer_num) + " head_num: " + str(head_num) +  " epoch: " + str(epoch) + "  " + str(new_performance))
        performance[layer_num][head_num] = new_performance
    params[layer_num] = param


#Dump the train performance and parameters
print("Dumping the train performance and weights")
with open(PATH + 'datafile/train_performance_dais_gpt2_'+args[1]+'.pkl','wb') as f:
    pickle.dump(performance,f)
with open(PATH + 'datafile/params_dais_gpt2_'+args[1]+'.pkl','wb') as f:
    pickle.dump(params,f)

