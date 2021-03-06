from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
import csv
import pickle
import random
from multiprocessing import Pool
import sys

sys.path.append('..')
args = sys.argv

#Load the hidden states
with open(f'../data/hidden_states_{args[1]}_{args[2]}_DO.pkl','rb') as f:
    hidden_DO = pickle.load(f)
with open(f'../data/hidden_states_{args[1]}_{args[2]}_PD.pkl','rb') as f:
    hidden_PD = pickle.load(f)
num_layers = hidden_DO.shape[1]
hidden_dim = hidden_DO.shape[2]

#Load labels
with open('../data/generated_pairs_with_results.csv') as f:
    reader = csv.reader(f)
    file = [row for row in reader]
    head = file[0]
    text = file[1:]
label = np.array([[float(row[head.index('BehavDOpreference')])/100] for row in text])

#Select only sentences with pronoun recipients
hidden_DO_pronoun = np.zeros((1000,num_layers,hidden_dim))
for i in range(1000):
    hidden_DO_pronoun[i] = hidden_DO[5*i+2]
hidden_PD_pronoun = np.zeros((1000,num_layers,hidden_dim))
for i in range(1000):
    hidden_PD_pronoun[i] = hidden_PD[5*i+2]
label_pronoun = np.zeros((1000,1))
for i in range(1000):
    label_pronoun[i] = label[5*i+2]


#Separate the sentences for training and test: separated randomly for alternating/non-alternating verbs
random_seq_alt = np.random.permutation(np.arange(100))
random_seq_non_alt = np.random.permutation(np.arange(100))+100
train_verb_pick = list(random_seq_alt[:80]) + list(random_seq_non_alt[:80])
test_verb_pick = list(random_seq_alt[80:]) + list(random_seq_non_alt[80:])

train_DO = np.zeros((len(train_verb_pick)*5,num_layers,hidden_dim))
for i, id in enumerate(train_verb_pick):
    train_DO[5*i:5*(i+1)] = hidden_DO_pronoun[5*id:5*(id+1)]
train_PD = np.zeros((len(train_verb_pick)*5,num_layers,hidden_dim))
for i, id in enumerate(train_verb_pick):
    train_PD[5*i:5*(i+1)] = hidden_PD_pronoun[5*id:5*(id+1)]
train_label = np.zeros((len(train_verb_pick)*5,1))
for i, id in enumerate(train_verb_pick):
    train_label[5*i:5*(i+1)] = label_pronoun[5*id:5*(id+1)]

test_DO = np.zeros((len(test_verb_pick)*5,num_layers,hidden_dim))
for i, id in enumerate(test_verb_pick):
    test_DO[5*i:5*(i+1)] = hidden_DO_pronoun[5*id:5*(id+1)]
test_PD = np.zeros((len(test_verb_pick)*5,num_layers,hidden_dim))
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
performance = np.zeros(num_layers)
alpha_list = np.zeros(num_layers)
for layer_num in range(num_layers):
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
with open(f'../data/test_performance_{args[1]}_{args[2]}_{args[3]}.pkl','wb') as f:
    pickle.dump(performance,f)
with open(f'../data/alpha_{args[1]}_{args[2]}_{arg[3]}.pkl','wb') as f:
    pickle.dump(alpha_list,f)
