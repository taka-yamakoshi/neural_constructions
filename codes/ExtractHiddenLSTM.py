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
    model = torch.load(f,map_location=torch.device('cpu'))
model.eval()

W_in = model.state_dict()['encoder.weight']
W_x0 = model.state_dict()['rnn.weight_ih_l0']
W_h0 = model.state_dict()['rnn.weight_hh_l0']
b_x0 = model.state_dict()['rnn.bias_ih_l0']
b_h0 = model.state_dict()['rnn.bias_hh_l0']
W_x1 = model.state_dict()['rnn.weight_ih_l1']
W_h1 = model.state_dict()['rnn.weight_hh_l1']
b_x1 = model.state_dict()['rnn.bias_ih_l1']
b_h1 = model.state_dict()['rnn.bias_hh_l1']
W_out = model.state_dict()['decoder.weight']
b_out = model.state_dict()['decoder.bias']

def sigmoid(x):
    return 1/(1+np.exp(-x))
def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x))

with open(PATH+'datafile/word2id.pkl', 'rb') as f:
    word2id = pickle.load(f)
with open(PATH+'datafile/id2word.pkl', 'rb') as f:
    id2word = pickle.load(f)

#Load sentences
print("Loading sentences")
with open(PATH+'textfile/generated_pairs_latest.csv') as f:
    reader = csv.reader(f)
    file = [row for row in reader]
    head = file[0]
    corpus = file[1:]
sent_list = [[row[head.index('DOsentence')],row[head.index('PDsentence')]] for row in corpus]
verb_list = [row[head.index('DOsentence')].split(" ")[1] for row in corpus]
recipient_list  = [row[head.index('PDsentence')].split(" ")[-1] for row in corpus]
theme_list = [row[head.index('PDsentence')].split(" ")[row[head.index('PDsentence')].split(" ").index("to")-1] for row in corpus]


sentence_data_DO = [torch.tensor([word2id[word] if word in word2id else word2id['<unk>'] for word in (['<eos>'] + sentences[0].split(" ")+['.'])]) for sentences in sent_list]
sentence_data_PD = [torch.tensor([word2id[word] if word in word2id else word2id['<unk>'] for word in (['<eos>'] + sentences[1].split(" ")+['.'])]) for sentences in sent_list]


#Pretrained LSTM model
class trained_model():
    def __init__(self,W_in, W_x0, W_h0, b_x0, b_h0, W_x1, W_h1, b_x1, b_h1, W_out, b_out, word2id,id2word):
        return None
    def embed(self,in_word,W_in):
        return W_in[in_word]
    def myLSTM(self,in_word,hid_state):
        c0 = hid_state[0]
        h0 = hid_state[1]
        c1 = hid_state[2]
        h1 = hid_state[3]
        
        Xm0 = torch.mv(W_x0,self.embed(in_word,W_in)) + b_x0
        Hm0 = torch.mv(W_h0,h0) + b_h0
        
        i0 = sigmoid(Xm0[:650] + Hm0[:650])
        f0 = sigmoid(Xm0[650:1300] + Hm0[650:1300])
        g0 = np.tanh(Xm0[1300:1950] + Hm0[1300:1950])
        o0 = sigmoid(Xm0[1950:] + Hm0[1950:])
        new_c0 = c0*f0 + g0*i0
        mid_c0 = c0*f0
        add_c0 = g0*i0
        new_h0 = np.tanh(new_c0)*o0
        
        Xm1 = torch.mv(W_x1,new_h0) + b_x1
        Hm1 = torch.mv(W_h1,h1) + b_h1
        
        i1 = sigmoid(Xm1[:650] + Hm1[:650])
        f1 = sigmoid(Xm1[650:1300] + Hm1[650:1300])
        g1 = np.tanh(Xm1[1300:1950] + Hm1[1300:1950])
        o1 = sigmoid(Xm1[1950:] + Hm1[1950:])
        new_c1 = c1*f1 + g1*i1
        mid_c1 = c1*f1
        add_c1 = g1*i1
        new_h1 = np.tanh(new_c1)*o1
        
        outvec = torch.mv(W_out,new_h1) + b_out
        return outvec, [new_c0,new_h0,new_c1,new_h1]
    
    def forward(self,sentence,init_hid_state):
        hid_state = init_hid_state
        probs = 0
        for i, word in enumerate(sentence):
            outvec, hid_state = self.myLSTM(word,hid_state)
            if i <= len(sentence)-2:
                prob = torch.log(F.softmax(outvec, dim=0))
                probs += prob[sentence[i+1]].item()
        return hid_state,probs
    def forward_with_hidden_output(self,sentence,init_hid_state,verb,first_obj):
        if verb in word2id:
            verb_id = word2id[verb]
        else:
            verb_id  = word2id['<unk>']
        if first_obj in word2id:
            first_obj_id = word2id[first_obj]
        else:
            first_obj_id  = word2id['<unk>']
        
        hid_state = init_hid_state
        verb_flag = True
        for i, word in enumerate(sentence):
            outvec, hid_state = self.myLSTM(word,hid_state)
            if word == verb_id and verb_flag:
                verb_flag = False
                hidden_verb = np.array([list(state) for state in hid_state])
            if word == first_obj_id:
                hidden_first_obj = np.array([list(state) for state in hid_state])
            if i == len(sentence)-1:
                hidden_eos = np.array([list(state) for state in hid_state])
        return hidden_verb,hidden_first_obj,hidden_eos


my_model = trained_model(W_in, W_x0, W_h0, b_x0, b_h0, W_x1, W_h1, b_x1, b_h1, W_out, b_out, word2id,id2word)

init_c0 = torch.zeros(650)
init_h0 = torch.zeros(650)
init_c1 = torch.zeros(650)
init_h1 = torch.zeros(650)
init_state = [init_c0,init_h0,init_c1,init_h1]

def calculate_probs(sentence_data):
    hid_state = [init_c0,init_h0,init_c1,init_h1]
    total_probs = []
    for i,sentence in enumerate(sentence_data):
        hid_state, probs =  my_model.forward(sentence,hid_state)
        total_probs.append(probs)
        if i%100 == 0:
            print(str(i) + " sentences done")
    return np.array(total_probs)

def calculate_probs_indiv(sentence_data):
    total_probs = []
    for i,sentence in enumerate(sentence_data):
        hid_state, probs =  my_model.forward(sentence,init_state)
        total_probs.append(probs)
        if i%100 == 0:
            print(str(i) + " sentences done")
    return np.array(total_probs)
def extract_hidden(sentence_data,verb_list,first_obj_list):
    hidden = np.zeros((5000,4,650))
    for i,sentence in enumerate(sentence_data):
        if args[1] == 'verb':
            hidden[i],_,_ = my_model.forward_with_hidden_output(sentence,init_state,verb_list[i],first_obj_list[i])
        elif args[1] == 'first_obj':
            _,hidden[i],_ = my_model.forward_with_hidden_output(sentence,init_state,verb_list[i],first_obj_list[i])
        elif args[1] == 'eos':
            _,_,hidden[i] = my_model.forward_with_hidden_output(sentence,init_state,verb_list[i],first_obj_list[i])
        if i%100 == 0:
            print(str(i)+" sentences done")
    return hidden
hidden_states = np.zeros((2,5000,4,650))
hidden_states[0] = extract_hidden(sentence_data_DO,verb_list,recipient_list)
hidden_states[1] = extract_hidden(sentence_data_PD,verb_list,theme_list)

#Save the data
with open(PATH+'datafile/hidden_states_LSTM_'+args[1]+'.pkl','wb') as f:
    pickle.dump(hidden_states,f)
