import pickle
import numpy as np
import torch
import csv
import logging
import model
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
args = sys.argv
##First argument: comparison categery (recipient or theme)
##Second argument: recipient/theme type
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

sentence_data_DO_alt = [torch.tensor([word2id[word] if word in word2id else word2id['<unk>'] for word in sentence[0].split(" ")]) for sentence in sent_list_alt]
sentence_data_PD_alt = [torch.tensor([word2id[word] if word in word2id else word2id['<unk>'] for word in sentence[1].split(" ")]) for sentence in sent_list_alt]
sentence_data_DO_nonalt = [torch.tensor([word2id[word] if word in word2id else word2id['<unk>'] for word in sentence[0].split(" ")]) for sentence in sent_list_nonalt]
sentence_data_PD_nonalt = [torch.tensor([word2id[word] if word in word2id else word2id['<unk>'] for word in sentence[1].split(" ")]) for sentence in sent_list_nonalt]

#LSTM model
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
                prob = torch.log(softmax(outvec))
                probs += prob[sentence[i+1]].item()
        return hid_state,probs


my_model = trained_model(W_in, W_x0, W_h0, b_x0, b_h0, W_x1, W_h1, b_x1, b_h1, W_out, b_out, word2id,id2word)

init_c0 = torch.zeros(650)
init_h0 = torch.zeros(650)
init_c1 = torch.zeros(650)
init_h1 = torch.zeros(650)

def calculate_probs(sentence_data):
    hid_state = [init_c0,init_h0,init_c1,init_h1]
    total_probs = []
    seed = 2020
    np.random.seed(seed)
    random_seq = np.random.permutation(len(sentence_data))
    random_sentence_data = [sentence_data[random_seq[j]] for j in range(len(random_seq))]
    for i,sentence in enumerate(random_sentence_data):
        hid_state, probs =  my_model.forward(sentence,hid_state)
        total_probs.append(probs)
        if i%100 == 0:
            print(str(i) + " sentences done")
    return total_probs

DO_alt_log_prob = calculate_probs(sentence_data_DO_alt)
PD_alt_log_prob = calculate_probs(sentence_data_PD_alt)
DO_nonalt_log_prob = calculate_probs(sentence_data_DO_nonalt)
PD_nonalt_log_prob = calculate_probs(sentence_data_PD_nonalt)
alt_log_ratio = np.array(DO_alt_log_prob) - np.array(PD_alt_log_prob)
nonalt_log_ratio = np.array(DO_nonalt_log_prob) - np.array(PD_nonalt_log_prob)
#Save the data
with open(PATH + 'datafile/lstm_log_ratio_'+args[2]+'.pkl','wb') as f:
    pickle.dump([alt_log_ratio,nonalt_log_ratio],f)
