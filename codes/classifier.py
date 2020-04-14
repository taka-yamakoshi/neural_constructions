import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x))
def batchfy(train_data,batch_num):
    batch_size = int(len(train_data)/batch_num)
    return [train_data[batch_size*i:batch_size*(i+1)] for i in range(batch_num)]

#Load the model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

#Load the hidden states
with open(PATH + 'datafile/hidden_states_DO.pkl','rb') as f:
    hidden_DO = pickle.load(f)
with open(PATH + 'datafile/hidden_states_PD.pkl','rb') as f:
    hidden_PD = pickle.load(f)

verb_list = ["showed", "told", "guaranteed", "lent", "offered", "loaned", "left", "promised", "slipped", "wrote", "taught", "gave", "fed", "paid", "voted", "handed", "served", "tossed", "sent", "sold"]


#Classifier
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.layer(x)
        return torch.tanh(x)

#Randomly separate the sentences for training and test
random_pick = random.sample([i for i in range(20)],15)
train_DO = []
test_DO = []
train_PD = []
test_PD = []
for id in random_pick:
    train_DO.extend(hidden_DO[10*id:10*(id+1)])
    train_PD.extend(hidden_PD[10*id:10*(id+1)])

for id in range(len(verb_list)):
    if id not in random_pick:
        test_DO.extend(hidden_DO[10*id:10*(id+1)])
        test_PD.extend(hidden_PD[10*id:10*(id+1)])


#Train the classfier
batch_num = 10
performance = np.zeros((13,12))
params = {}
for layer_num in range(13):
    param = {}
    for head_num in range(12):
        model = Net().double()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        DO_data = [train_DO[sent_num][layer_num][64*head_num:64*(head_num+1)].double() for sent_num in range(150)]
        PD_data = [train_PD[sent_num][layer_num][64*head_num:64*(head_num+1)].double() for sent_num in range(150)]
        train_data = DO_data.copy()
        train_data.extend(PD_data)
        DO_label = [torch.tensor([1]).double() for i in range(150)]
        PD_label = [torch.tensor([-1]).double() for i in range(150)]
        train_label = DO_label.copy()
        train_label.extend(PD_label)
        new_performance = 0
        prev_performance = 0
        epoch = 0
        while new_performance > prev_performance-0.05 and epoch < 50:
            random_seq = np.random.permutation(np.arange(len(train_data)))
            shuffled_x_data = [train_data[random_id] for random_id in random_seq]
            shuffled_t_data = [train_label[random_id] for random_id in random_seq]
            x_train = shuffled_x_data[:250]
            x_test = shuffled_x_data[250:]
            t_train = shuffled_t_data[:250]
            t_test = shuffled_t_data[250:]
            x_batch = batchfy(x_train,batch_num)
            t_batch = batchfy(t_train,batch_num)
            total_loss = 0
            for i in range(batch_num):
                for x,t in zip(x_batch[i],t_batch[i]):
                    x, t = Variable(x), Variable(t)
                    optimizer.zero_grad()
                    output = model(x)
                    loss = (output-t)**2
                    loss.backward()
                    optimizer.step()
                    total_loss+= loss.item()
            j = 0
            k = 0
            for x, t in zip(x_test,t_test):
                x, t = Variable(x), Variable(t)
                optimizer.zero_grad()
                output = model(x)
                if output >0 and t ==1:
                    j+=1
                elif output <0 and t==-1:
                    j+=1
                k+=1
            new_performance = j/k
            prev_performance = np.max(np.array([new_performance,prev_performance]))
            epoch += 1
        param["head_" + str(head_num)]= model.state_dict()
        print(new_performance)
        performance[layer_num][head_num] = new_performance
    params["layer_"+str(layer_num)] = param


#Dump the train performance and parameters
with open(PATH + 'datafile/train_performance.pkl','wb') as f:
    pickle.dump(performance,f)
with open(PATH + 'datafile/params.pkl','wb') as f:
    pickle.dump(params,f)

#Calculate performance on the test set
test_performance = np.zeros((13,12))
for layer_num in range(13):
    for head_num in range(12):
        DO_data = [test_DO[sent_num][layer_num][64*head_num:64*(head_num+1)].double() for sent_num in range(50)]
        PD_data = [test_PD[sent_num][layer_num][64*head_num:64*(head_num+1)].double() for sent_num in range(50)]
        stored_param = params["layer_"+str(layer_num)]["head_" + str(head_num)]
        weight = stored_param['layer.weight'][0]
        bias = stored_param['layer.bias'][0]
        DO_performance = np.array([(torch.dot(weight,DO_vec)+bias).item() for DO_vec in DO_data]) > 0
        PD_performance = np.array([(torch.dot(weight,PD_vec)+bias).item() for PD_vec in PD_data]) < 0
        test_performance[layer_num][head_num] = (np.sum(DO_performance)+np.sum(PD_performance))/100

#Dump the test performance
with open(PATH + 'datafile/test_performance.pkl','wb') as f:
    pickle.dump(test_performance,f)
