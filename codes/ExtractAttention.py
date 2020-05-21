import pickle
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import sys
import logging

sys.path.append('..')
args = sys.argv
##First argument: construction type (DO or PD)
##Second arguemnt: recipient type (shortDefinite only at this moment)
##Third argument: default sentence length (8 for DO, 9 for PD)
with open("mypath.txt") as f:
    PATH = f.read()

def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x))

def shrink_attention(sent_type,theme_num,raw_attn):
    if sent_type == "DO":
        added_attn = np.zeros((12,len(tokenized_sentence),int(args[3])))
        added_attn[:,:,:5] = raw_attn[:,:,:5]
        added_attn[:,:,-2:] = raw_attn[:,:,-2:]
        added_attn[:,:,5] = np.sum(raw_attn[:,:,5:(5+theme_num)],axis=2)
        shrunk_attn = np.zeros((12,int(args[3]),int(args[3])))
        shrunk_attn[:,:5,:] = added_attn[:,:5,:]
        shrunk_attn[:,-2:,:] = added_attn[:,-2:,:]
        shrunk_attn[:,5,:] = np.average(added_attn[:,5:(5+theme_num),:],axis=1)
    elif sent_type == "PD":
        added_attn = np.zeros((12,len(tokenized_sentence),int(args[3])))
        added_attn[:,:,:3] = raw_attn[:,:,:3]
        added_attn[:,:,-5:] = raw_attn[:,:,-5:]
        added_attn[:,:,3] = np.sum(raw_attn[:,:,3:(3+theme_num)],axis=2)
        shrunk_attn = np.zeros((12,int(args[3]),int(args[3])))
        shrunk_attn[:,:3,:] = added_attn[:,:3,:]
        shrunk_attn[:,-5:,:] = added_attn[:,-5:,:]
        shrunk_attn[:,3,:] = np.average(added_attn[:,3:(3+theme_num),:],axis=1)
    else:
        logging.error('Invalid construction type')
    return shrunk_attn

def extract_theme_attn(sent_type,attn):


#Load sentences
with open(PATH+'textfile/generated_pairs.csv') as f:
    reader = csv.reader(f)
    file = [row for row in reader]
    head = file[0]
    corpus = file[1:]
sent_list = [[pair[head.index('DOsentence')],pair[head.index('PDsentence')]] for pair in corpus if pair[head.index('recipient_id')] == args[2]]
verb_list = [corpus[25*i][head.index('DOsentence')].split(" ")[1] for i in range(200)]

#Load the model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

const_type = ["DO","PD"]

##Only works for shortDefinite recipients##
attention_list = []
mask_id = tokenizer.encode("[MASK]")[1:-1][0]
for sentence in sent_list:
    words = sentence[const_type.index(args[1])].split(" ")
    words.append(".")
    words[1] = "[MASK]"
    tokenized_sentence = tokenizer.encode(" ".join(words))
    input_tensor = torch.tensor([tokenized_sentence])
    with torch.no_grad():
        outputs = model(input_tensor)
        attention = []
        ##shrink the attention map as appropriate for each condition##
        if len(tokenized_sentence) == int(args[3]):
            for layer in outputs[2]:
                attention.append(list(layer[0].numpy()))
            attention_list.append(np.array(attention))
        elif len(tokenized_sentence) > int(args[3]):
            theme_num = len(tokenized_sentence) - int(args[3]) + 1
            for layer in outputs[2]:
                raw_attn = layer[0].numpy()
                shrunk_attn = shrink_attention(args[1],theme_num,raw_attn)
                attention.append(list(shrunk_attn))
            attention_list.append(np.array(attention))
ave_attention = np.average(np.array(attention_list),axis = 0)
with open(PATH+'datafile/attention_map_'+args[1]+'_'+args[2]+'.pkl','wb') as f:
    pickle.dump(ave_attention,f)
fig, axis = plt.subplots(12,12)
for i in range(12):
    for j in range(12):
        axis[i,j].imshow(ave_attention[i][j])
plt.show()
