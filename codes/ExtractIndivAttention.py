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
with open("mypath.txt") as f:
    PATH = f.read()

def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x))

##Only works for shortDefinite
def extract_theme(sent_type,recipient_type,sent,attn):
    verb_attn = attn[2]
    if sent_type == "DO":
        if sent[4] in tokenized_recipient:
            return np.sum(verb_attn[5:-2])
        else:
            logging.error('This function does not work')
            print(tokenizer.decode(sent))
            exit()
    elif sent_type == "PD":
        if sent[-5] == tokenizer.encode("to")[1:-1][0]:
            return np.sum(verb_attn[3:-5])
        else:
            logging.error('This function does not work')
            print(tokenizer.decode(sent))
            exit()
    else:
        logging.error('Invalid construction type')
        exit()

def extract_recipient(sent_type,theme_num,sent,attn):
    verb_attn = attn[2]
    if sent_type == "DO":
        if sent[4] in tokenized_recipient:
            return np.sum(verb_attn[3:5])
        else:
            logging.error('This function does not work')
            print(tokenizer.decode(sent))
            exit()
    elif sent_type == "PD":
        if sent[-5] == tokenizer.encode("to")[1:-1][0]:
            return np.sum(verb_attn[-4:-2])
        else:
            logging.error('This function does not work')
            print(tokenizer.decode(sent))
            exit()
    else:
        logging.error('Invalid construction type')
        exit()

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
recipient_list = ["man", "woman","team"]
tokenized_recipient = [tokenizer.encode(recipient)[1:-1][0] for recipient in recipient_list]

##Only works for shortDefinite recipients##
attention_theme = np.zeros((len(sent_list),12,12))
attention_recipient = np.zeros((len(sent_list),12,12))
for sent_num, sentence in enumerate(sent_list):
    words = sentence[const_type.index(args[1])].split(" ")
    words.append(".")
    words[1] = "[MASK]"
    tokenized_sentence = tokenizer.encode(" ".join(words))
    input_tensor = torch.tensor([tokenized_sentence])
    with torch.no_grad():
        outputs = model(input_tensor)
        for layer_num, layer in enumerate(outputs[2]):
            attention_layer = layer[0].numpy()
            attention_theme[sent_num,layer_num,:] = np.array([extract_theme(args[1],args[2],tokenized_sentence,attention) for attention in attention_layer])
            attention_recipient[sent_num,layer_num,:] = np.array([extract_recipient(args[1],args[2],tokenized_sentence,attention) for attention in attention_layer])

with open(PATH+'datafile/attention_theme_'+args[1]+'_'+args[2]+'.pkl','wb') as f:
    pickle.dump(attention_theme,f)
with open(PATH+'datafile/attention_recipient_'+args[1]+'_'+args[2]+'.pkl','wb') as f:
    pickle.dump(attention_recipient,f)
