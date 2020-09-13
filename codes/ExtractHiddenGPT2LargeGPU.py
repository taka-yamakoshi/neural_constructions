import pickle
import numpy as np
import csv
from multiprocessing import Pool
import torch
import sys

sys.path.append('..')
args = sys.argv

def padding(sent_list,verb_list,first_obj_list,max_sent_len):
    padded_sent = torch.zeros((5000,max_sent_len))
    verb_pos_list = np.zeros(5000)
    first_obj_pos_list = np.zeros(5000)
    eos_pos_list = np.zeros(5000)
    for i,sentence in enumerate(sent_list):
        words = sentence.split(" ")
        words = ["<|endoftext|>"] + words + ["."]
        tokenized_sentence = tokenizer.encode(" ".join(words), add_special_tokens=True, add_prefix_space=True)
        verb_pos_list[i] = tokenized_sentence.index(tokenizer.encode(verb_list[i],add_prefix_space=True)[-1])
        first_obj_pos_list[i]  = tokenized_sentence.index(tokenizer.encode(first_obj_list[i],add_prefix_space=True)[-1])
        eos_pos_list[i] = tokenized_sentence.index(eos)
        if len(tokenized_sentence) > max_sent_len:
            print("Need to increase the max_sent_len")
            exit()
        padding = " ".join(["0" for i in range(max_sent_len-len(tokenized_sentence))])
        tokenized_sentence.extend(tokenizer.encode(padding))
        padded_sent[i] = torch.tensor(tokenized_sentence)
    return padded_sent, verb_pos_list, first_obj_pos_list, eos_pos_list

def batchify(data_list,batch_size):
    batch_num = int(len(data_list)/batch_size)
    if torch.is_tensor(data_list):
        return torch.tensor([list(data_list[batch_size*i:batch_size*(i+1)].numpy()) for i in range(batch_num)])
    else:
        return np.array([list(data_list[batch_size*i:batch_size*(i+1)]) for i in range(batch_num)])

def extract_hidden_GPT2_large(sentence_batch,verb_batch,first_obj_batch,eos_batch):
    outputs = model(sentence_batch, labels=sentence_batch)
    print(outputs.shape)
    exit()
    state = np.zeros((len(outputs[3]),outputs[3][0][0].shape[1]))
    for i, layer in enumerate(outputs[3]):
        if args[1] == 'verb':
            state[i] = layer[0][verb_pos].detach().numpy()
        elif args[1] == 'first_obj':
            state[i] = layer[0][first_obj_pos].detach().numpy()
        elif args[1] == 'eos':
            state[i] = layer[0][eos_pos].detach().numpy()
    return state


print("Loading the model")
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large')
model.eval()
                
#Load sentences
print("Loading sentences")
with open('../textfile/generated_pairs_latest.csv') as f:
    reader = csv.reader(f)
    file = [row for row in reader]
    head = file[0]
    corpus = file[1:]
DO_sent_list = [row[head.index('DOsentence')] for row in corpus]
PD_sent_list = [row[head.index('PDsentence')] for row in corpus]
verb_list = [row[head.index('DOsentence')].split(" ")[1] for row in corpus]
recipient_list  = [row[head.index('PDsentence')].split(" ")[-1] for row in corpus]
theme_list = [row[head.index('PDsentence')].split(" ")[row[head.index('PDsentence')].split(" ").index("to")-1] for row in corpus]

#Padding
max_sent_len = 20
eos = tokenizer.encode(".",add_prefix_space=True)[-1]
padded_DO, verb_pos_list_DO, first_obj_pos_list_DO, eos_pos_list_DO = padding(DO_sent_list,verb_list,recipient_list,max_sent_len)
padded_PD, verb_pos_list_PD, first_obj_pos_list_PD, eos_pos_list_PD = padding(PD_sent_list,verb_list,theme_list,max_sent_len)

#Batchfy
batch_size = 50
padded_DO = batchify(padded_DO,batch_size)
padded_PD = batchify(padded_PD,batch_size)
verb_pos_list_DO = batchify(verb_pos_list_DO,batch_size)
verb_pos_list_PD = batchify(verb_pos_list_PD,batch_size)
first_obj_pos_list_DO = batchify(first_obj_pos_list_DO,batch_size)
first_obj_pos_list_PD = batchify(first_obj_pos_list_PD,batch_size)
eos_pos_list_DO = batchify(eos_pos_list_DO,batch_size)
eos_pos_list_PD = batchify(eos_pos_list_PD,batch_size)

print(padded_DO.shape)
print(verb_pos_list_DO.shape)

hidden_states = np.zeros((2,len(sent_list),37,1280))
print("Calculating")
for i in range(len(padded_DO)):
    hidden_states[0][batch_size*i:batch_size*(i+1)] = extract_hidden_GPT2_large(padded_DO[i],verb_pos_list_DO[i],first_obj_pos_list_DO[i])


with open('../datafile/hidden_states_dais_gpt2_large_'+args[1]+'.pkl','wb') as f:
    pickle.dump(hidden_states,f)
