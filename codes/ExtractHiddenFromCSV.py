import pickle
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

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


def extract_hidden_BERT(sentence,verb):
    words = sentence.split(" ")
    words[1] = "[MASK]"
    words.append(".")
    tokenized_sentence = tokenizer.encode(" ".join(words))
    input_tensor = torch.tensor([tokenized_sentence])
    with torch.no_grad():
        outputs = model(input_tensor)
    state = torch.zeros((len(outputs[1]),768))
    for i, layer in enumerate(outputs[1]):
        state[i] = layer[0][tokenized_sentence.index(mask_id)]
    return state

def extract_hidden_GPT2(sentence,verb,first_obj):
    words = sentence.split(" ")
    words = ["<|endoftext|>"] + words + ["."]
    tokenized_sentence = tokenizer.encode(" ".join(words), add_special_tokens=True, add_prefix_space=True)
    verb_pos = tokenized_sentence.index(tokenizer.encode(verb,add_prefix_space=True)[-1])
    first_obj_pos  = tokenized_sentence.index(tokenizer.encode(first_obj,add_prefix_space=True)[-1])
    input_ids = torch.tensor(tokenized_sentence).unsqueeze(0)
    outputs = model(input_ids, labels=input_ids)
    state = np.zeros((len(outputs[3]),outputs[3][0][0].shape[1]))
    for i, layer in enumerate(outputs[3]):
        if args[2] == 'verb':
            state[i] = layer[0][verb_pos].detach().numpy()
        elif args[2] == 'first_obj':
            state[i] = layer[0][first_obj_pos].detach().numpy()
        elif args[2] == 'eos':
            state[i] = layer[0][-1].detach().numpy()
    return state

##Under Construction
def extract_hidden_with_verb_BERT(sentence):
    words = sentence.split(" ")
    words.append(".")
    tokenized_sentence = tokenizer.encode(" ".join(words))
    input_tensor = torch.tensor([tokenized_sentence])
    with torch.no_grad():
        outputs = model(input_tensor)
    state = np.zeros((len(outputs[1]),768))
    for i, layer in enumerate(outputs[1]):
        state[i] = layer[0][tokenized_sentence.index(mask_id)].numpy()
    return state

if args[1] == 'bert':
    print("Loading the model")
    from transformers import BertTokenizer, BertModel, BertForMaskedLM
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.eval()
    mask_id = tokenizer.encode("[MASK]")[1:-1][0]
    print("Calculating DO")
    hidden_states_DO = [extract_hidden_BERT(sentences[0]) for sentences in sent_list]
    print("Calculating PD")
    hidden_states_PD = [extract_hidden_BERT(sentences[1]) for sentences in sent_list]
    hidden_states = [hidden_states_DO,hidden_states_PD]

elif args[1] == 'gpt2':
    print("Loading the model")
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    hidden_states = np.zeros((2,len(sent_list),13,768))
    print("Calculating")
    for i, sentences in enumerate(sent_list):
        hidden_states[0][i] = extract_hidden_GPT2(sentences[0],verb_list[i],recipient_list[i])
        hidden_states[1][i] = extract_hidden_GPT2(sentences[1],verb_list[i],theme_list[i])
        if i%100 == 0:
            print(str(i) + " sentences done.")
else:
    print("Invalid model name")
    exit()

with open(PATH + 'datafile/hidden_states_dais_'+args[1]+'_'+args[2]+'.pkl','wb') as f:
    pickle.dump(hidden_states,f)
