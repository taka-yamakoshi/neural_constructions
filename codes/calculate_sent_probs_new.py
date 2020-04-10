import pickle
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x))
def calc_sent_prob(sentence):
    words = sentence.split(" ")
    words.append(".")
    tokenized_sentence = tokenizer.encode(" ".join(words))
    mask_id = tokenizer.encode("[MASK]")[1:-1][0]
    sent_prob = 0
    for masked_index in range(1,(len(tokenized_sentence)-1)):
        masked_sentence = tokenized_sentence.copy()
        masked_sentence[masked_index] = mask_id
        input_tensor = torch.tensor([masked_sentence])
        with torch.no_grad():
            outputs = model(input_tensor)
            predictions = outputs[0]
        probs = predictions[0, masked_index]
        log_probs = torch.log(softmax(probs))
        prob = log_probs[tokenized_sentence[masked_index]].item()
        sent_prob += prob
    return sent_prob



#Load sentences
with open(PATH+'textfile/generated_pairs.csv') as f:
    reader = csv.reader(f)
    corpus = [row for row in reader][1:]

sent_list = []

for pair in corpus:
    if pair[-1] == args[1]:
        sent_list.append([pair[4],pair[5]])

#Load the model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

print("Calculating probability...")
DO_prob = np.array([calc_sent_prob(sent[0]) for sent in sent_list])
print("Finished with DO")
PD_prob = np.array([calc_sent_prob(sent[1]) for sent in sent_list])
print("Finished with PD")

ratio = DO_prob - PD_prob

#Dump the data
with open(PATH+'datafile/sent_prob_ratio_'+args[1]+'.pkl','wb') as f:
    pickle.dump(ratio,f)
