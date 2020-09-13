import pickle
import numpy as np
import csv
import logging
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
def calc_prob(tokenized_sentence,masked_index):
    masked_sentence = tokenized_sentence.copy()
    masked_sentence[masked_index] = mask_id
    input_tensor = torch.tensor([masked_sentence])
    with torch.no_grad():
        outputs = model(input_tensor)
        predictions = outputs[0]
        probs = predictions[0, masked_index]
        log_probs = torch.log(softmax(probs))
        prob = log_probs[tokenized_sentence[masked_index]].item()
    return prob

def calc_sent_prob(sentence):
    words = sentence.split(" ")
    words.append(".")
    tokenized_sentence = tokenizer.encode(" ".join(words))
    prob_list = [calc_prob(tokenized_sentence,masked_index) for masked_index in range(1,(len(tokenized_sentence)-1))]
    sent_prob = np.sum(np.array(prob_list))
    return sent_prob



#Load sentences
with open(PATH+'textfile/generated_pairs.csv') as f:
    reader = csv.reader(f)
    file = [row for row in reader]
    head = file[0]
    corpus = file[1:]

sent_list = [[pair[head.index('DOsentence')],pair[head.index('PDsentence')]] for pair in corpus]

#Load the model
print("Loading the model")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

#Calculate probability
mask_id = tokenizer.encode("[MASK]")[1:-1][0]
print("Calculating DO")
DO_prob = np.array([calc_sent_prob(sent[0]) for sent in sent_list[:100]])
print("Calculating PD")
PD_prob = np.array([calc_sent_prob(sent[1]) for sent in sent_list[:100]])
ratio = DO_prob - PD_prob

#Dump the data
print("Dumping data")
with open(PATH+'datafile/bert_DO_test.pkl','wb') as f:
    pickle.dump(DO_prob,f)
with open(PATH+'datafile/bert_PD_test.pkl','wb') as f:
    pickle.dump(PD_prob,f)
with open(PATH+'datafile/bert_log_ratio_test.pkl','wb') as f:
    pickle.dump(ratio,f)


with open(PATH+'textfile/generated_pairs_probs_test.csv','w') as f:
    writer = csv.writer(f)
    head.extend(['DO_prob','PD_prob','ratio'])
    writer.writerow(head)
    for i,row in enumerate(corpus[:100]):
        row.extend([DO_prob[i],PD_prob[i],ratio[i]])
        writer.writerow(row)

