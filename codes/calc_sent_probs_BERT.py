import pickle
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import sys

sys.path.append('..')
args = sys.argv
##First argument: comparison categery (recipient or theme)
##Second argument: recipient/theme type (pronoun, shortDefinite, etc.)
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
if args[1] == "recipient":
    sent_list_alt = [[pair[head.index('DOsentence')],pair[head.index('PDsentence')]] for pair in corpus if pair[head.index('recipient_id')] == args[2] and pair[head.index('classification')] == "alternating"]
    sent_list_nonalt = [[pair[head.index('DOsentence')],pair[head.index('PDsentence')]] for pair in corpus if pair[head.index('recipient_id')] == args[2] and pair[head.index('classification')] == "non-alternating"]
elif args[1] == "theme":
    sent_list_alt = [[pair[head.index('DOsentence')],pair[head.index('PDsentence')]] for pair in corpus if pair[head.index('theme_type')] == args[2] and pair[head.index('classification')] == "alternating"]
    sent_list_nonalt = [[pair[head.index('DOsentence')],pair[head.index('PDsentence')]] for pair in corpus if pair[head.index('theme_type')] == args[2] and pair[head.index('classification')] == "non-alternating"]
else:
    logging.error('Invalid comparison category: should be either "recipient" or "theme"')


#Load the model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

mask_id = tokenizer.encode("[MASK]")[1:-1][0]
print("Calculating probability...")
DO_prob_alt = np.array([calc_sent_prob(sent[0]) for sent in sent_list_alt])
DO_prob_nonalt = np.array([calc_sent_prob(sent[0]) for sent in sent_list_nonalt])
print("Finished with DO")
PD_prob_alt = np.array([calc_sent_prob(sent[1]) for sent in sent_list_alt])
PD_prob_nonalt = np.array([calc_sent_prob(sent[1]) for sent in sent_list_nonalt])
print("Finished with PD")

alt_ratio = DO_prob_alt - PD_prob_alt
nonalt_ratio = DO_prob_nonalt - PD_prob_nonalt

#Dump the data
with open(PATH+'datafile/bert_log_ratio_'+args[2]+'.pkl','wb') as f:
    pickle.dump([alt_ratio,nonalt_ratio],f)
