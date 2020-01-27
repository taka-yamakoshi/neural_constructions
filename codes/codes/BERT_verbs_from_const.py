import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import sys

sys.path.append('..')
args = sys.argv
PATH = 'Path to the folder'

def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x))

sentence_file = open(PATH+'textfile/'+ args[1] + '_' + args[2] +'_const'+'.txt')
sentence_str = sentence_file.read()
sentence_file.close()

sentence_str = sentence_str.split('\n')[:-1]


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()


with open(PATH + 'textfile/probs_from_const_'+args[1]+'_'+args[2]+'.txt', 'w') as f:
    for text in sentence_str:
        tokenized_text = text.split(" ")
        masked_index = 2
        masked_text = tokenized_text.copy()
        masked_text[masked_index] = '[MASK]'
        f.write(" ".join(masked_text))
        f.writelines('\n')
        indexed_tokens = tokenizer.convert_tokens_to_ids(masked_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]
        probs = predictions[0, masked_index]
        log_probs = torch.log(softmax(probs)).numpy()
        order = np.argsort(log_probs)[::-1]
        # write out top 50 verbs for each construction
        for id in order[:50]:
            prob = log_probs[id].item()
            f.write(tokenizer.convert_ids_to_tokens([id])[0])
            f.writelines('\t')
            f.write(str(prob))
            f.writelines('\n')
