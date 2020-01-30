import pickle
import numpy as np
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

#Load sentences
sentence_file = open(PATH+'textfile/lex_semantics.txt')
sentence_str = sentence_file.read()
sentence_file.close()
sentence_str = sentence_str.split('\n')[:-1]


#Load the model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()



with open(PATH + 'textfile/probs_lex_sematics.txt','w') as f:
    for sentence in sentence_str:
        tokenized_text = sentence.split(" ")
        masked_index = 5
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
        # write out top 50 prepositions for each verb
        for id in order[:50]:
            prob = log_probs[id].item()
            f.write(tokenizer.convert_ids_to_tokens([id])[0])
            f.writelines('\t')
            f.write(str(prob))
            f.writelines('\n')
        f.writelines('\n')
        f.writelines('\n')
