import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM


def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x))

sentence = "[CLS] the man brought her the box . [SEP]"

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

tokenized_text = sentence.split(" ")
masked_index = 3
masked_text = tokenized_text.copy()
masked_text[masked_index] = '[MASK]'
indexed_tokens = tokenizer.convert_tokens_to_ids(masked_text)
tokens_tensor = torch.tensor([indexed_tokens])
with torch.no_grad():
    #print(model.config)
    outputs = model(tokens_tensor)
    print(outputs[2][0].shape)
    predictions = outputs[0]

probs = predictions[0, masked_index]
log_probs = torch.log(softmax(probs))
print(tokenizer.convert_ids_to_tokens([torch.argmax(log_probs)]))

