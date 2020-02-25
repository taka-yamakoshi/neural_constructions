import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x))

def gelu(x):
    return 0.5*x*(1+torch.tanh(np.sqrt(2/np.pi)*(x + 0.044715*x**3)))

def attention(query,key,value):
    raw_attn = torch.mm(query.T,key)
    attn = torch.zeros_like(raw_attn)
    for i in range(raw_attn.shape[0]):
        attn[i] = softmax(raw_attn[i])
    return torch.mm(attn,value.T).T

def layer_norm(input,weight,bias):
    mean = input.mean(dim=1)
    sigma = torch.sqrt(torch.sum(torch.tensor([torch.dot(vec - mean, vec - mean) for vec in input.T]))/input.shape[1])
    return weight.view(768,1) * (input-mean.view(768,1))/sigma.item() + bias.view(768,1)

tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()
params = model.state_dict()

class trained_model():
    def __init__(self,params):
        return None
    def embed(self,input):
        word_embed = params['bert.embeddings.word_embeddings.weight']
        position_embed = params['bert.embeddings.position_embeddings.weight']
        token_type_embed = params['bert.embeddings.token_type_embeddings.weight']
        layer_norm_weight = params['bert.embeddings.LayerNorm.weight']
        layer_norm_bias = params['bert.embeddings.LayerNorm.bias']
        word_vecs = torch.tensor([word_embed[id].tolist() for id in input]).T
        position_vecs = position_embed[:word_vecs.shape[1]].T
        embed = word_vecs + position_vecs
        embed_norm = layer_norm(embed,layer_norm_weight,layer_norm_bias)
        return embed_norm
    
    def MyAttnLayer(self,layer_num,input):
        query_weight = params['bert.encoder.layer.'+str(layer_num)+'.attention.self.query.weight']
        query_bias = params['bert.encoder.layer.'+str(layer_num)+'.attention.self.query.bias']
        key_weight = params['bert.encoder.layer.'+str(layer_num)+'.attention.self.key.weight']
        key_bias = params['bert.encoder.layer.'+str(layer_num)+'.attention.self.key.bias']
        value_weight = params['bert.encoder.layer.'+str(layer_num)+'.attention.self.value.weight']
        value_bias = params['bert.encoder.layer.'+str(layer_num)+'.attention.self.value.bias']
        
        attn_out_weight = params['bert.encoder.layer.'+str(layer_num)+'.attention.output.dense.weight']
        attn_out_bias = params['bert.encoder.layer.'+str(layer_num)+'.attention.output.dense.bias']
        attn_layer_norm_weight = params['bert.encoder.layer.'+str(layer_num)+'.attention.output.LayerNorm.weight']
        attn_layer_norm_bias = params['bert.encoder.layer.'+str(layer_num)+'.attention.output.LayerNorm.bias']
        interm_weight = params['bert.encoder.layer.'+str(layer_num)+'.intermediate.dense.weight']
        interm_bias = params['bert.encoder.layer.'+str(layer_num)+'.intermediate.dense.bias']
        out_weight = params['bert.encoder.layer.'+str(layer_num)+'.output.dense.weight']
        out_bias = params['bert.encoder.layer.'+str(layer_num)+'.output.dense.bias']
        out_layer_norm_weight = params['bert.encoder.layer.'+str(layer_num)+'.output.LayerNorm.weight']
        out_layer_norm_bias = params['bert.encoder.layer.'+str(layer_num)+'.output.LayerNorm.bias']
        
        query = torch.mm(query_weight,input) + query_bias.view(768,1)
        key = torch.mm(key_weight,input) + key_bias.view(768,1)
        value = torch.mm(value_weight,input) + value_bias.view(768,1)
        attn = torch.zeros_like(value)
        for head_num in range(12):
            attn[64*head_num:64*(head_num+1)] = attention(query[64*head_num:64*(head_num+1)],key[64*head_num:64*(head_num+1)],value[64*head_num:64*(head_num+1)])
        attn_out = torch.mm(attn_out_weight,attn) + attn_out_bias.view(768,1)
        attn_added = attn_out + input
        attn_norm = layer_norm(attn_added,attn_layer_norm_weight,attn_layer_norm_bias)
        
        interm_vecs = gelu(torch.mm(interm_weight,attn_norm) + interm_bias.view(3072,1))
        out_vecs = torch.mm(out_weight,interm_vecs) + out_bias.view(768,1)
        out_added = out_vecs + attn_norm
        out_norm = layer_norm(out_added,out_layer_norm_weight,out_layer_norm_bias)
        return out_norm

sent_str = "[SEP] the man [MASK] her the box . [SEP]"
sent_list = sent_str.split(" ")
input_list = tokenizer.convert_tokens_to_ids(sent_list)
embed_vecs = trained_model(params).embed(input_list)
hidden = embed_vecs
MyModel = trained_model(params)
for layer_num in range(12):
    hidden = MyModel.MyAttnLayer(layer_num,hidden)

out_pool_weight = params['bert.pooler.dense.weight']
out_pool_bias = params['bert.pooler.dense.bias']
word_embed = params['bert.embeddings.word_embeddings.weight']

out = torch.tanh(torch.mm(out_pool_weight,hidden) + out_pool_bias.view(768,1))
decoded = torch.mm(out.T,word_embed.T)
prediction = decoded[5]
print(tokenizer.convert_ids_to_tokens([torch.argmax(prediction)]))

