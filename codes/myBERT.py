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
        return None
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


        return
layer_num = 0
input = torch.randn(768,10,dtype = torch.float)

