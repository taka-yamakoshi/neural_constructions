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

def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x))

with open(PATH+'datafile/attention_theme_DO_shortDefinite.pkl','rb') as f:
    attention_theme_DO = pickle.load(f)
with open(PATH+'datafile/attention_theme_PD_shortDefinite.pkl','rb') as f:
    attention_theme_PD = pickle.load(f)
with open(PATH+'datafile/attention_recipient_DO_shortDefinite.pkl','rb') as f:
    attention_recipient_DO = pickle.load(f)
with open(PATH+'datafile/attention_recipient_PD_shortDefinite.pkl','rb') as f:
    attention_recipient_PD = pickle.load(f)


with open(PATH+'datafile/test_performance.pkl','rb') as f:
    performance = pickle.load(f)
mean = np.average(performance[1:])
stdev = np.std(performance[1:])
good_heads = performance[1:]==1
bad_heads = performance[1:] < (mean-2.626*stdev)
bottom_20_heads = performance[1:] < (mean-1.1*stdev)




good_attention_theme_DO = np.array([attention[good_heads] for attention in attention_theme_DO])
bad_attention_theme_DO = np.array([attention[bad_heads] for attention in attention_theme_DO])
bottom_20_attention_theme_DO = np.array([attention[bottom_20_heads] for attention in attention_theme_DO])

good_attention_recipient_DO = np.array([attention[good_heads] for attention in attention_recipient_DO])
bad_attention_recipient_DO = np.array([attention[bad_heads] for attention in attention_recipient_DO])
bottom_20_attention_recipient_DO = np.array([attention[bottom_20_heads] for attention in attention_recipient_DO])

good_attention_theme_PD = np.array([attention[good_heads] for attention in attention_theme_PD])
bad_attention_theme_PD = np.array([attention[bad_heads] for attention in attention_theme_PD])
bottom_20_attention_theme_PD = np.array([attention[bottom_20_heads] for attention in attention_theme_PD])

good_attention_recipient_PD = np.array([attention[good_heads] for attention in attention_recipient_PD])
bad_attention_recipient_PD = np.array([attention[bad_heads] for attention in attention_recipient_PD])
bottom_20_attention_recipient_PD = np.array([attention[bottom_20_heads] for attention in attention_recipient_PD])


good_attention_theme = np.zeros((2000,np.sum(good_heads)))
good_attention_recipient = np.zeros((2000,np.sum(good_heads)))
good_attention_theme[:1000] = good_attention_theme_DO
good_attention_theme[1000:] = good_attention_theme_PD
good_attention_recipient[:1000] = good_attention_recipient_DO
good_attention_recipient[1000:] = good_attention_recipient_PD

bad_attention_theme = np.zeros((2000,np.sum(bad_heads)))
bad_attention_recipient = np.zeros((2000,np.sum(bad_heads)))
bad_attention_theme[:1000] = bad_attention_theme_DO
bad_attention_theme[1000:] = bad_attention_theme_PD
bad_attention_recipient[:1000] = bad_attention_recipient_DO
bad_attention_recipient[1000:] = bad_attention_recipient_PD

bottom_20_attention_theme = np.zeros((2000,np.sum(bottom_20_heads)))
bottom_20_attention_recipient = np.zeros((2000,np.sum(bottom_20_heads)))
bottom_20_attention_theme[:1000] = bottom_20_attention_theme_DO
bottom_20_attention_theme[1000:] = bottom_20_attention_theme_PD
bottom_20_attention_recipient[:1000] = bottom_20_attention_recipient_DO
bottom_20_attention_recipient[1000:] = bottom_20_attention_recipient_PD

def plot_attention_indiv_head(good_attn,bad_attn):
    plot_data = np.zeros(((good_attn.shape[1]+bad_attn.shape[1]),good_attn.shape[0]))
    for i in range(good_attn.shape[1]):
        plot_data[i] = good_attn[:,i]
    for i in range(bad_attn.shape[1]):
        plot_data[(good_attn.shape[1]+i)] = bad_attn[:,i]
    fig, axis = plt.subplots()
    attention_list = axis.bar(np.arange(plot_data.shape[0]),np.array([np.average(data) for data in plot_data]),yerr =np.array([np.std(data)/np.sqrt(len(data)) for data in plot_data]))
    for i in range(good_attn.shape[1]):
        attention_list[i].set_color('blue')
    for i in range(bad_attn.shape[1]):
        attention_list[good_attn.shape[1]+i].set_color('red')
    plt.show()

plot_attention_indiv_head(good_attention_theme,bad_attention_theme)
plot_attention_indiv_head(good_attention_recipient,bad_attention_recipient)


good_head_data = [good_attention_theme,good_attention_recipient]
bottom_20_head_data = [bottom_20_attention_theme,bottom_20_attention_recipient]
fig, axis = plt.subplots()
plot_list_1 = axis.bar([0,2],np.array([np.average(data) for data in good_head_data]),yerr =np.array([np.std(data)/np.sqrt(data.size) for data in good_head_data]),label = "Good heads",color='blue')
plot_list_2 = axis.bar([1,3],np.array([np.average(data) for data in bottom_20_head_data]),yerr =np.array([np.std(data)/np.sqrt(data.size) for data in bottom_20_head_data]),label = "Bad heads",color='red')
plt.xticks([0.5,2.5],['Theme','Recipient'])
axis.set_ylabel("Attention weight")
axis.set_title("Average attention weight for good/bad 20 heads")
plt.legend()
plt.show()

