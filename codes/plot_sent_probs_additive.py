import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

with open(PATH + 'datafile/sent_log_probs_'+args[1]+'_do_def_art.pkl','rb') as f:
    probs_def_do = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_'+args[1]+'_pd_def_art.pkl','rb') as f:
    probs_def_pd = pickle.load(f)

with open(PATH + 'datafile/sent_log_probs_'+args[1]+'_do_indef_art.pkl','rb') as f:
    probs_indef_do = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_'+args[1]+'_pd_indef_art.pkl','rb') as f:
    probs_indef_pd = pickle.load(f)

with open(PATH + 'datafile/sent_log_probs_'+args[1]+'_do_len.pkl','rb') as f:
    probs_len_do = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_'+args[1]+'_pd_len.pkl','rb') as f:
    probs_len_pd = pickle.load(f)

with open(PATH + 'datafile/sent_log_probs_'+args[1]+'_do_indef_art_len.pkl','rb') as f:
    probs_indef_len_do = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_'+args[1]+'_pd_indef_art_len.pkl','rb') as f:
    probs_indef_len_pd = pickle.load(f)




def calculate_t(x,y):
    x_ave = np.average(x)
    y_ave = np.average(y)
    x_var = np.var(x)
    y_var = np.var(y)
    pooled_var = (x_var*(x.size-1)+y_var*(y.size-1))/(x.size+y.size-2)
    return (x_ave-y_ave)/(np.sqrt(pooled_var*(1/x.size+ 1/y.size)))


def_log_ratio = [probs_def_do[j] - probs_def_pd[j] for j in range(len(probs_def_do))]
indef_log_ratio = [probs_indef_do[j] - probs_indef_pd[j] for j in range(len(probs_indef_do))]
len_log_ratio = [probs_len_do[j] - probs_len_pd[j] for j in range(len(probs_len_do))]
indef_len_log_ratio = [probs_indef_len_do[j] - probs_indef_len_pd[j] for j in range(len(probs_indef_len_do))]
expected_log_ratio = list(np.array(indef_log_ratio)+np.array(len_log_ratio)-np.array(def_log_ratio))

fig, axis = plt.subplots()
probs = [def_log_ratio,indef_log_ratio,len_log_ratio,indef_len_log_ratio,expected_log_ratio]

log_ratio_list = axis.bar(np.arange(5),np.array([np.average(probs[i]) for i in range(5)]),yerr =np.array([np.std(probs[i])/np.sqrt(len(probs[i])) for i in range(5)]))
if args[1] == 'good':
    log_ratio_list[0].set_color('blue')
    log_ratio_list[1].set_color('blue')
    log_ratio_list[2].set_color('blue')
    log_ratio_list[3].set_color('blue')
    log_ratio_list[4].set_color('purple')
    log_ratio_list.set_label("Alternating Verb")
if args[1] == 'bad':
    log_ratio_list[0].set_color('red')
    log_ratio_list[1].set_color('red')
    log_ratio_list[2].set_color('red')
    log_ratio_list[3].set_color('red')
    log_ratio_list[4].set_color('purple')
    log_ratio_list.set_label("Non-alternating Verb")

plt.xticks([0,1,2,3,4],["Definite Recipients","Indefinite Recipients","Long Recipients","Long Indefinite Recipients","Purely Additive"],fontsize = 12)
#axis.legend()
axis.set_xlabel("Types of recipients",fontsize = 15)
axis.set_ylabel("Log likelihood ratio",fontsize = 15)
axis.text(-0.05, 0, 'Double Object',
        transform=axis.transAxes,fontsize=15,rotation = 90,color = 'green')
axis.text(-0.05, 0.75, 'Preposition Dative',
          transform=axis.transAxes,fontsize=15,rotation = 90,color = 'green')
plt.gca().invert_yaxis()
plt.show()

print(calculate_t(np.array(indef_len_log_ratio),np.array(expected_log_ratio)))
