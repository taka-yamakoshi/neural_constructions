import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

with open(PATH + 'datafile/sent_log_probs_good_do_'+args[1]+'.pkl','rb') as f:
    probs_good_do_1 = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_good_pd_'+args[1]+'.pkl','rb') as f:
    probs_good_pd_1 = pickle.load(f)

with open(PATH + 'datafile/sent_log_probs_good_do_'+args[2]+'.pkl','rb') as f:
    probs_good_do_2 = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_good_pd_'+args[2]+'.pkl','rb') as f:
    probs_good_pd_2 = pickle.load(f)

with open(PATH + 'datafile/sent_log_probs_bad_do_'+args[1]+'.pkl','rb') as f:
    probs_bad_do_1 = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_bad_pd_'+args[1]+'.pkl','rb') as f:
    probs_bad_pd_1 = pickle.load(f)

with open(PATH + 'datafile/sent_log_probs_bad_do_'+args[2]+'.pkl','rb') as f:
    probs_bad_do_2 = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_bad_pd_'+args[2]+'.pkl','rb') as f:
    probs_bad_pd_2 = pickle.load(f)




def calculate_t(x,y):
    x_ave = np.average(x)
    y_ave = np.average(y)
    x_var = np.var(x)
    y_var = np.var(y)
    pooled_var = (x_var*(x.size-1)+y_var*(y.size-1))/(x.size+y.size-2)
    return (x_ave-y_ave)/(np.sqrt(pooled_var*(1/x.size+ 1/y.size)))


good_1_log_ratio = [probs_good_do_1[j] - probs_good_pd_1[j] for j in range(len(probs_good_do_1))]
good_2_log_ratio = [probs_good_do_2[j] - probs_good_pd_2[j] for j in range(len(probs_good_do_2))]
bad_1_log_ratio = [probs_bad_do_1[j] - probs_bad_pd_1[j] for j in range(len(probs_bad_do_1))]
bad_2_log_ratio = [probs_bad_do_2[j] - probs_bad_pd_2[j] for j in range(len(probs_bad_do_2))]


fig, axis = plt.subplots()
probs = [good_1_log_ratio,good_2_log_ratio,bad_1_log_ratio,bad_2_log_ratio]

log_ratio_list = axis.bar(np.arange(4),np.array([np.average(probs[i]) for i in range(4)]),yerr =np.array([np.std(probs[i])/np.sqrt(len(probs[i])) for i in range(4)]))
log_ratio_list[0].set_color('blue')
log_ratio_list[1].set_color('blue')
log_ratio_list[2].set_color('red')
log_ratio_list[3].set_color('red')
log_ratio_list[0].set_label("Alternating Verb")
log_ratio_list[2].set_label("Non-alternating Verb")
plt.xticks([0,1,2,3],["Definite Recipients","Long Indefinite Recipients","Definite Recipients","Long Indefinite Recipients"],fontsize = 12)
axis.legend()
axis.set_xlabel("Types of recipients",fontsize = 15)
axis.set_ylabel("Log likelihood ratio",fontsize = 15)
axis.text(-0.05, 0, 'Double Object',
        transform=axis.transAxes,fontsize=15,rotation = 90,color = 'green')
axis.text(-0.05, 0.75, 'Preposition Dative',
          transform=axis.transAxes,fontsize=15,rotation = 90,color = 'green')
plt.gca().invert_yaxis()
plt.show()

