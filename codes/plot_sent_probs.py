import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

with open(PATH + 'datafile/sent_log_probs_good_do_bert.pkl','rb') as f:
    probs_good_do = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_good_pd_bert.pkl','rb') as f:
    probs_good_pd = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_bad_do_bert.pkl','rb') as f:
    probs_bad_do = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_bad_pd_bert.pkl','rb') as f:
    probs_bad_pd = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_good_do_indef_art.pkl','rb') as f:
    probs_indef_do = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_good_pd_indef_art.pkl','rb') as f:
    probs_indef_pd = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_good_do_len.pkl','rb') as f:
    probs_len_do = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_good_pd_len.pkl','rb') as f:
    probs_len_pd = pickle.load(f)

def calculate_t(x,y):
    x_ave = np.average(x)
    y_ave = np.average(y)
    x_var = np.var(x)
    y_var = np.var(y)
    pooled_var = (x_var*(x.size-1)+y_var*(y.size-1))/(x.size+y.size-2)
    return (x_ave-y_ave)/(np.sqrt(pooled_var*(1/x.size+ 1/y.size)))


good_log_ratio = [probs_good_do[j] - probs_good_pd[j] for j in range(len(probs_good_do))]
bad_log_ratio = [probs_bad_do[j] - probs_bad_pd[j] for j in range(len(probs_bad_do))]
indef_log_ratio = [probs_indef_do[j] - probs_indef_pd[j] for j in range(len(probs_indef_do))]
len_log_ratio = [probs_len_do[j] - probs_len_pd[j] for j in range(len(probs_len_do))]
print("good vs bad: t = " + str(calculate_t(np.array(good_log_ratio),np.array(bad_log_ratio))))
print("good vs indef: t = " + str(calculate_t(np.array(good_log_ratio),np.array(indef_log_ratio))))
print("good vs len: t = " + str(calculate_t(np.array(good_log_ratio),np.array(len_log_ratio))))

fig, axis = plt.subplots()
probs = [good_log_ratio,bad_log_ratio,indef_log_ratio,len_log_ratio]
plt.bar(np.arange(4),np.array([np.average(probs[i]) for i in range(4)]),yerr =np.array([np.std(probs[i])/np.sqrt(len(probs[i])) for i in range(4)]))
plt.xticks([0,1,2,3],["Alternating verbs","Non-alternating Verbs","Nouns with indefinite articles","Long noun phrases" ])
axis.set_xlabel("Types of verbs/objects")
axis.set_ylabel("Log likelihood ratio")
plt.gca().invert_yaxis()
plt.show()

