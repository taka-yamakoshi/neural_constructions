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
    probs_good_do_indef = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_good_pd_indef_art.pkl','rb') as f:
    probs_good_pd_indef = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_bad_do_indef_art.pkl','rb') as f:
    probs_bad_do_indef = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_bad_pd_indef_art.pkl','rb') as f:
    probs_bad_pd_indef = pickle.load(f)

with open(PATH + 'datafile/sent_log_probs_good_do_len.pkl','rb') as f:
    probs_good_do_len = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_good_pd_len.pkl','rb') as f:
    probs_good_pd_len = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_bad_do_len.pkl','rb') as f:
    probs_bad_do_len = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_bad_pd_len.pkl','rb') as f:
    probs_bad_pd_len = pickle.load(f)

with open(PATH + 'datafile/sent_log_probs_good_do_indef_art_len.pkl','rb') as f:
    probs_good_do_indef_len = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_good_pd_indef_art_len.pkl','rb') as f:
    probs_good_pd_indef_len = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_bad_do_indef_art_len.pkl','rb') as f:
    probs_bad_do_indef_len = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_bad_pd_indef_art_len.pkl','rb') as f:
    probs_bad_pd_indef_len = pickle.load(f)

def calculate_t(x,y):
    x_ave = np.average(x)
    y_ave = np.average(y)
    x_var = np.var(x)
    y_var = np.var(y)
    pooled_var = (x_var*(x.size-1)+y_var*(y.size-1))/(x.size+y.size-2)
    return (x_ave-y_ave)/(np.sqrt(pooled_var*(1/x.size+ 1/y.size)))


good_log_ratio = [probs_good_do[j] - probs_good_pd[j] for j in range(len(probs_good_do))]
bad_log_ratio = [probs_bad_do[j] - probs_bad_pd[j] for j in range(len(probs_bad_do))]
good_indef_log_ratio = [probs_good_do_indef[j] - probs_good_pd_indef[j] for j in range(len(probs_good_do_indef))]
bad_indef_log_ratio = [probs_bad_do_indef[j] - probs_bad_pd_indef[j] for j in range(len(probs_bad_do_indef))]
good_len_log_ratio = [probs_good_do_len[j] - probs_good_pd_len[j] for j in range(len(probs_good_do_len))]
bad_len_log_ratio = [probs_bad_do_len[j] - probs_bad_pd_len[j] for j in range(len(probs_bad_do_len))]
good_indef_len_log_ratio = [probs_good_do_indef_len[j] - probs_good_pd_indef_len[j] for j in range(len(probs_good_do_indef_len))]
bad_indef_len_log_ratio = [probs_bad_do_indef_len[j] - probs_bad_pd_indef_len[j] for j in range(len(probs_bad_do_indef_len))]
print("good vs bad: t = " + str(calculate_t(np.array(good_log_ratio),np.array(bad_log_ratio))))
print("good vs good_indef: t = " + str(calculate_t(np.array(good_log_ratio),np.array(good_indef_log_ratio))))
print("good vs bad_indef: t = " + str(calculate_t(np.array(good_log_ratio),np.array(bad_indef_log_ratio))))
print("good vs good_len: t = " + str(calculate_t(np.array(good_log_ratio),np.array(good_len_log_ratio))))
print("good vs bad_len: t = " + str(calculate_t(np.array(good_log_ratio),np.array(bad_len_log_ratio))))
print("good vs good_indef_len: t = " + str(calculate_t(np.array(good_log_ratio),np.array(good_indef_len_log_ratio))))
print("good vs bad_indef_len: t = " + str(calculate_t(np.array(good_log_ratio),np.array(bad_indef_len_log_ratio))))


fig, axis = plt.subplots()
probs = [good_log_ratio,bad_log_ratio,good_indef_log_ratio,bad_indef_log_ratio,good_len_log_ratio,bad_len_log_ratio, good_indef_len_log_ratio, bad_indef_len_log_ratio]
matrix = [[calculate_t(np.array(prob_1),np.array(prob_2)) for prob_1 in probs] for prob_2 in probs]

plt.bar(np.arange(8),np.array([np.average(probs[i]) for i in range(8)]),yerr =np.array([np.std(probs[i])/np.sqrt(len(probs[i])) for i in range(8)]))
plt.xticks([0,1,2,3,4,5,6,7],["A","NA","A with indefinite articles","NA with indefinite articles","A with long noun phrases","NA with long noun phrases","A with long nouns with indefinite articles","NA with long nouns with indefinite articles"],rotation=5)
axis.set_xlabel("Types of verbs/objects")
axis.set_ylabel("Log likelihood ratio")
plt.gca().invert_yaxis()
plt.show()
plt.imshow(abs(np.array(matrix)))
plt.show()
