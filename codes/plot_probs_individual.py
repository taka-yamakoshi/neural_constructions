import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

with open(PATH + 'datafile/log_probs_good_do_'+args[1]+'.pkl','rb') as f:
    probs_good_do = pickle.load(f)
with open(PATH + 'datafile/log_probs_good_pd_'+args[1]+'.pkl','rb') as f:
    probs_good_pd = pickle.load(f)
with open(PATH + 'datafile/log_probs_bad_do_'+args[1]+'.pkl','rb') as f:
    probs_bad_do = pickle.load(f)
with open(PATH + 'datafile/log_probs_bad_pd_'+args[1]+'.pkl','rb') as f:
    probs_bad_pd = pickle.load(f)


with open(PATH + 'textfile/good_do_'+args[1]+'.txt') as f:
    good_text = f.read().split('\n')[:-1]
with open(PATH + 'textfile/bad_do_'+args[1]+'.txt') as f:
    bad_text = f.read().split('\n')[:-1]


good_verb = [sentence.split(" ")[3] for sentence in good_text]
bad_verb = [sentence.split(" ")[3] for sentence in bad_text]
red_good_verb = [good_verb[2*i] for i in range(int(len(good_verb)/2))]
red_bad_verb = [bad_verb[2*i] for i in range(int(len(bad_verb)/2))]

good_log_ratio = [probs_good_do[j] - probs_good_pd[j] for j in range(len(probs_good_do))]
bad_log_ratio = [probs_bad_do[j] - probs_bad_pd[j] for j in range(len(probs_bad_do))]
red_good_log_ratio = [(good_log_ratio[2*i]+good_log_ratio[2*i+1])/2 for i in range(len(red_good_verb))]
red_bad_log_ratio = [(bad_log_ratio[2*i]+bad_log_ratio[2*i+1])/2 for i in range(len(red_bad_verb))]

concat_ratio = red_good_log_ratio.copy()
concat_ratio.extend(red_bad_log_ratio.copy())
concat_ratio = np.array(concat_ratio)
verbs = red_good_verb.copy()
verbs.extend(red_bad_verb.copy())
verbs = np.array(verbs)

order = np.argsort(concat_ratio)
good_order = order[:len(red_good_verb)]
bad_order = order[len(red_good_verb):]

probs = concat_ratio[order]
new_verbs = verbs[order]
fig, axis = plt.subplots()
probs_list = axis.bar(np.arange(len(probs)),probs)
for i in range(len(red_good_verb)):
    probs_list[list(order).index(i)].set_color('blue')
for i in range(len(red_good_verb),len(verbs)):
    probs_list[list(order).index(i)].set_color('red')
plt.xticks(np.arange(len(probs)),new_verbs,rotation=90)
axis.set_xlabel("Verbs")
axis.set_ylabel("Log likelihood ratio")
plt.show()


