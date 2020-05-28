import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

with open(PATH + 'datafile/train_performance_dais_gpt2_verb.pkl','rb') as f:
    train_performance_verb = pickle.load(f)
with open(PATH + 'datafile/train_performance_dais_gpt2_first_obj.pkl','rb') as f:
    train_performance_first_obj = pickle.load(f)
with open(PATH + 'datafile/test_performance_recipient_dais_verb.pkl','rb') as f:
    test_performance_recipient_verb = pickle.load(f)
with open(PATH + 'datafile/test_performance_verb_dais_verb.pkl','rb') as f:
    test_performance_verb_verb = pickle.load(f)
with open(PATH + 'datafile/test_performance_recipient_dais_first_obj.pkl','rb') as f:
    test_performance_recipient_first_obj = pickle.load(f)
with open(PATH + 'datafile/test_performance_verb_dais_first_obj.pkl','rb') as f:
    test_performance_verb_first_obj = pickle.load(f)
with open(PATH + 'datafile/test_performance_recipient_dais_eos.pkl','rb') as f:
    test_performance_recipient_eos = pickle.load(f)
with open(PATH + 'datafile/test_performance_verb_dais_eos.pkl','rb') as f:
    test_performance_verb_eos = pickle.load(f)

test_performance_verb = (test_performance_recipient_verb*1200+test_performance_verb_verb*8000)/(1200+8000)
test_performance_first_obj = (test_performance_recipient_first_obj*1200+test_performance_verb_first_obj*8000)/(1200+8000)
test_performance_eos = (test_performance_recipient_eos*1200+test_performance_verb_eos*8000)/(1200+8000)
performance_list = [test_performance_verb,test_performance_first_obj,test_performance_eos]
fig, axis = plt.subplots()
plot_list = axis.bar(np.arange(3),np.array([np.average(performance) for performance in performance_list]),yerr =np.array([np.std(performance)/np.sqrt(len(performance)) for performance in performance_list]))
plt.xticks([0,1,2],["End of the verb","End of the first object","End of the sentence "])
axis.set_xlabel("Timing")
axis.set_ylabel("Accuracy")
plt.show()
