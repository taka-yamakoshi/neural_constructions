import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

LSTM_verb = np.zeros(10)
LSTM_first_obj = np.zeros(10)
LSTM_eos = np.zeros(10)
LSTM_large_verb = np.zeros(10)
LSTM_large_first_obj = np.zeros(10)
LSTM_large_eos = np.zeros(10)
GPT2_verb = np.zeros(10)
GPT2_first_obj = np.zeros(10)
GPT2_eos = np.zeros(10)
GPT2_large_verb = np.zeros(10)
GPT2_large_first_obj = np.zeros(10)
GPT2_large_eos = np.zeros(10)

for i in range(10):
    with open(PATH + 'datafile/LSTM/test_performance_dais_LSTM_behav_pronoun_verb_'+str(i)+'.pkl','rb') as f:
        LSTM_verb[i] = np.average(pickle.load(f))
    with open(PATH + 'datafile/LSTM/test_performance_dais_LSTM_behav_pronoun_first_obj_'+str(i)+'.pkl','rb') as f:
        LSTM_first_obj[i] = np.average(pickle.load(f))
    with open(PATH + 'datafile/LSTM/test_performance_dais_LSTM_behav_pronoun_eos_'+str(i)+'.pkl','rb') as f:
        LSTM_eos[i] = np.average(pickle.load(f))
    with open(PATH + 'datafile/LSTM_Large/test_performance_dais_LSTM_large_behav_pronoun_verb_'+str(i)+'.pkl','rb') as f:
        LSTM_large_verb[i] = np.average(pickle.load(f))
    with open(PATH + 'datafile/LSTM_Large/test_performance_dais_LSTM_large_behav_pronoun_first_obj_'+str(i)+'.pkl','rb') as f:
        LSTM_large_first_obj[i] = np.average(pickle.load(f))
    with open(PATH + 'datafile/LSTM_Large/test_performance_dais_LSTM_large_behav_pronoun_eos_'+str(i)+'.pkl','rb') as f:
        LSTM_large_eos[i] = np.average(pickle.load(f))
    with open(PATH + 'datafile/GPT2/test_performance_dais_gpt2_behav_pronoun_verb_'+str(i)+'.pkl','rb') as f:
        GPT2_verb[i] = np.average(pickle.load(f))
    with open(PATH + 'datafile/GPT2/test_performance_dais_gpt2_behav_pronoun_first_obj_'+str(i)+'.pkl','rb') as f:
        GPT2_first_obj[i] = np.average(pickle.load(f))
    with open(PATH + 'datafile/GPT2/test_performance_dais_gpt2_behav_pronoun_eos_'+str(i)+'.pkl','rb') as f:
        GPT2_eos[i] = np.average(pickle.load(f))
    with open(PATH + 'datafile/GPT2_Large/test_performance_dais_gpt2_large_behav_pronoun_verb_all_head_'+str(i)+'.pkl','rb') as f:
        GPT2_large_verb[i] = np.average(pickle.load(f))
    with open(PATH + 'datafile/GPT2_Large/test_performance_dais_gpt2_large_behav_pronoun_first_obj_all_head_'+str(i)+'.pkl','rb') as f:
        GPT2_large_first_obj[i] = np.average(pickle.load(f))
    with open(PATH + 'datafile/GPT2_Large/test_performance_dais_gpt2_large_behav_pronoun_eos_all_head_'+str(i)+'.pkl','rb') as f:
        GPT2_large_eos[i] = np.average(pickle.load(f))

LSTM = [LSTM_verb,LSTM_first_obj,LSTM_eos]
LSTM_large = [LSTM_large_verb,LSTM_large_first_obj,LSTM_large_eos]
GPT2 = [GPT2_verb,GPT2_first_obj,GPT2_eos]
GPT2_large = [GPT2_large_verb,GPT2_large_first_obj,GPT2_large_eos]
fig, axis = plt.subplots()
plot_list_LSTM = axis.bar([0,5,10],np.array([np.average(performance) for performance in LSTM]),yerr =np.array([np.std(performance)/np.sqrt(len(performance)) for performance in LSTM]),color='lightseagreen',label='LSTM')
plot_list_LSTM = axis.bar([1,6,11],np.array([np.average(performance) for performance in LSTM_large]),yerr =np.array([np.std(performance)/np.sqrt(len(performance)) for performance in LSTM_large]),color='teal',label='LSTM-large')
plot_list_GPT2 = axis.bar([2,7,12],np.array([np.average(performance) for performance in GPT2]),yerr =np.array([np.std(performance)/np.sqrt(len(performance)) for performance in GPT2]),color='salmon',label='GPT2')
plot_list_GPT2 = axis.bar([3,8,13],np.array([np.average(performance) for performance in GPT2_large]),yerr =np.array([np.std(performance)/np.sqrt(len(performance)) for performance in GPT2_large]),color='crimson',label='GPT2-large')
plt.legend(bbox_to_anchor=(0.875, 1.03), loc='upper right', borderaxespad=1,fontsize=8)
plt.xticks([1.5,6.5,11.5],["End of the verb","End of the first object","End of the sentence "])
axis.set_xlabel("Timing")
axis.set_ylabel("Explained Variance")
plt.show()

def calculate_t(x,y):
    x_ave = np.average(x)
    y_ave = np.average(y)
    x_var = np.var(x)
    y_var = np.var(y)
    pooled_var = (x_var*(x.size-1)+y_var*(y.size-1))/(x.size+y.size-2)
    return (x_ave-y_ave)/(np.sqrt(pooled_var*(1/x.size+ 1/y.size)))

verb_data = [LSTM_verb,LSTM_large_verb,GPT2_verb,GPT2_large_verb]
first_obj_data = [LSTM_first_obj,LSTM_large_first_obj,GPT2_first_obj,GPT2_large_first_obj]
eos_data = [LSTM_eos,LSTM_large_eos,GPT2_eos,GPT2_large_eos]
t_matrix_verb = np.zeros((4,4))
t_matrix_first_obj = np.zeros((4,4))
t_matrix_eos = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        t_matrix_verb[i,j] = calculate_t(verb_data[i],verb_data[j])
        t_matrix_first_obj[i,j] = calculate_t(first_obj_data[i],first_obj_data[j])
        t_matrix_eos[i,j] = calculate_t(eos_data[i],eos_data[j])
plt.imshow(t_matrix_verb)
plt.show()
plt.imshow(t_matrix_first_obj)
plt.show()
plt.imshow(t_matrix_eos)
plt.show()
