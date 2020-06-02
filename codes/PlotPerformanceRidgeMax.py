import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

LSTM_verb = np.zeros((10,4))
LSTM_first_obj = np.zeros((10,4))
LSTM_eos = np.zeros((10,4))
LSTM_large_verb = np.zeros(10)
LSTM_large_first_obj = np.zeros(10)
LSTM_large_eos = np.zeros(10)
GPT2_verb = np.zeros((10,13))
GPT2_first_obj = np.zeros((10,13))
GPT2_eos = np.zeros((10,13))
GPT2_large_verb = np.zeros((10,37))
GPT2_large_first_obj = np.zeros((10,37))
GPT2_large_eos = np.zeros((10,37))

for i in range(10):
    with open(PATH + 'datafile/LSTM/test_performance_dais_LSTM_behav_pronoun_verb_'+str(i)+'_new.pkl','rb') as f:
        LSTM_verb[i] = pickle.load(f)
    with open(PATH + 'datafile/LSTM/test_performance_dais_LSTM_behav_pronoun_first_obj_'+str(i)+'_new.pkl','rb') as f:
        LSTM_first_obj[i] = pickle.load(f)
    with open(PATH + 'datafile/LSTM/test_performance_dais_LSTM_behav_pronoun_eos_'+str(i)+'_new.pkl','rb') as f:
        LSTM_eos[i] = pickle.load(f)
    with open(PATH + 'datafile/LSTM_Large/test_performance_dais_LSTM_large_behav_pronoun_verb_'+str(i)+'.pkl','rb') as f:
        LSTM_large_verb[i] = pickle.load(f)
    with open(PATH + 'datafile/LSTM_Large/test_performance_dais_LSTM_large_behav_pronoun_first_obj_'+str(i)+'.pkl','rb') as f:
        LSTM_large_first_obj[i] = pickle.load(f)
    with open(PATH + 'datafile/LSTM_Large/test_performance_dais_LSTM_large_behav_pronoun_eos_'+str(i)+'.pkl','rb') as f:
        LSTM_large_eos[i] = pickle.load(f)
    with open(PATH + 'datafile/GPT2/test_performance_dais_gpt2_behav_pronoun_verb_'+str(i)+'_new.pkl','rb') as f:
        GPT2_verb[i] = pickle.load(f)
    with open(PATH + 'datafile/GPT2/test_performance_dais_gpt2_behav_pronoun_first_obj_'+str(i+10)+'_new.pkl','rb') as f:
        GPT2_first_obj[i] = pickle.load(f)
    with open(PATH + 'datafile/GPT2/test_performance_dais_gpt2_behav_pronoun_eos_'+str(i)+'_new.pkl','rb') as f:
        GPT2_eos[i] = pickle.load(f)
    with open(PATH + 'datafile/GPT2_Large/test_performance_dais_gpt2_large_behav_pronoun_verb_all_head_'+str(i)+'.pkl','rb') as f:
        GPT2_large_verb[i] = pickle.load(f)
    with open(PATH + 'datafile/GPT2_Large/test_performance_dais_gpt2_large_behav_pronoun_first_obj_all_head_'+str(i)+'.pkl','rb') as f:
        GPT2_large_first_obj[i] = pickle.load(f)
    with open(PATH + 'datafile/GPT2_Large/test_performance_dais_gpt2_large_behav_pronoun_eos_all_head_'+str(i)+'_new.pkl','rb') as f:
        GPT2_large_eos[i] = pickle.load(f)

LSTM = [np.max(LSTM_verb,axis=1),np.max(LSTM_first_obj,axis=1),np.max(LSTM_eos,axis=1)]
LSTM_large = [LSTM_large_verb,LSTM_large_first_obj,LSTM_large_eos]
GPT2 = [np.max(GPT2_verb,axis=1),np.max(GPT2_first_obj,axis=1),np.max(GPT2_eos,axis=1)]
GPT2_large = [np.max(GPT2_large_verb,axis=1),np.max(GPT2_large_first_obj,axis=1),np.max(GPT2_large_eos,axis=1)]
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

fig, axis = plt.subplots()
plot_list_LSTM = axis.bar([0,1,2,3],np.array([np.average(LSTM_verb[:,i]) for i in range(4)]),yerr =np.array([np.std(LSTM_verb[:,i])/np.sqrt(len(LSTM_verb[:,i])) for i in range(4)]),color='lightseagreen',label='LSTM')
plot_list_LSTM_large = axis.bar([4],np.average(LSTM_large_verb),yerr=np.std(LSTM_large_verb)/np.sqrt(len(LSTM_large_verb)),color='teal',label='LSTM_large')
plot_list_GPT2 = axis.bar(np.arange(5,18),np.array([np.average(GPT2_verb[:,i]) for i in range(13)]),yerr =np.array([np.std(GPT2_verb[:,i])/np.sqrt(len(GPT2_verb[:,i])) for i in range(13)]),color='salmon',label='LSTM')
plot_list_GPT2 = axis.bar(np.arange(18,55),np.array([np.average(GPT2_large_verb[:,i]) for i in range(37)]),yerr =np.array([np.std(GPT2_large_verb[:,i])/np.sqrt(len(GPT2_large_verb[:,i])) for i in range(37)]),color='crimson',label='GPT2_large')
plt.legend(bbox_to_anchor=(-0.018, 1.03), loc='upper left', borderaxespad=1,fontsize=8)
plt.show()


fig, axis = plt.subplots()
plot_list_LSTM = axis.bar([0,1,2,3],np.array([np.average(LSTM_first_obj[:,i]) for i in range(4)]),yerr =np.array([np.std(LSTM_first_obj[:,i])/np.sqrt(len(LSTM_first_obj[:,i])) for i in range(4)]),color='lightseagreen',label='LSTM')
plot_list_LSTM_large = axis.bar([4],np.average(LSTM_large_first_obj),yerr=np.std(LSTM_large_first_obj)/np.sqrt(len(LSTM_large_first_obj)),color='teal',label='LSTM_large')
plot_list_GPT2 = axis.bar(np.arange(5,18),np.array([np.average(GPT2_first_obj[:,i]) for i in range(13)]),yerr =np.array([np.std(GPT2_first_obj[:,i])/np.sqrt(len(GPT2_first_obj[:,i])) for i in range(13)]),color='salmon',label='LSTM')
plot_list_GPT2 = axis.bar(np.arange(18,55),np.array([np.average(GPT2_large_first_obj[:,i]) for i in range(37)]),yerr =np.array([np.std(GPT2_large_first_obj[:,i])/np.sqrt(len(GPT2_large_first_obj[:,i])) for i in range(37)]),color='crimson',label='GPT2_large')
plt.legend(bbox_to_anchor=(-0.018, 1.03), loc='upper left', borderaxespad=1,fontsize=8)
plt.show()

fig, axis = plt.subplots()
plot_list_LSTM = axis.bar([0,1,2,3],np.array([np.average(LSTM_eos[:,i]) for i in range(4)]),yerr =np.array([np.std(LSTM_eos[:,i])/np.sqrt(len(LSTM_eos[:,i])) for i in range(4)]),color='lightseagreen',label='LSTM')
plot_list_LSTM_large = axis.bar([4],np.average(LSTM_large_eos),yerr=np.std(LSTM_large_eos)/np.sqrt(len(LSTM_large_eos)),color='teal',label='LSTM_large')
plot_list_GPT2 = axis.bar(np.arange(5,18),np.array([np.average(GPT2_eos[:,i]) for i in range(13)]),yerr =np.array([np.std(GPT2_eos[:,i])/np.sqrt(len(GPT2_eos[:,i])) for i in range(13)]),color='salmon',label='LSTM')
plot_list_GPT2 = axis.bar(np.arange(18,55),np.array([np.average(GPT2_large_eos[:,i]) for i in range(37)]),yerr =np.array([np.std(GPT2_large_eos[:,i])/np.sqrt(len(GPT2_large_eos[:,i])) for i in range(37)]),color='crimson',label='GPT2_large')
plt.legend(bbox_to_anchor=(-0.018, 1.03), loc='upper left', borderaxespad=1,fontsize=8)
plt.show()

