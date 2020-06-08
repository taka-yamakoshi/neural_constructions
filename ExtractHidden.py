import pickle
import numpy as np
import torch
import torch.nn.functional as F
import csv
import sys
from ExtractHiddenModel import ExtractHiddenModel

sys.path.append('..')
args = sys.argv

#Load sentences
with open('experiment_input/generated_pairs.csv') as f:
    reader = csv.reader(f)
    file = [row for row in reader]
    head = file[0]
    text = file[1:]

sent_list_DO = [row[head.index('DOsentence')] for row in text]
sent_list_PD = [row[head.index('PDsentence')] for row in text]
verb_list = [row[head.index('DOsentence')].split(" ")[1] for row in text]
recipient_list  = [row[head.index('PDsentence')].split(" ")[-1] for row in text]
theme_list = [row[head.index('PDsentence')].split(" ")[row[head.index('PDsentence')].split(" ").index("to")-1] for row in text]

#Calculate probability
AnalysisModel = ExtractHiddenModel(args[1],args[2])
AnalysisModel.load_model()
print("Calculating DO")
DO_Hidden = AnalysisModel.extract_hidden_states(sent_list_DO,verb_list,recipient_list,theme_list)
print("Calculating PD")
PD_Hidden = AnalysisModel.extract_hidden_states(sent_list_PD,verb_list,theme_list,recipient_list)

#Dump the data
print("Dumping data")
with open('datafile/hidden_states_'+args[1]+'_'+args[2]+'_DO.pkl','wb') as f:
    pickle.dump(DO_Hidden,f)
with open('datafile/hidden_states_'+args[1]+'_'+args[2]+'_PD.pkl','wb') as f:
    pickle.dump(PD_Hidden,f)

