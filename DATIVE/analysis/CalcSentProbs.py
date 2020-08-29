import pickle
import numpy as np
import torch
import torch.nn.functional as F
import csv
import sys
sys.path.append('../..')
from Models.CalcSentProbsModel import CalcSentProbsModel

args = sys.argv

#Load sentences
with open('../data/generated_pairs.csv') as f:
    reader = csv.reader(f)
    file = [row for row in reader]
    head = file[0]
    text = file[1:]

sent_list_DO = [row[head.index('DOsentence')] for row in text]
sent_list_PD = [row[head.index('PDsentence')] for row in text]


#Calculate probability
AnalysisModel = CalcSentProbsModel(args[1])
AnalysisModel.load_model()
print("Calculating DO")
DO_prob = AnalysisModel.calculate_sent_probs(sent_list_DO)
print("Calculating PD")
PD_prob = AnalysisModel.calculate_sent_probs(sent_list_PD)
ratio = DO_prob - PD_prob

#Dump the data
print("Dumping data")
with open(f'../data/{args[1]}_DO.pkl','wb') as f:
    pickle.dump(DO_prob,f)
with open(f'../data/{args[1]}_PD.pkl','wb') as f:
    pickle.dump(PD_prob,f)
with open(f'../data/{args[1]}_ratio.pkl','wb') as f:
    pickle.dump(ratio,f)

with open('../data/generated_pairs_with_results.csv') as f:
    reader = csv.reader(f)
    file = [row for row in reader]
    head = file[0]
    text = file[1:]

with open('../data/generated_pairs_with_results.csv','w') as f:
    if f'{args[1]}_ratio' in head:
        writer = csv.writer(f)
        writer.writerow(head)
        for i,row in enumerate(text):
            row[head.index(f'{args[1]}_ratio')] = ratio[i]
            writer.writerow(row)
    else:
        writer = csv.writer(f)
        head.append(f'{args[1]}_ratio')
        writer.writerow(head)
        for i,row in enumerate(text):
            row.append(ratio[i])
            writer.writerow(row)
