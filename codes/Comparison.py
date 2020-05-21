import pickle
import numpy as np
import scipy
import csv
import pandas as pd
from sklearn.linear_model import LinearRegression
import logging
import matplotlib.pyplot as plt
import torch
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

def sigmoid(x):
    return 1/(1+np.exp(-x))
def inv_sigmoid(x):
    return np.log((x+(1e-100))/(1-x+1e-100))

df = pd.read_csv(PATH + 'textfile/generated_pairs_behav.csv')
behav = df[['BehavDOpreference']].values/100
behav_inv = inv_sigmoid(behav)
model = df[['ratio']].values
#print(df[['BehavDOpreference','ratio']].corr(method='spearman'))


lr = LinearRegression()
X = model
Y = behav_inv
lr.fit(X, Y)

prediction = sigmoid(lr.coef_[0]*model+lr.intercept_)
plt.scatter(prediction,behav)
plt.xlabel('Model prediction')
plt.ylabel('Human ratings')
plt.xlim([0,1])
plt.show()

df_pronoun = df[df['recipient_id'].isin(['pronoun'])]
df_shortDefinite = df[df['recipient_id'].isin(['shortDefinite'])]
df_shortIndefinite = df[df['recipient_id'].isin(['shortIndefinite'])]
df_longDefinite = df[df['recipient_id'].isin(['longDefinite'])]
df_longIndefinite = df[df['recipient_id'].isin(['longIndefinite'])]
df_theme_def = df[df['theme_type'].isin(['def'])]
df_theme_indef = df[df['theme_type'].isin(['indef'])]
df_theme_something = df[df['theme_type'].isin(['something'])]


print('ALL:',scipy.stats.spearmanr(df[['BehavDOpreference','ratio']].values))
print('Pronoun:',scipy.stats.spearmanr(df_pronoun[['BehavDOpreference','ratio']].values))
print('ShortDefinite:',scipy.stats.spearmanr(df_shortDefinite[['BehavDOpreference','ratio']].values))
print('ShortIndefinite:',scipy.stats.spearmanr(df_shortIndefinite[['BehavDOpreference','ratio']].values))
print('longDefinite:',scipy.stats.spearmanr(df_longDefinite[['BehavDOpreference','ratio']].values))
print('longIndefinite:',scipy.stats.spearmanr(df_longIndefinite[['BehavDOpreference','ratio']].values))
print('DefTheme:',scipy.stats.spearmanr(df_theme_def[['BehavDOpreference','ratio']].values))
print('IndefTheme:',scipy.stats.spearmanr(df_theme_indef[['BehavDOpreference','ratio']].values))
print('SomethingTheme:',scipy.stats.spearmanr(df_theme_something[['BehavDOpreference','ratio']].values))
