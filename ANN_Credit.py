# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 12:51:47 2019

@author: gaura
"""
import pandas as pd
import numpy as np

df=pd.read_excel("CreditHistory_Clean.xlsx")

def my_encoder(z):
    for i in z:
        a=df[i][df[i].notnull()].unique()
        for col_name in a:
            df[i+'_'+str(col_name)]= df[i].apply(lambda x: 1 if x==col_name else 0)
            
categorical = ['checking','coapp','depends','employed','existcr','foreign','history','housing','installp','job','marital','other','property','purpose','resident','savings','telephon']
my_encoder(categorical)

df= df.drop(columns=categorical)

X= np.asarray(df.drop(columns="good_bad"))
Y= df["good_bad"]
Y= Y.map({"good":1,"bad":0})
Y = np.asarray(Y)

from sklearn.model_selection import cross_val_score
score_list=['recall','accuracy','precision','f1']
network_list = [(3),(11),(5,4),(6,5),(7,6),(8,7)]

Table = pd.DataFrame(index=range(6),columns= score_list)
from sklearn.neural_network import MLPClassifier



k=0
for nn in network_list:
    fnn= MLPClassifier(hidden_layer_sizes=nn, activation='relu', solver='lbfgs', max_iter=2000, random_state=123)
    mean_score=[]
    std_score=[]
    for s in score_list:
        fnn_4 = cross_val_score(fnn,X,Y,scoring= s, cv=4)
        mean= fnn_4.mean()
        Table.loc[k,s] = mean
    k=k+1
    
Table = Table.assign(networks= network_list)

    
    
from sklearn.model_selection import train_test_split
#splitting into train and test sets
X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)     

#Best Model = (6,5)
#Model training using the training data and displaying binary metrics for test data
from AdvancedAnalytics import NeuralNetwork

fnn_train= MLPClassifier(hidden_layer_sizes=(6,5), activation='relu', solver='lbfgs', max_iter=2000, random_state=123)
fnn = fnn_train.fit(X_train, Y_train) 
NeuralNetwork.display_binary_metrics(fnn_train, X_test, Y_test)





