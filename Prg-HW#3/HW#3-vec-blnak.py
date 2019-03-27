# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:10:58 2018

@author: IVCL
"""

import numpy as np
import mni
import matplotlib.pyplot as plt

def soft(TH,Img):
    Unhypo=np.exp(np.matmul(TH.T, Img))
    return  Unhypo/sum(Unhypo)

train = list(mni.read(dataset = "training", path = "./mnist_data"))
test = list(mni.read(dataset = "testing", path = "./mnist_data"))
trLb = []
trImg = []
teLb = []
teImg = []
for i in range(len(train)):
        trLb.append(train[i][0])
        trImg.append(train[i][1])

for i in range(len(test)):
       teLb.append(test[i][0])
       teImg.append(test[i][1]) 

m_tr = len(trLb)
m_te = len(teLb)
K=10  #number of Class
for x in range(m_tr):
    trImg[x] = np.insert(trImg[x]/255.,0,1)
for x in range(m_te):
    teImg[x] = np.insert(teImg[x]/255.,0,1)   
    
n= len(trImg[0])  

theta = np.random.random((n,K))*0.001
alpha=0.00001 #small learning rate
trainLoss=[]
testLoss=[]

trImg = np.asarray(trImg).T
teImg = np.asarray(teImg).T
     
trTarget = np.zeros((K,m_tr)) # Train Target
for i in range(m_tr):
    trTarget[trLb[i]][i] = 1.
teTarget = np.zeros((K,m_te)) # Test Target
for i in range(m_te):
    teTarget[teLb[i]][i] = 1.   
    
for k in range(200):  
    '''
       your code here (Use trTarget and teTarget here)
       
       '''
    if k % 5 == 0:
         '''
         your code here (Calculate Train cost and Test cost)
       
         '''        
         trainLoss.append(cost1/m_tr)
         testLoss.append(cost2/m_te)
plt.plot(trainLoss)       
plt.plot(testLoss)
        
correct =  sum(np.argmax(soft(theta, teImg), axis=0) ==  teLb)

print(correct / m_te)
print(cost1, cost2)     