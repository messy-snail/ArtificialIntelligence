# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:10:58 2018

@author: IVCL
"""

import numpy as np
import mni
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

train = list(mni.read(dataset = "training", path = "./mnist_data"))
test = list(mni.read(dataset = "testing", path = "./mnist_data"))
trLb = []
trImg = []
teLb = []
teImg = []
for i in range(len(train)):
    if (train[i][0] == 0) or (train[i][0] == 1):
        trLb.append(train[i][0])
        trImg.append(train[i][1])

for i in range(len(test)):
    if (test[i][0] == 0) or (test[i][0] == 1):    
       teLb.append(test[i][0])
       teImg.append(test[i][1]) 
    
print(trLb[100])
mni.show(trImg[100])    


m_tr = len(trLb)
m_te = len(teLb)

for x in range(m_tr):
    trImg[x] = np.insert(trImg[x]/255.,0,1)
for x in range(m_te):
    teImg[x] = np.insert(teImg[x]/255.,0,1)    
    
n= len(trImg[0])  

theta = np.random.random((n))*0.001

alpha=0.0005 #small learning rate
trainLoss=[]
testLoss=[]

trImg = np.asarray(trImg).T
teImg = np.asarray(teImg).T      
for k in range(400):  # 100 is the number of epoch
    ''' 
    your code here (vectorized implementation)
    '''
    if k % 5 == 0:
         '''
         your code here (Calculate Train cost and Test cost)
       
         '''                
         trainLoss.append(cost1/m_tr)
         testLoss.append(cost2/m_te)
plt.plot(trainLoss)       
plt.plot(testLoss)
correct =  sum((sigmoid(np.matmul(teImg.T, theta)) > 0.5) == teLb)
print(correct / m_te)         