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

#Normalization
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

#기존 레이블을 One hot encoding으로 표현.
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
    #x*(label-softmax)
    #1-hot encoding
    #hypothesis = np.argmax(soft(theta, trImg), axis=0)
    hypothesis = soft(theta, trImg)
    #hypothesis = soft(theta, trImg)

    grad = -np.matmul(trImg, (trTarget-hypothesis).T)
    theta = theta-alpha*grad

    if k % 5 == 0:
         '''
         your code here (Calculate Train cost and Test cost)
       
         '''
         #MSE 관점
         cost1=np.sum((trTarget-soft(theta, trImg))**2)
         cost2 = np.sum((teTarget - soft(theta, teImg)) ** 2)
         trainLoss.append(cost1/m_tr)
         testLoss.append(cost2/m_te)
         print(k, cost1/m_tr, cost2/m_te)

plt.plot(trainLoss, label='Train loss')
plt.plot(testLoss, label='Test loss')
plt.legend(loc='upper right')
plt.show()
correct =  sum(np.argmax(soft(theta, teImg), axis=0) ==  teLb)
print(correct / m_te)

print('-----Learning Finished-----')

#성능 테스트를 위해 랜덤한 입력 대비 예측값 비교.
for i in range(10):
    idx = np.random.randint(0, len(teImg[0]))
    print('Ground Truth = ',teLb[idx])
    prediction = np.argmax(soft(theta, teImg[:,idx]))
    print('Prediction = ', prediction)

