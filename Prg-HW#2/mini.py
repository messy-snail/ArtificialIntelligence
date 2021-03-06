# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:10:58 2018

@author: IVCL
"""

import numpy as np
import mni
import matplotlib.pyplot as plt
#시그모이드 함수 정의
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

train = list(mni.read(dataset = "training", path = "./mnist_data"))
test = list(mni.read(dataset = "testing", path = "./mnist_data"))
trLb = []
trImg = []
teLb = []
teImg = []
#Selecting data having label 0 or 1
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
#Scaling for easy of learning
for x in range(m_tr):
    trImg[x] = np.insert(trImg[x]/255.,0,1)
for x in range(m_te):
    teImg[x] = np.insert(teImg[x]/255.,0,1)

n= len(trImg[0])
# initialization of parameter vector
theta = np.random.random((n))*0.001

alpha=0.0005 #small learning rate
trainLoss=[]
testLoss=[]

trImg = np.asarray(trImg)
teImg = np.asarray(teImg)
#grad=[]
for k in range(10):  # 100 is the number of epoch
    '''
    your codes here  (stochastic update of parameters)
                   '''
    cost1=0
    cost2=0
    for i in range(m_tr):
        for j in range(n):
            hypothesis = sigmoid(trImg[i,j]*theta[j])
            grad = trImg[i,j]*(hypothesis-trLb[i])
            theta = theta-alpha*grad
            #train cost 계산
            cost1 = cost1 + sum((sigmoid(trImg[i, :] * theta) - trLb[i]) ** 2)
        # test cost 계산
        # for i in range(m_te):
        #     cost2 = cost2 + sum((sigmoid(teImg[i, :] * theta) - teLb[i]) ** 2)

    if k % 1 == 0:
         trainLoss.append(cost1/(m_tr*n))
         # testLoss.append(cost2/(m_te*n))
         # print(k, cost1/(m_tr*n), cost2/(m_te*n))
         print(k, cost1 / (m_tr * n))
# for k in range(400):  # 100 is the number of epoch
#     '''
#     your codes here  (stochastic update of parameters)
#                    '''
#     cost1=0
#     cost2=0
#     for i in range(m_tr):
#         #의미 단위로 계산.(하나의 데이터에 대해서)
#         hypothesis = sigmoid(trImg[i,:]*theta)
#         grad = trImg[i,:]*(hypothesis-trLb[i])
#         theta = theta-alpha*grad
#         #train cost 계산
#         cost1 = cost1 + sum((sigmoid(trImg[i, :] * theta) - trLb[i]) ** 2)
#     # test cost 계산
#     for i in range(m_te):
#         cost2 = cost2 + sum((sigmoid(teImg[i, :] * theta) - teLb[i]) ** 2)
#
#     if k % 5 == 0:
#          trainLoss.append(cost1/(m_tr*n))
#          testLoss.append(cost2/(m_te*n))
#          print(k, cost1/(m_tr*n), cost2/(m_te*n))
plt.plot(trainLoss, label='Train loss')
plt.plot(testLoss, label='Test loss')
plt.legend(loc='upper right')
plt.show()
correct = 0
for i in range(m_te):
    correct = correct + int((sigmoid(np.dot(theta,teImg[i])) > 0.5) == teLb[i])
print(correct / m_te)