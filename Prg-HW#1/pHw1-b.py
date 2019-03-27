# -*- coding: utf-8 -*-
"""


This exercise uses a data from the UCI repository:
 Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository
 http://archive.ics.uci.edu/ml
 Irvine, CA: University of California, School of Information and Computer Science.

Data created by:
 Harrison, D. and Rubinfeld, D.L.
 ''Hedonic prices and the demand for clean air''
 J. Environ. Economics & Management, vol.5, 81-102, 1978.
"""

import numpy as np
import matplotlib.pyplot as plt
#시간 비교를 위해 추가
import time
# Load housing data from file.
data = np.loadtxt('housing.data') #(506-data,14-input and  label) last column is label
np.random.shuffle(data) #shuffle data along axis 0
data=data.T # Transpose data()  (14,506)
#Normalize input feature of Data for easiness of Training

#데이터 0에서 1사이 값을 갖도록 normalize.
for j in range(13):
    data[j] = (data[j]-min(data[j]))/(max(data[j])-min(data[j]))
# adding 1 for all data
# 0 열에 1을 추가(더미 변수)
data=np.insert(data,0, 1, axis=0)

m=400 # number of training data
m_test = 106 # number of test data

#Trainin Data
Train_X = data[:14,:400]
Train_Y = data[14,:400]

#Test Data
Test_X = data[:14,400:]
Test_Y = data[14,400:]
#Parameter Initialization
theta = np.random.rand(14,)

#Learning rate
alpha = 0.0001
trainLoss=[]
testLoss=[]

#시작 시간
start_time = time.time()
# for k in range(2000):
#     for i in range(m):
#         # "Your Code Here"
#         xi = Train_X[:,i]
#         sum_x = 0
#         grad = 0
#         for j in range(14):
#             sum_x = sum_x + xi[j]*theta[j]
#             grad = grad+(sum_x-Train_Y[i])*xi[j]
#             theta[j] = theta[j]-alpha*grad
#     cost_train = 0  # initial cost for each epoch
#     for i in range(m):
#         # "Your Code Here"
#         cost_train = cost_train + (sum_x - Train_Y[i])*(sum_x - Train_Y[i])
#     #MSE로 표현
#     cost_train = cost_train/m
#     cost_test = 0
#     for i in range(m_test):
#         # "Your Code Here"
#         xi = Test_X[:,i]
#         sum_x = 0
#         for j in range(14):
#             sum_x = sum_x + xi[j]*theta[j]
#             cost_test = cost_test + (sum_x - Test_Y[i]) * (sum_x - Test_Y[i])
#     #MSE로 표현
#     cost_test = cost_test/m_test
#
#     if k % 10 == 0:
#          trainLoss.append(cost_train/m)
#          testLoss.append(cost_test/m_test)
#          print("[step: {}] Cost_train: {}".format(k, cost_train))
for k in range(2000):
    grad = 0
    grad = grad + np.dot(Train_X, (np.dot(theta, Train_X) - Train_Y))
    theta = theta - alpha * grad
    cost_train = 0  # initial cost for each epoch
    # MSE로 표현
    cost_train = np.mean((np.dot(theta, Train_X) - Train_Y) * (np.dot(theta, Train_X) - Train_Y))
    cost_test = 0
    #MSE로 표현
    cost_test = np.mean((np.dot(theta, Test_X) - Test_Y) * (np.dot(theta, Test_X) - Test_Y))

    if k % 10 == 0:
        trainLoss.append(cost_train / m)
        testLoss.append(cost_test / m_test)
        print("[step: {}] Cost_train: {}".format(k, cost_train))
print("소요시간 : %s sec" %(time.time() - start_time))
print(cost_train)  #printing final train loss
plt.figure(0)
#각 그래프에 레이블을 할당함.
plt.plot(trainLoss, label='train loss')
plt.plot(testLoss, label='test loss')
#레전드를 통해 그래프를 구분함.
plt.legend(loc='upper left')
plt.show() #show가 없어서 추가
print(np.argmin(testLoss))   
 
pred=[]
for i in range(m_test):
    pred.append(np.dot(theta, Test_X[:,i])) 
pred=np.asarray(pred) #converting list to numpy array
ind=np.argsort(Test_Y)
plt.figure(1)
plt.plot(pred[ind],'*', label='prediction')
plt.plot(Test_Y[ind],'.', label='test data')
plt.legend(loc='upper left')
plt.show() #show가 없어서 추가

pred=[]
for i in range(m):
    pred.append(np.dot(theta, Train_X[:,i])) 
pred=np.asarray(pred) #converting list to numpy array
ind=np.argsort(Train_Y)

plt.figure(2)
plt.plot(pred[ind],'*', label='prediction')
plt.plot(Train_Y[ind],'.', label='train data')
plt.legend(loc='upper left')
plt.show() #show가 없어서 추가