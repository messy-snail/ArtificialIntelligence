# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:51:55 2018
The data set in this home work is from
http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/frame.html
Homework-0 code for getting familiar with numpy
"""
#넘파이 임포트
import numpy as np
#데이터 불러오기
file=open('BasketballPlayer-1.txt','r')
#데이터를 data 변수에 float 형으로 저장
data=file.read()  #data is a long string
data=data.split() #now a list with string elements
data=np.array(data) #now np array with string elements
data = data.astype(np.float) #now np array with float elements
#1d 2700짜리 데이터를 2d (54,5)로 바꿈.
data=data.reshape((54,5)) #now 2-D array
'''data is now a (54,5) nparray 54 is the number of player
Each row in data (X1, X2, X3, X4, X5) are for each player.
X1 = height in feet
X2 = weight in pounds
X3 = percent of successful field goals (out of 100 attempted)
X4 = percent of successful free throws (out of 100 attempted)
X5 = average points scored per game (the label)'''

''' Preparing Training and Test Data '''
#앞에 1을 추가함. axis는 추가할 방향이며, 지정하지 않을 시 1차원 배열을 반환한다.
#index가 0인 column에 1을 추가함.
data= np.insert(data,0, 1, axis=1)  #insert 1 for each row, now (54,6) array
#행을 기준으로 데이터를 섞음.
np.random.shuffle(data)  #Shuffle the data (in this exercise, it is not necessary)
#데이터 전치
data=data.T # now (6,54) array
print(data.shape) #print the dimension od data

#트레인 데이터와 테스트 데이터로 구분
#37개의 X1-X4까지의 데이터를 트레인 데이터로 지정
trainX=data[0:5,0:37] #(5,37)
#37개의 X5(label) 데이터를 트레인 레이블로 지정
trainY=data[-1,0:37] #(37,)
#17개의 X1-X4까지의 데이터를 테스트 데이터로 지정
testX=data[0:5,37:54] #(5,17)
#17개의 X5(label) 데이터를 테스트 레이블로 지정
testY=data[-1,37:54] #(17,)

