# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:51:55 2018
The data set in this home work is from
http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/frame.html
Homework-0 code for getting familiar with numpy
"""
import numpy as np
file=open('BasketballPlayer-1.txt','r')
data=file.read()  #data is a long string
data=data.split() #now a list with string elements
data=np.array(data) #now np array with string elements
data = data.astype(np.float) #now np array with float elements
data=data.reshape((54,5)) #now 2-D array
'''data is now a (54,5) nparray 54 is the number of player
Each row in data (X1, X2, X3, X4, X5) are for each player.
X1 = height in feet
X2 = weight in pounds
X3 = percent of successful field goals (out of 100 attempted)
X4 = percent of successful free throws (out of 100 attempted)
X5 = average points scored per game (the label)'''

''' Preparing Training and Test Data '''
data= np.insert(data,0, 1, axis=1)  #insert 1 for each row, now (54,6) array
np.random.shuffle(data)  #Shuffle the data (in this exercise, it is not necessary)
data=data.T # now (6,54) array
data.shape #print the dimension od data
trainX=data[0:5,0:37] #(5,37)
trainY=data[-1,0:37] #(37,)
testX=data[0:5,37:54] #(5,17)
testY=data[-1,37:54] #(17,)

