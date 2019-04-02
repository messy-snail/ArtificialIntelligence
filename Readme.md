# Artificial Intelligence (2019 Spring)

대학원 강의인 인공지능응용 강의의 소스 코드를 정리합니다. 개인적인 정리 자료로 주석은 다소 상세하지 않을 수 있습니다. 

## Homework 0
1. 1에서 100까지를더하는프로그램 (for 또는while loop 사용)  
2. 다음과같이 tuple을 element로갖는 list가 주어진 경우 tuple 인수값의 합이 증가하는대로 list의 element의 순서를 재배치 하는 프로그램 작성
sample list : [ (2,5), (1,2), (4,4), (2,3), (2,1) ]  
one correct result: [ (2,1), (1,2), (2,3), (2,5), (4,4) ]  
3. 1에서 100까지의 숫자 중 5의 배수와 6의 배수를 모두 찾고 이들의 합을 구하는 프로그램
4. 어떤 두 개의 dictionary를 통합(merge)하는 프로그램을 작성하시오. 
sample1: { kiwi: 30, apple: 20, pineapple: 50 }  
sample2: { apple: 20, banana: 15, pear: 20, grape: 40 }  
5. 함수(function)을사용하여 어떤 list에 있는 모든 값을 더하는 프로그램을 작성하시오.
6. 첨부하는BasketballPlayer-1.txt를 읽어들여 Data를 가공하는 Hw0.py 프로그램을 실행해보고 각 프로그램 라인들의 역할을 설명하시오.

---
**Result**
1) 1에서 100까지 합 :  5050
2) sample list :  [(2, 5), (1, 2), (4, 4), (2, 3), (2, 1)]  
result :  [(1, 2), (2, 1), (2, 3), (2, 5), (4, 4)]  
3) 5배수와 6배수 :  [5, 6, 10, 12, 15, 18, 20, 24, 25, 30, 35, 36, 40, 42, 45, 48, 50, 54, 55, 60, 65, 66, 70, 72, 75, 78, 80, 84, 85, 90, 95, 96]  
합 결과 :  1586
4) [병합 전]  
{'kiwi': 30, 'apple': 20, 'pineapple': 50}  
{'apple': 20, 'banana': 15, 'pear': 20, 'grape': 40}  
[병합 후]  
{'kiwi': 30, 'apple': 20, 'pineapple': 50, 'banana': 15, 'pear': 20, 'grape': 40}  
5) [사용자 정의 함수]  
sum of a :  15  
sum of b :  30  
[자체 함수]  
sum of a :  15  
sum of b :  30  


## Homework 1
### House Price Data 
- data is a (506,14) array, and the number of data is 506 Each column in data (X1, X2, …, X13) are features. X14 is the house price (label)
- Training data is the first 400 data 
- Testing data is the remaining data 
- Coding two sample codes (missing part coding) Programming 1), 2)   

1) Stochastic gradient update implementation (using for loops)
2) Batch(Vectorized) gradient update implementation (without for loops)

---
#### **Source Code**  
**1. SGD**    
for문을 돌면서 웨이트를 업데이트함. batch와 달리 매번 업데이트를 수행함.
```python
for k in range(2000):
    for i in range(m):
        # "Your Code Here"
        xi = Train_X[:,i]
        sum_x = 0
        grad = 0
        for j in range(14):
            sum_x = sum_x + xi[j]*theta[j]
            grad = grad+(sum_x-Train_Y[i])*xi[j]
            theta[j] = theta[j]-alpha*grad
    cost_train = 0  # initial cost for each epoch
    for i in range(m):
        # "Your Code Here"
        cost_train = cost_train + (sum_x - Train_Y[i])*(sum_x - Train_Y[i])
    #MSE로 표현
    cost_train = cost_train/m
    cost_test = 0
    for i in range(m_test):
        # "Your Code Here"
        xi = Test_X[:,i]
        sum_x = 0
        for j in range(14):
            sum_x = sum_x + xi[j]*theta[j]
            cost_test = cost_test + (sum_x - Test_Y[i]) * (sum_x - Test_Y[i])
    #MSE로 표현
    cost_test = cost_test/m_test
```
**2. BGD**  
numpy를 활용하여 웨이트를 batch 단위로 업데이트함. 여기서는 mini batch가 아닌 full batch를 사용함.

```python
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
```
---
**3. Result**  
* SGD 결과  
  * Loss 곡선  
![Figure_0](https://user-images.githubusercontent.com/38720524/55375153-03b86a80-5546-11e9-8ed4-fe6e0930ac26.png)

  * Regression 결과(Test)  
![Figure_1](https://user-images.githubusercontent.com/38720524/55375178-34989f80-5546-11e9-8d48-bb18ad51ed1d.png)

  * Regression 결과(Training)  
![Figure_2](https://user-images.githubusercontent.com/38720524/55375201-509c4100-5546-11e9-974d-17ca6f7d8360.png)

* BGD 결과  
  * Loss 곡선  
![Figure_0](https://user-images.githubusercontent.com/38720524/55375239-6c9fe280-5546-11e9-8a9a-483cf66bef70.png)

  * Regression 결과(Test)  
![Figure_1](https://user-images.githubusercontent.com/38720524/55375240-6c9fe280-5546-11e9-8f24-4e575059b23a.png)

  * Regression 결과(Training)   
![Figure_2](https://user-images.githubusercontent.com/38720524/55375241-6d387900-5546-11e9-8340-ab88ddbc9da7.png)

* BGD 결과  
  * 실험한 PC 환경은 i5-6600 CPU @ 3.30GHz, Ram 8.00GB이며, 두 코드 간의 소요시간을 비교하면 아래와 같다. 소요시간 차이는 약 48.311배가 발생하며, 이는 for문과 벡터 연산에서 발생하는 속도 차이다.
  * 관련 코드는 time 모듈을 이용하여 구현하였다.      
|SGD|BGD|
|:--------:|:-------:|
|18.020 sec|0.373 sec|

## Homework 2
### MNIST Logistic classification(0 or 1) 
- Hand written image data(28x28)
- Classification of 0 and 1
- Logistic Regression algorithm

1) Stochastic gradient update implementation (using for loops)
2) Batch(Vectorized) gradient update implementation (without for loops)

**[주의사항]**  
for 루프를 이용한 구현에서는 총 두 가지 방법으로 구현되었다. 화소 하나 당 weight를 업데이트 하는 방법과 그림 하나 당(28x28) weight를 업데이트 하는 방법으로 구현되었다.

---
#### **Source Code**  
**1. SGD - 화소 단위**    
for문을 돌면서 웨이트를 업데이트함. batch와 달리 매번 업데이트를 수행함.