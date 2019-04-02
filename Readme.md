# Artificial Intelligence (2019 Spring)

대학원 강의인 인공지능응용 강의의 소스 코드를 정리합니다. 개인적인 정리 자료로 주석은 다소 상세하지 않을 수 있습니다. 

## Homework list  
* 파이썬 기초 : [HW#0](#homework-0)
* Linear Regression : [HW#1](#homework-1)
* Logistirc Regression(Classification) : [HW#1](#homework-2)
* Softmax Regression(Classification) : [HW#1](#homework-3)

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

$\frac{1}{3}$  

\\( x(t)=\frac{-b\pm \sqrt{{b}^{2}-4ac}}{2a} \\)  

$$ \frac{\partial J(\theta)}{\partial \theta_j} = \sum_i x^{(i)}_j \left(h_\theta(x^{(i)}) - y^{(i)}\right) $$  
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

* 처리시간 비교  
  * 실험한 PC 환경은 i5-6600 CPU @ 3.30GHz, Ram 8.00GB이며, 두 코드 간의 소요시간을 비교하면 아래와 같다. 소요시간 차이는 약 48.311배가 발생하며, 이는 for문과 벡터 연산에서 발생하는 속도 차이다.
  * 관련 코드는 time 모듈을 이용하여 구현하였다.  
          
|SGD|BGD|
|:---:|:---:|
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
이중 for문을 돌면서 웨이트를 업데이트함. 화소 단위로 매번 업데이트를 수행함. 그림 단위로 해석하는 것이 아니기에 학습률이 좋지 않을 것으로 예상함.  
~~소요시간이 너무 많이 걸려 실제 실행해보지는 못함.~~
```python
for k in range(10):  # 100 is the number of epoch
    cost1=0
    cost2=0
    for i in range(m_tr):
        for j in range(n):
            hypothesis = sigmoid(trImg[i,j]*theta[j])
            grad = trImg[i,j]*(hypothesis-trLb[i])
            theta = theta-alpha*grad
            # train cost 계산
            cost1 = cost1 + sum((sigmoid(trImg[i, :] * theta) - trLb[i]) ** 2)
        # test cost 계산
        for i in range(m_te):
            cost2 = cost2 + sum((sigmoid(teImg[i, :] * theta) - teLb[i]) ** 2)

    if k % 1 == 0:
         trainLoss.append(cost1/(m_tr*n))
         testLoss.append(cost2/(m_te*n))
         print(k, cost1/(m_tr*n), cost2/(m_te*n))
```

**2. SGD - 그림 단위**    
for문을 돌면서 웨이트를 업데이트함. 그림 단위(28x28)로 매번 업데이트를 수행함. 이전과 달리 하나의 for를 이용하여 weight를 업데이트함.    

```python
for k in range(400):  # 100 is the number of epoch
    cost1=0
    cost2=0
    for i in range(m_tr):
        #의미 단위로 계산.(하나의 데이터에 대해서)
        hypothesis = sigmoid(trImg[i,:]*theta)
        grad = trImg[i,:]*(hypothesis-trLb[i])
        theta = theta-alpha*grad
        #train cost 계산
        cost1 = cost1 + sum((sigmoid(trImg[i, :] * theta) - trLb[i]) ** 2)
    # test cost 계산
    for i in range(m_te):
        cost2 = cost2 + sum((sigmoid(teImg[i, :] * theta) - teLb[i]) ** 2)

    if k % 5 == 0:
         trainLoss.append(cost1/(m_tr*n))
         testLoss.append(cost2/(m_te*n))
         print(k, cost1/(m_tr*n), cost2/(m_te*n))
```

**3. BGD**    
for문이 아닌 벡터로 연산을 수행함. Full batch 단위로 weight를 업데이트함.      

```python
for k in range(400):  # 100 is the number of epoch
    ''' 
    hypothesis = sigmoid(np.matmul(trImg.T, theta))
    grad = np.matmul(trImg, hypothesis-trLb)
    theta = theta-np.dot(alpha, grad)

    if k % 5 == 0:
        #MSE 관점.
         cost1 = sum((sigmoid(np.matmul(trImg.T, theta))-trLb)**2)
         cost2 = sum((sigmoid(np.matmul(teImg.T, theta))-teLb)**2)

         trainLoss.append(cost1/m_tr)
         testLoss.append(cost2/m_te)
         print(k, cost1 / m_tr, cost2 / m_te)
```

---
**3. Result**  
* SGD 결과(화소 단위)  
  * 너무 느려서 따로 결과를 확인하지 못함.
  
* SGD 결과(그림 단위)
  * Loss 곡선      
![mini](https://user-images.githubusercontent.com/38720524/55388003-6f61fe00-556d-11e9-8580-94d494b103d7.png)

  * **Accuracy : 0.9839243498817967**

* BGD 결과  
  * Loss 곡선  
![Figure_1](https://user-images.githubusercontent.com/38720524/55388045-80127400-556d-11e9-85c4-bb6321e13307.png)

  * **Accuracy : 0.9995271867612293**

## Homework 3
### MNIST Sotfmax classification(0 to 9) 
- Hand written image data(28x28)
- Classification of 0 ,1, …, 9 (10 Class)
- Softmax Regression algorithm

1) Batch(Vectorized) gradient update implementation (without for loops)

---
#### **Source Code**  
**1. BGD**    
* TBD  
```python
#기존 레이블을 One hot encoding으로 표현.
trTarget = np.zeros((K,m_tr)) # Train Target
for i in range(m_tr):
    trTarget[trLb[i]][i] = 1.
teTarget = np.zeros((K,m_te)) # Test Target
for i in range(m_te):
    teTarget[teLb[i]][i] = 1.   
```  
* TBD  
```python
k in range(200):
    hypothesis = soft(theta, trImg)

    grad = -np.matmul(trImg, (trTarget-hypothesis).T)
    theta = theta-alpha*grad

    if k % 5 == 0:
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
```  
* TBD  
```python
#성능 테스트를 위해 랜덤한 입력 대비 예측값 비교.
for i in range(10):
    idx = np.random.randint(0, len(teImg[0]))
    print('Ground Truth = ',teLb[idx])
    prediction = np.argmax(soft(theta, teImg[:,idx]))
    print('Prediction = ', prediction)
```  

---
**3. Result**  
* BGD 결과 
  * Loss 곡선      
  * 실제 prediction 결과 (random 10개)
