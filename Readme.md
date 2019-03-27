##Artificial Intelligence (2019 Spring)

대학원 강의인 인공지능응용 강의의 소스 코드를 정리합니다. 개인적인 정리 자료로 주석은 다소 상세하지 않을 수 있습니다. 

### Homework 0
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


### Homework 1
#### House Price Data 
- data is a (506,14) array, and the number of data is 506 Each column in data (X1, X2, …, X13) are features. X14 is the house price (label)
- Training data is the first 400 data 
- Testing data is the remaining data 
- Coding two sample codes (missing part coding) Programming 1), 2)   

1) Stochastic gradient update implementation (using for loops)
2) Batch(Vectorized) gradient update implementation (without for loops)

---
**Result**