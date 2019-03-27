num=0
sum=0
list = []
#1부터 100까지 5의 배수와 6의 배수를 list에 저장
while num<99:
    num+=1
    #5의 배수이거나 6의 배수일 때
    if num%5==0 or num%6==0:
        list.append(num)
#5의 배수와 6의 배수 모두 합.
for idx in range(len(list)):
    sum+=list[idx]

#결과 출력
print('5배수와 6배수 : ', list)
print('합 결과 : ', sum)

