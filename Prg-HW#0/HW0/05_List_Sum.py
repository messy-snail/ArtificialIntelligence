#함수 정의
def list_sum(list):
    sum=0
    for idx in range(len(list)):
        sum+=list[idx]
    return sum

a=[1, 2, 3, 4, 5]
b=[2, 4, 6, 8, 10]

print('[사용자 정의 함수]')
print('sum of a : ', list_sum(a))
print('sum of b : ', list_sum(b))
print('[자체 함수]')
print('sum of a : ', sum(a))
print('sum of b : ', sum(b))
