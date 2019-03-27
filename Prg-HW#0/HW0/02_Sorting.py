list = [ (2,5), (1,2), (4,4), (2,3), (2,1)]
#sorted 함수를 이용하여 정렬.
result = sorted(list, key=lambda x:sum(x))
print('sample list : ', list)
print('result : ', result)

