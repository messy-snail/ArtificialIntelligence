sample1= {'kiwi':30, 'apple': 20, 'pineapple': 50 }
sample2= {'apple': 20, 'banana': 15, 'pear': 20, 'grape':40}

print('[병합 전]')
print(sample1)
print(sample2)

#update를 통해 sample1과 sample2 합침.
sample1.update(sample2)
print('[병합 후]')
print(sample1)