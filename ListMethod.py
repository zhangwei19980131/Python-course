numbers = [5, 2, 1, 5, 7, 4]
numbers.append(20)
print(numbers)
numbers.insert(0, 10)
print(numbers)
numbers.remove(2)
print(numbers)
numbers.pop()  # 删除最后一个
print(numbers)
# numbers.clear() 删除所有
print(numbers.index(1))
print(50 in numbers)
print(numbers.count(5))
print('===================')
print(numbers)
numbers.sort()  # 升序
print(numbers)
numbers.reverse()
print(numbers)

numbers2 = numbers.copy()
print('---------------')
# 删除列表中重复的元素
numbers1 = [2, 2, 4, 6, 3, 4, 6, 1]
uniques = []
for number in numbers1:
    if number not in uniques:
        uniques.append(number)
print(uniques)