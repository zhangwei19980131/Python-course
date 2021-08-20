first = 'John'
last = 'Smith'
message = first + ' [' + last + '] is a coder'
msg = f'{first} [{last}] is a coder'  # f “{表达式}”是用于格式化输出的
# #Python中的f字符串的用法：
# 要在字符串中插入变量的值，
# 可在前引号前加上字母f，
# 再将要插入的变量放在花括号内。
print(message)
print(msg)

course = 'Python for beginners'
print(len(course))  # 空格也算
print(course.upper())
print(course.lower())
a = course.find('P')  # find 方法区分大小写  返回的是index of character or sequence （字符或多个字符的序列）
print(a)
b = course.find('o')
print(b)
print(course.replace('beginners', 'absolute beginners'))
print('for ' in course)  # in 返回的是bool类型的值
print(course.title())  # 返回"标题化"的字符串,就是说所有单词都是以大写开始，其余字母均为小写
print(10 / 3)
print(10 // 3)
print(10 % 3)
print(10 ** 3)

x = 3
x += 3
print(x)
