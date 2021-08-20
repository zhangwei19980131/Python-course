def greet_user(first_name, last_name):
    print(f'Hi {first_name} {last_name}!')
    print('Welcome aboard')


print("Start")
greet_user("John", "Smith")
greet_user("Mary", "Alice")
greet_user(last_name="Smith", first_name="John")
greet_user("John", last_name="Smith")
# greet_user(first_name="John", "Smith") 错误
# 同时使用位置参数和关键字参数，使用的关键字参数应该在位置参数之后
print("Finish")

print('=======================')


def square(number):
    return number * number


result = square(3)
print(result)
