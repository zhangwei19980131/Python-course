try:
    age = int(input('Age:'))
    income = 200000
    risk = income / age
    print(age)
except ZeroDivisionError:
    print("Age can not be zero!")
except ValueError:
    print('Invalid value')
