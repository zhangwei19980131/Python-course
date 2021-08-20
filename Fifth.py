import math

x = 2.9
print(round(x))
print(abs(-2.9))
print(math.ceil(x))
print(math.floor(x))

is_hot = False
is_cold = False
if is_hot:
    print("It's a hot day")
    print("Drink plenty of water")
elif is_cold:
    print("It's a cold day")
    print("Wear warm clothes")
else:
    print("It's a lovely day")
print("Enjoy your day")

house_price = 1000000
is_goodCredit = True
if is_goodCredit:
    down_payment = 0.1 * house_price
else:
    down_payment = 0.2 * house_price
print(f"Down payment: ${down_payment}")