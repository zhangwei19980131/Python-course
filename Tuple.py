numbers = (1, 2, 3)
# numbers.__doc__  magic method  魔术方法
# numbers[0] = 10  ×  元祖不可变
print(numbers[0])

coordinates = (1, 2, 3)
x, y, z = coordinates
print(x)
print(y)
print(z)

customer = {
    "name": "John Smith",
    "age": 30,
    "is_verified": True
}  # 键值对唯一
print(customer["name"])
print(customer.get("birthdate"))
print(customer.get("birthdate", "Jan 1 1980"))

print("=============")

phone = input("Phone:")
digits_mapping = {
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four"
}
output = ""
for ch in phone:
    output += digits_mapping.get(ch, "!") + " "
print(output)
