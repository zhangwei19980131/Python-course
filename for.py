for item in 'Python':
    print(item)
for item in ['Mosh', 'John', 'Alice']:
    print(item)
for item in range(10):
    print(item)
for item in range(5, 10):
    print(item)
print("---------------------")
for item in range(5, 10, 2):
    print(item)
print("=====================")

prices = [10,20,30]
total = 0
for price in prices:
    total += price
print(f"Total:{total}")
