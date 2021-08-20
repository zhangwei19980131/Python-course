import random

for i in range(3):
    print(random.randint(10, 20))

members = ['John', 'Mary', 'Bob', 'Mosh']
leader = random.choice(members)
print(leader)

print('===============')


class Dice:
    def roll(self):
        first = random.randint(1, 6)
        second = random.randint(1, 6)
        return first, second


dice = Dice()
print(dice.roll())


from pathlib import Path

path = Path("ecommerce")
print(path.exists())

path = Path()
for file in path.glob('*.py'):
    print(file)