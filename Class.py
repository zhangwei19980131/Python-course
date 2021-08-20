class Point:
    def __init__(self, x, y):  # 初始化对象 构造函数
        self.x = x
        self.y = y

    def move(self):
        print('move')

    def draw(self):
        print('draw')


'''
point1 = Point()
point1.x = 10
point1.y = 20
point1.draw()
'''
point = Point(10, 20)
print(point.x)

print('==============')


class Person:
    def __init__(self, name):
        self.name = name

    def talk(self):
        print(f"Hi, I am {self.name}")


john = Person("john Smith")

john.talk()

bob = Person("Bob Smith")
bob.talk()