weight = int(input('Weight: '))
unit = input('(L)bs or (K)g: ')
if unit.upper() == "L":
    converted = weight * 0.45
    print(f"You are {converted} kilos")

else:
    converted = weight / 0.45
    print(f"You are {converted} pounds")

# 1磅(lb)=0.4535924公斤(kg) 即 1公斤(kg)=2.2046226磅(lb) 。


