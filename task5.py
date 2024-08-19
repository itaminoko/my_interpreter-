
from functools import reduce

print(reduce(lambda acc, x: acc + x, map(lambda x: x ** 2, filter(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6]))))
