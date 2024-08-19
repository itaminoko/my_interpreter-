
fib = lambda n, a=0, b=1: a if n == 0 else fib(n - 1, b, a + b)

fib_sequence = lambda n: [fib(i) for i in range(n)]

print(fib_sequence(5))