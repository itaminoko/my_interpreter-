input_lst = [10, 3, 5, 8, 1, 14, 8, 2]

is_prime = lambda x: x > 1 and all(x % i != 0 for i in range(2, int(x ** 0.5) + 1))
get_primes_desc = lambda lst: sorted([x for x in lst if is_prime(x)], reverse=True)

result = get_primes_desc(input_lst)
print(result)
