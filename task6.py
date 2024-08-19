
input_lst = [["madam", "test", "dad"], ["mine", "level"], ["bcb", "def", "ghi"]]
count_palindromes = lambda lst: list(map(lambda sublist: len(list(filter(lambda s: s == s[::-1], sublist))), lst))

result = count_palindromes(input_lst)
print(result)
