from functools import reduce

delimiter = '&&'
str_lst = ['123', '45', '6789', '0']

concat_strings = lambda lst: reduce(lambda x, y: x + delimiter + y, lst)

print(concat_strings(str_lst))
