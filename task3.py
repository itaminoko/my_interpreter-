from functools import reduce

lst = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
def cumulative_even_squares(lst):
    return list(map(
        lambda sublist: reduce(
            lambda acc, num: acc + num,
            map(
                lambda x: x ** 2,
                filter(
                    lambda x: x % 2 == 0,
                    sublist
                )
            ),
            0
        ),
        lst
    ))


result = cumulative_even_squares(lst)
print(result)
