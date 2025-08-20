#date: 2025-08-20T16:54:19Z
#url: https://api.github.com/gists/8633c579aa346bd11a68dbdab8df4af8
#owner: https://api.github.com/users/Dmytro-Pin

#Зробити все через функції:
# У списку цілих, заповненому випадковими числами обчислити:

# Суму від'ємних чисел;
# Суму парних чисел;
# Суму непарних чисел;
# Добуток елементів з індексами кратними 3;
# Добуток елементів між мінімальним і максимальним елементом;
# Суму елементів, що знаходяться між першим і останнім додатними елементами.

import random
start_arr=[]
for i in range(10):
    x=random.randint(-10, 10)
    start_arr.append(x)
print(start_arr)
import functions as fn

fn.neg_sum(start_arr, len(start_arr))
fn.even_sum(start_arr, len(start_arr))
fn.odd_sum(start_arr, len(start_arr))
fn.index_mult3(start_arr, len(start_arr))
fn.min_max_mult(start_arr)
fn.sum_between_positives(start_arr)

