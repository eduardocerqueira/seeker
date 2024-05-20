#date: 2024-05-20T16:42:31Z
#url: https://api.github.com/gists/af2eade592d4839c2a0db24af1f91790
#owner: https://api.github.com/users/KerimovEmil

"""
Total money = $1000
People: A (Alice), B (Bob), C (Carol)

A throws a D8 dice
B throws a D10 dice
C throws a D10 dice

Everytime A == C then A gets $1
Everytime B == C then B gets $1
Game ends when the $1000 runs out.

Question: What is the expected winnings of A and B?

Results (n=5000):
a_avg:499.4608, b_avg:499.5876, avg_num_rounds:4998.5046
a_win=2496, b_win=2496, draw=8
"""
import random


def run_simulation(total_money):
    a_money = 0
    b_money = 0
    rounds = 0

    while total_money > 1:
        rounds += 1

        a = random.randint(1, 8)
        b = random.randint(1, 10)
        c = random.randint(1, 10)

        if a == c:
            total_money -= 1
            a_money += 1
        if b == c:
            total_money -= 1
            b_money += 1
    return a_money, b_money, rounds


if __name__ == '__main__':
    a_total = 0
    b_total = 0
    round_total = 0
    n = 50
    a_win = 0
    b_win = 0
    draw = 0
    for i in range(n):
        a_earning, b_earning, n_rounds = run_simulation(total_money=1000)
        a_total += a_earning
        b_total += b_earning
        round_total += n_rounds
        if a_earning < b_earning:
            b_win += 1
        elif a_earning > b_earning:
            a_win += 1
        else:
            draw += 1
    print(f'a_avg:{a_total/n}, b_avg:{b_total/n}, avg_num_rounds:{round_total/n}')
    print(f'{a_win=}, {b_win=}, {draw=}')
