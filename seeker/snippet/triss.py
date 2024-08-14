#date: 2024-08-14T19:08:09Z
#url: https://api.github.com/gists/068fc7c89a7db059e5b9ef3e60f4e01a
#owner: https://api.github.com/users/wlinds

import numpy as np

# https://www.svenskaspel.se/triss/spelguide/triss-30
prizes = np.array([
    (1, 2765000),
    (4, 300000),
    (6, 265000),
    (2, 100000),
    (3, 50000),
    (8, 20000),
    (46, 10000),
    (30, 5000),
    (15, 2500),
    (50, 2000),
    (80, 1500),
    (160, 1000),
    (60, 900),
    (50, 750),
    (200, 600),
    (200, 500),
    (125, 450),
    (930, 300),
    (1200, 180),
    (3760, 150),
    (7200, 120),
    (26000, 90),
    (200355, 60),
    (188515, 30)
])

ticket_wins  = 429_000
ticket_total = 2_000_000
ticket_price = 30
simulations  = 1000
budget = 400

ticket_list = np.hstack([np.full(units, prize) for units, prize in prizes])
ticket_list = np.hstack([ticket_list, np.zeros(ticket_total - ticket_wins)])
np.random.shuffle(ticket_list)

def one_epoch(size):
    tickets = np.random.choice(ticket_list, size, replace=False)
    sum_win = tickets.sum()
    return sum_win

results = np.array([one_epoch(budget // ticket_price) for _ in range(simulations)])
print(f"{'='*9} Results {'='*9} \nAverage win: {results.mean()} SEK")
print(f"Highest win: {results.max()} SEK\n")
print(f"Average gain/loss: {results.mean()-budget} SEK\n")
print(f"Top 10 wins out of {simulations} simulations: \n {np.sort(results)[-10:]}")