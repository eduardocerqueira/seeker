#date: 2023-05-18T16:49:47Z
#url: https://api.github.com/gists/5b27a920e01b13540bef4e90bf19f880
#owner: https://api.github.com/users/miahj1

from tqdm import trange

num_steps = 7000
hit_rates = []

for _ in trange(1, num_steps + 1):
  selected_items = set(independent_bandits.choose())
  # Pick a users choice at random
  random_user = jester_data.sample().iloc[0, :]
  ground_truth = set(random_user[random_user == 1].index)
  hit_rate = (len(ground_truth.intersection(selected_items)) /
              len(ground_truth))
  feedback_list = [1.0 if item in ground_truth else 
                   0.0 for item in selected_items]
  independent_bandits.update(selected_items, feedback_list)
  hit_rates.append(hit_rate)