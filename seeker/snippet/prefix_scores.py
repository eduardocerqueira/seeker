#date: 2022-10-04T17:11:18Z
#url: https://api.github.com/gists/a143a415b9e74da6dda41fa55e2866f5
#owner: https://api.github.com/users/Frazmatic

def prefix_scores(n, input_list):
    scores = []
    for i in range(n):
        prefix = input_list[:i+1]
        current_max = max(prefix)
        updated_prefix = []
        for n in prefix:
            number = n + current_max
            updated_prefix.append(number)
            current_max = max(current_max, number)
        scores.append(sum(updated_prefix))
    return scores