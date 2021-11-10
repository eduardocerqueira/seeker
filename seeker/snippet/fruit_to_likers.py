#date: 2021-11-10T16:51:02Z
#url: https://api.github.com/gists/5451db8533b8f61c7c62cc4ab7ea6b07
#owner: https://api.github.com/users/kaushikcfd

from kanren import (var, itero, membero, run, facts, Relation)


# equivalent python program
# --------------------------
fruit_to_its_likers = {}

for fruit, liker in [("banana", "sam"),
                     ("apple", "tom"),
                     ("mango", "nancy"),
                     ("apple", "charlie"),
                     ("banana", "david")]:
    fruit_to_its_likers.setdefault(fruit, set()).add(liker)

print(fruit_to_its_likers)

# Solution via kanren
# -------------------

all_likers = var()
fruit = var()
liker = var()

likes = Relation()
facts(likes, ("banana", "sam"))
facts(likes, ("apple", "tom"))
facts(likes, ("mango", "nancy"))
facts(likes, ("apple", "charlie"))
facts(likes, ("banana", "david"))


solns = run(2,
            (fruit, all_likers),
            itero(all_likers),
            membero(liker, all_likers),
            likes(fruit, liker))

print(solns)