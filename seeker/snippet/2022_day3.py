#date: 2022-12-09T17:07:19Z
#url: https://api.github.com/gists/7652a1c73e5dfc5d84061eb3afd6bdb0
#owner: https://api.github.com/users/pxrpxr

#Lowercase item types a through z have priorities 1 through 26.
#Uppercase item types A through Z have priorities 27 through 52.

def test_score():
    assert 26 == find_score("z")
    assert 16 == find_score("p")
    assert 38 == find_score("L")
    assert 42 == find_score("P")
    assert 22 == find_score("v")
    assert 20 == find_score("t")
    assert 19 == find_score("s")
    assert 1 == find_score("a")
    assert 27 == find_score("A")
    assert 52 == find_score("Z")

def test_group3():

    group = [
        "vJrwpWtwJgWrhcsFMMfFFhFp",
        "jqHRNqRjqzjGDLGLrsFMfFZSrLrFZsSL",
        "PmmdzqPrVvPwwTWBwg",
    ]
    common = find_common_3(group)
    assert 'r' == common
    group = [
        "wMqvLMZHhHMvwLHjbvcjnnSBnvTQFn",
        "ttgJtRGJQctTZtZT",
        "CrZsJsPPZsGzwwsLwLmpwMDw",
    ]
    common = find_common_3(group)
    assert 'Z' == common

def find_common_3(group):
    a = group[0]
    b = group[1]
    c = group[2]
    for x in a:
        for y in b:
            for z in c:
                if x == y == z:
                    return x
    return None


def find_common(a,b):
    for x in a:
        for y in b:
            if x == y:
                return x
    return None

def find_score(c):
    if ord(c) in range(ord('a'), ord('z')+1):
        score = ord(c) - 96
        print("%s is lower case %s" % (c, score))
        return score
    score = ord(c) - 38
    print("%s is upper case: %s" % (c, score))
    return score

def play():
    items = []

    with open('../temp/day3-full.txt') as input_file:
        for r in input_file:
            r = r.rstrip()
            items.append(r)

    total_score = 0
    for item in items:
        a = item[:int(len(item)/2)]
        b = item[int(len(item)/2):]
        common = find_common(a, b)
        total_score += find_score(common)
    print(total_score)

def play_part_2():
    items = []

    with open('../temp/day3-full.txt') as input_file:
        for r in input_file:
            r = r.rstrip()
            items.append(r)

    total_score = 0
    group_counter = 0
    group_list = []
    for item in items:
        group_list.append(item)
        group_counter += 1
        if group_counter == 3:
            common = find_common_3(group_list)
            score = find_score(common)
            total_score += score
            group_counter = 0
            group_list = []
    print(total_score)


def main():
    #play()
    #test_score()
    test_group3()
    play_part_2()


if __name__ == '__main__':
    main()