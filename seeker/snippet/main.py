#date: 2023-12-06T16:56:09Z
#url: https://api.github.com/gists/ee1c54a76aceaafbeb10534d8ffae7f4
#owner: https://api.github.com/users/joshpearce

#!/usr/bin/env python3

from aocd.models import Puzzle
from collections import defaultdict
from operator import mul

# Reading session cookie value from: "**********"
puzzle = Puzzle(year=2023, day=5)
AllowSubmitA = True
AllowSubmitB = True
UseExampleData = False

data = puzzle.examples[0].input_data if UseExampleData else puzzle.input_data

lines = data.split('\n')
seeds = list(map(int, lines[0].split(':')[1].split()))
maps = {}
mode = 'key'
_map = None
for line in lines[2:]:
    if mode == 'key':
        _map = []
        maps[line.split()[0]] = _map
        mode = 'map'
    else:
        if not line.strip():
            mode = 'key'
        else:
            ranges = list(map(int, line.split()))
            _map.append((ranges[1], ranges[1]+ranges[2]-1, ranges[0], ranges[0]+ranges[2]-1, ranges[0] - ranges[1]))

def range_int(s1, e1, s2, e2):
    if s1 <= e2:
        if s1 >= s2:
            return s1, min(e1, e2)
        elif e1 >= s2:
            return s2, min(e1, e2)

def remainder(s1, e1, s2, e2):
    rem = []
    if e2 < s1 or s2 > e1:
        rem = [(s1, e1)]
    elif s1 >= s2 and e1 <= e2:
        rem = []
    elif s2 > s1 and e2 < e1:
        rem = [(s1, s2-1), (e2+1, e1)]
    elif s2 <= s1:
        rem =  [(e2+1, e1)]
    else:
        rem =  [(s1, s2-1)]
    for x in rem:
        assert(x[1] >= x[0])
    return rem
    
maps_keys_reversed = list(maps.keys())[::-1]

end_map = maps[maps_keys_reversed[0]]

for key in maps_keys_reversed[1:]:
    cur_map = maps[key]
    pass_thrus = []
    end_map_remainders = []
    for em in end_map:
        cur_map_remainders = []
        intersections = []
        for cm in cur_map:
            isect = range_int(cm[2], cm[3], em[0], em[1])
            if isect:
                intersections.append(isect)
                pass_thru = (isect[0]-cm[4], isect[1]-cm[4], isect[0]+em[4], isect[1]+em[4], em[4]+cm[4])
                cm_remainder_src = remainder(cm[0], cm[1], pass_thru[0], pass_thru[1])
                for x in cm_remainder_src:
                    assert(x[1] >= x[0])
                pass_thrus.append(pass_thru)
                for i in range(len(cm_remainder_src)):
                    cm_remainder = (cm_remainder_src[i][0], cm_remainder_src[i][1], cm_remainder_src[i][0]+cm[4], cm_remainder_src[i][1]+cm[4], cm[4])
                    cur_map_remainders.append(cm_remainder)
            else:
                cur_map_remainders.append(cm)
            pass       


        updated_end_map = [em]
        while intersections:
            isect = intersections.pop()
            em_remainders = []
            for em in updated_end_map:
                remainders = remainder(em[0], em[1], isect[0], isect[1])
                for rem in remainders:
                    em_remainders.append((rem[0], rem[1], rem[0]+em[4], rem[1]+em[4], em[4]))
            updated_end_map = em_remainders


        for cm in cur_map:
            temp = []
            for uem in updated_end_map:
                remainders = remainder(uem[0], uem[1], cm[0], cm[1])
                for rem in remainders:
                    temp.append((rem[0], rem[1], rem[0]+uem[4], rem[1]+uem[4], uem[4]))
            updated_end_map = temp
                
        end_map_remainders += updated_end_map
        cur_map = cur_map_remainders
    for c in cur_map:
        for p in pass_thrus:
            assert(range_int(c[0], c[1], p[0], p[1]) is None)
    cur_map += pass_thrus

    for c in cur_map:
        for p in end_map_remainders:
            assert(range_int(c[0], c[1], p[0], p[1]) is None)

    maps[key] = cur_map + end_map_remainders
        
    end_map = maps[key]

    print(_map)

if not puzzle.answered_a:
    
    locs = []
    for seed in seeds:
        mapped = False
        for map in maps['seed-to-soil']:
            if map[0] <= seed and map[1] >= seed:
                locs.append(seed+map[4])
                mapped = True
                break
        if not mapped:
            locs.append(seed)

    ans_a = min(locs)

    if AllowSubmitA:
        puzzle.answer_a = ans_a
    else:
        print(f'Example A: {"Correct" if str(ans_a) == puzzle.examples[0].answer_a else "Wrong"}')

if not puzzle.answered_b:

    seed_ranges = []
    for i in range(0, len(seeds), 2):
        seed_ranges.append((seeds[i], seeds[i] + seeds[i+1]-1))

    locs = []
    for seed_range in seed_ranges:
        found_intersection = False
        for map in maps['seed-to-soil']:
            intersection = range_int(map[0], map[1], seed_range[0], seed_range[1])
            if intersection:
                found_intersection = True
                for seed in intersection:
                    mapped = False
                    if map[0] <= seed and map[1] >= seed:
                        locs.append(seed+map[4])
                        mapped = True
                        break
                    if not mapped:
                        locs.append(seed)
        if not found_intersection:
            locs += seed_range

    ans_b = min(locs)

    if AllowSubmitB:
        puzzle.answer_b = ans_b
    else:
        print(f'Example B: {"Correct" if str(ans_b) == puzzle.examples[0].answer_b else "Wrong"}')

ng"}')

