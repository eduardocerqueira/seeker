#date: 2024-11-21T17:12:19Z
#url: https://api.github.com/gists/0e16f88ade6d778298fdaa68073d372e
#owner: https://api.github.com/users/Fuanyi-237

def recite(start, take=1):
    verses = []
    for current in range(start, start - take, -1):
        if verses:
            verses.append("")
        verses.extend(song(current))
    return verses

def song(start):
    start_dict = {
        10: "Ten",
        9: "Nine",
        8: "Eight",
        7: "Seven",
        6: "Six",
        5: "Five",
        4: "Four",
        3: "Three",
        2: "Two",
        1: "One",
        0: "No"
    }

    #for invalid input
    if start < 0 or start > 10:
        raise ValueError("Start must be between 0 and 10")

    end = max(start - 1, 0)
    begin = start_dict.get(start)
    end_word = start_dict.get(end).lower()

    song_lines = [
        "{} green bottle{} hanging on the wall,".format(begin, '' if start == 1 else 's'),
        "{} green bottle{} hanging on the wall,".format(begin, '' if start == 1 else 's'),
        "And if one green bottle should accidentally fall,",
        "There'll be {} green bottle{} hanging on the wall.".format(end_word, '' if end == 1 else 's')
    ]
    
    return song_lines