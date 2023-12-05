#date: 2023-12-05T17:05:16Z
#url: https://api.github.com/gists/3793d09d23936c976965662ec2ae9b36
#owner: https://api.github.com/users/luqmansen

@dataclass
class Deck:
    id: int
    winning: Set[int]
    mine: Set[int]

    def count_point(self) -> int:
        mine_winning = self.winning.intersection(self.mine)
        num_of_match = len(mine_winning)
        if num_of_match <= 1:
            return num_of_match
        else:
            return 2**(num_of_match-1)


def parse_raw_cards(raw_cards: str) -> List[int]:
    return [
        (int(c)) for c
        in raw_cards.strip().split(' ')
        if c != ''
    ]


def solve():
    decks = []

    for line in _input.splitlines():
        deck, card = line.split(':')
        deck_id = deck.split(' ')[-1]
        raw_winning, raw_mine = card.split('|')
        decks.append(
            Deck(
                id=int(deck_id),
                winning=set(parse_raw_cards(raw_winning)),
                mine=set(parse_raw_cards(raw_mine))
            )
        )

    print(sum([deck.count_point() for deck in decks]))


if __name__ == '__main__':
    solve()
