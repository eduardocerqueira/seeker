//date: 2021-10-27T17:05:50Z
//url: https://api.github.com/gists/2894359126d2ccddfdda4363dc997400
//owner: https://api.github.com/users/tlarsen7572

func Solve(cards []*Card) (bool, []*Card) {
	if len(cards) > 9 {
		panic(`max number of cards is 9`)
	}
	solution := make([]*Card, 9)
	for index, card := range cards {
		solution[0] = card
		newCards := removeCard(cards, index)
		if len(newCards) == 0 {
			return true, solution
		}
		rotations := 0
		for rotations < 5 {
			solved := placeNextCard(solution, 1, newCards)
			if solved {
				return true, solution
			}
			card.Rotate()
			rotations++
		}
	}
	return false, solution
}

func placeNextCard(solution []*Card, intoIndex int, cards []*Card) bool {
	var topCard *Card
	var fits func(current *Card, left *Card, top *Card) bool
	leftCard := solution[intoIndex-1]
	if intoIndex < 3 {
		fits = checkLeft
	} else {
		topCard = solution[intoIndex-3]
		if intoIndex%3 == 0 {
			fits = checkTop
		} else {
			fits = checkTopLeft
		}
	}
	for index, card := range cards {
		rotations := 0
		for rotations < 5 {
			if fits(card, leftCard, topCard) {
				solution[intoIndex] = card
				newCards := removeCard(cards, index)
				if len(newCards) == 0 {
					return true
				}
				if placeNextCard(solution, intoIndex+1, newCards) {
					return true
				}
			}
			card.Rotate()
			rotations++
		}
	}
	return false
}

func checkLeft(current *Card, left *Card, _ *Card) bool {
	return current.MatchesRight(left)
}

func checkTop(current *Card, _ *Card, top *Card) bool {
	return current.MatchesBottom(top)
}

func checkTopLeft(current *Card, left *Card, top *Card) bool {
	return current.MatchesRightAndBottom(left, top)
}
