//date: 2021-10-27T16:52:20Z
//url: https://api.github.com/gists/285744ce44a5ed518415bd69d934bfeb
//owner: https://api.github.com/users/tlarsen7572

func CreateCards() []*Card {
	return []*Card{
		{Id: 1, Parts: [4]int{RoundIslandTop, FortNiagaraBottom, SplitRockBottom, MarbleheadBottom}},
		{Id: 2, Parts: [4]int{FortNiagaraBottom, MarbleheadTop, SplitRockTop, RoundIslandTop}},
		{Id: 3, Parts: [4]int{MarbleheadTop, RoundIslandBottom, FortNiagaraTop, SplitRockTop}},
		{Id: 4, Parts: [4]int{SplitRockBottom, MarbleheadTop, RoundIslandBottom, FortNiagaraTop}},
		{Id: 5, Parts: [4]int{RoundIslandTop, SplitRockTop, MarbleheadBottom, SplitRockTop}},
		{Id: 6, Parts: [4]int{MarbleheadTop, RoundIslandTop, FortNiagaraBottom, FortNiagaraTop}},
		{Id: 7, Parts: [4]int{MarbleheadTop, SplitRockBottom, RoundIslandBottom, FortNiagaraTop}},
		{Id: 8, Parts: [4]int{RoundIslandTop, SplitRockBottom, FortNiagaraTop, MarbleheadTop}},
		{Id: 9, Parts: [4]int{MarbleheadBottom, RoundIslandTop, FortNiagaraTop, SplitRockTop}},
	}
}