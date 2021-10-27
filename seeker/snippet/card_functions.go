//date: 2021-10-27T16:59:55Z
//url: https://api.github.com/gists/978aeb9616d44998a39a72181fe9ca14
//owner: https://api.github.com/users/tlarsen7572

func (c *Card) Rotate() {
	first := c.Parts[3]
	c.Parts[3] = c.Parts[2]
	c.Parts[2] = c.Parts[1]
	c.Parts[1] = c.Parts[0]
	c.Parts[0] = first
}

func (c *Card) MatchesRight(other *Card) bool {
	return c.Parts[left]+other.Parts[right] == 0
}

func (c *Card) MatchesBottom(other *Card) bool {
	return c.Parts[top]+other.Parts[bottom] == 0
}

func (c *Card) MatchesRightAndBottom(toLeft *Card, toTop *Card) bool {
	return c.MatchesRight(toLeft) && c.MatchesBottom(toTop)
}
