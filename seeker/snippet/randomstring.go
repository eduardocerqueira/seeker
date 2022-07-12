//date: 2022-07-12T16:38:35Z
//url: https://api.github.com/gists/ad351114a218330a7475a9c28585a502
//owner: https://api.github.com/users/bryanpedini

import (
	"fmt"
	"math/rand"
	"time"
)

func rnd(ln int) string {
	str := ""
	for idx := 0; idx < ln; idx++ {
		rand.Seed(time.Now().UnixNano())
		ch := rand.Intn(36)
		if ch < 10 {
			ch += 48
		} else {
			ch += 55
		}
		str += string(rune(ch))
	}
	return str
}
