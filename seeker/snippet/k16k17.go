//date: 2024-06-27T16:53:15Z
//url: https://api.github.com/gists/341ea3bc0287a02a19bcd89ee021f9a0
//owner: https://api.github.com/users/theoremoon

package main

import (
	"encoding/json"
	"fmt"
	"os"
)

func _mul(x, y uint16) uint16 {
	xx := uint32(x)
	yy := uint32(y)
	if x == 0 {
		xx = 1 << 16
	}
	if y == 0 {
		yy = 1 << 16
	}
	z := xx * yy % (1<<16 + 1)
	return uint16(z % (1 << 16))
}

func egcd(a, b int) (int, int, int) {
	if a == 0 {
		return b, 0, 1
	}
	g, x, y := egcd(b%a, a)
	return g, y - (b/a)*x, x
}

func _inv(v uint16) uint16 {
	m := 65537
	g, x, _ := egcd(int(v), m)
	if g != 1 {
		return 0
	}
	return uint16((m + x) % m)
}

func _add(x, y uint16) uint16 {
	return (x + y) & 0xffff
}

func _sub(x, y uint16) uint16 {
	return (x - y) & 0xffff
}

func main() {
	var data struct {
		Pairs [][][]uint16 `json:"pairs"`
		K18   int          `json:"k18"`
		K19   int          `json:"k19"`
		K20   int          `json:"k20"`
		K21   int          `json:"k21"`
	}
	err := json.Unmarshal([]byte(os.Args[1]), &data)
	if err != nil {
		fmt.Println(err)
		return
	}

	k18 := uint16(data.K18)
	k19 := uint16(data.K19)
	k20 := uint16(data.K20)
	k21 := uint16(data.K21)

	k18inv := _inv(k18)
	k21inv := _inv(k21)

	for k16 := 0; k16 < 65536; k16++ {
		for k17 := 0; k17 < 65536; k17++ {

			ok := true
			for _, ct := range data.Pairs {
				x1 := _mul(ct[0][0], k18inv)
				x1_ := _mul(ct[1][0], k18inv)
				x2 := _sub(ct[0][1], k19)
				x2_ := _sub(ct[1][1], k19)

				q := _sub(ct[0][1], k19) ^ _mul(ct[0][3], k21inv)
				q_ := _sub(ct[1][1], k19) ^ _mul(ct[1][3], k21inv)
				p := _mul(ct[0][0], k18inv) ^ _sub(ct[0][2], k20)

				s := _mul(p, uint16(k16))
				t := _mul(_add(q, s), uint16(k17))
				t_ := _mul(_add(q_, s), uint16(k17))

				u := _add(s, t)
				u_ := _add(s, t_)

				if !((x1^t == x1_^t_) && (x2^u == x2_^u_)) {
					ok = false
					break
				}
			}
			if ok {
				fmt.Printf("%d %d\n", k16, k17)
			}

		}
	}

}
