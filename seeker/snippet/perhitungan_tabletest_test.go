//date: 2022-03-16T17:01:56Z
//url: https://api.github.com/gists/08eba875caea957d8e83c0edf2fc7206
//owner: https://api.github.com/users/muhammadyusuf22

package helper

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPerhitunganTableTest(t *testing.T) {
	// Membuat Data Slice Struct dengan beberapa properti yang kita butuhkan untuk parameter pengujian
	datas := []struct {
		name string
		nilaiA   int
		nilaiB   int
		operator string
		expected int
	}{
		{
			name: "KALI",
			nilaiA:   10,
			nilaiB:   2,
			operator: "*",
			expected: 20,
		},
		{
			name: "BAGI",
			nilaiA:   50,
			nilaiB:   10,
			operator: ":",
			expected: 5,
		},
		{
			name: "TAMBAH",
			nilaiA:   10,
			nilaiB:   10,
			operator: "+",
			expected: 20,
		},
		{
			name: "KURANG",
			nilaiA:   20,
			nilaiB:   10,
			operator: "-",
			expected: 10,
		},
	}

	// Dari Data Slice Struct Di atas kita lakukan perulangan
	// Untuk dilakukan pengujian dari setiap data di dalamnya
	for _, data := range datas {
		t.Run(data.name, func(t *testing.T) {
			hasil, _ := PerhitunganDuaBilangan(data.nilaiA, data.nilaiB, data.operator)
			assert.Equal(t, data.expected, hasil)
		})
	}

}
