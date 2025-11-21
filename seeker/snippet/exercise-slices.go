//date: 2025-11-21T16:49:42Z
//url: https://api.github.com/gists/6811a20426a8de2906c2f4ec3e36e773
//owner: https://api.github.com/users/monooso

package main

import (
	"golang.org/x/tour/pic"
)

func Pic(dx, dy int) [][]uint8 {
	rows := make([][]uint8, dx, dy)

	for rowIndex := range rows {
		cols := make([]uint8, dx, dx)

		for colIndex := range cols {
			cols[colIndex] = uint8(rowIndex ^ colIndex)
		}

		rows[rowIndex] = cols
	}

	return rows
}

func main() {
	pic.Show(Pic)
}