//date: 2022-12-08T16:57:57Z
//url: https://api.github.com/gists/cc6f92a51192861692749df638422db7
//owner: https://api.github.com/users/shoenig

package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

type matrix [][]byte

func main() {
	r, err := os.Open(os.Args[1])
	check(err)

	var grid matrix

	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := scanner.Text()
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		row := []byte(line)
		grid = append(grid, row)
	}

	max := 0
	maxRow := 0
	maxCol := 0

	for r := 0; r < len(grid); r++ {
		for c := 0; c < len(grid[0]); c++ {
			local := score(grid, r, c)
			if local > max {
				max = local
				maxRow = r
				maxCol = c
			}
		}
	}
	fmt.Println("max", max, "row", maxRow, "col", maxCol)
}

func score(grid matrix, row, col int) int {
	north := visibleToNorth(grid, row, col)
	south := visibleToSouth(grid, row, col)
	east := visibleToEast(grid, row, col)
	west := visibleToWest(grid, row, col)
	return north * south * east * west
}

func visibleToNorth(grid matrix, row, col int) int {
	if row == 0 {
		return 0
	}
	height := grid[row][col]
	count := 0
	for r := row - 1; r >= 0; r-- {
		local := grid[r][col]
		if local < height {
			count++
		} else {
			count++
			break
		}
	}
	return count
}

func visibleToSouth(grid matrix, row, col int) int {
	if row == len(grid)-1 {
		return 0
	}
	height := grid[row][col]
	count := 0
	for r := row + 1; r <= len(grid)-1; r++ {
		local := grid[r][col]
		if local < height {
			count++
		} else {
			count++
			break
		}
	}
	return count
}

func visibleToWest(grid matrix, row, col int) int {
	if col == 0 {
		return 0
	}
	height := grid[row][col]
	count := 0
	for c := col - 1; c >= 0; c-- {
		local := grid[row][c]
		if local < height {
			count++
		} else {
			count++
			break
		}
	}
	return count
}

func visibleToEast(grid matrix, row, col int) int {
	if col == len(grid[0])-1 {
		return 0
	}
	height := grid[row][col]
	count := 0
	for c := col + 1; c <= len(grid[0])-1; c++ {
		local := grid[row][c]
		if local < height {
			count++
		} else {
			count++
			break
		}
	}
	return count
}

func check(err error) {
	if err != nil {
		fmt.Println("err", err)
		os.Exit(1)
	}
}

func ToInt(s string) int {
	i, err := strconv.Atoi(s)
	check(err)
	return i
}
