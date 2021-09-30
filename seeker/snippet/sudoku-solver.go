//date: 2021-09-30T17:01:35Z
//url: https://api.github.com/gists/460c8c5cb20f2e5d56812324785deceb
//owner: https://api.github.com/users/linuskmr

package main

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

const (
	LENGTH = 9
	BOX_SIZE = 3
	EMPTY_CELL = 0
	CELL_MIN_VALUE = 1
	CELL_MAX_VALUE = 9
)

type Sudoku struct {
	field []uint8
}

func (s *Sudoku) indexFromXY(x, y uint8) uint8 {
	return y * LENGTH + x
}

func (s * Sudoku) validRow(y uint8) bool {
	var foundNumbers [CELL_MAX_VALUE+1]bool
	for xi := uint8(0); xi < LENGTH; xi++ {
		cell := s.field[s.indexFromXY(xi, y)]
		if cell == EMPTY_CELL {
			continue
		}
		if foundNumbers[cell] {
			return false
		}
		foundNumbers[cell] = true
	}
	return true
}

func (s * Sudoku) validColumn(x uint8) bool {
	var foundNumbers [CELL_MAX_VALUE+1]bool
	for yi := uint8(0); yi < LENGTH; yi++ {
		cell := s.field[s.indexFromXY(x, yi)]
		if cell == EMPTY_CELL {
			continue
		}
		if foundNumbers[cell] {
			return false
		}
		foundNumbers[cell] = true
	}
	return true
}

func (s * Sudoku) validBox(boxX, BoxY uint8) bool {
	var foundNumbers [CELL_MAX_VALUE+1]bool
	x := boxX * BOX_SIZE
	y := BoxY * BOX_SIZE
	for yi := y; yi < BOX_SIZE; yi++ {
		for xi := x; xi < BOX_SIZE; xi++ {
			cell := s.field[s.indexFromXY(xi, yi)]
			if cell == EMPTY_CELL {
				continue
			}
			if foundNumbers[cell] {
				return false
			}
			foundNumbers[cell] = true
		}
	}
	return true
}

func (s * Sudoku) allRowsAndColumnsValid() bool {
	for i := uint8(0); i < LENGTH; i++ {
		if !s.validRow(i) || !s.validColumn(i) {
			return false
		}
	}
	return true
}

func (s *Sudoku) allBoxesValid() bool {
	for yi := uint8(0); yi < BOX_SIZE; yi++ {
		for xi := uint8(0); xi < BOX_SIZE; xi++ {
			if !s.validBox(xi, yi) {
				return false
			}
		}
	}
	return true
}

func (s *Sudoku) valid() bool {
	return s.allRowsAndColumnsValid() && s.allBoxesValid()
}

func (s *Sudoku) firstEmptyCellIndex() int16 {
	for i := int16(0); i < int16(len(s.field)); i++ {
		if s.field[i] == EMPTY_CELL {
			return i
		}
	}
	return -1
}

func (s* Sudoku) solve() bool {
	// fmt.Println("\n----------\n")
	// fmt.Println(s)
	s.printOverwrite()
	if !s.valid() {
		// fmt.Println("Sudoku not valid")
		return false
	}

	firstEmptyCellIndex := s.firstEmptyCellIndex()
	if firstEmptyCellIndex == -1 {
		// fmt.Println("No empty cell found and sudoku is valid -> solved")
		return true
	}
	// fmt.Println("First empty cell", firstEmptyCellIndex)
	firstEmptyCell := &s.field[firstEmptyCellIndex]

	*firstEmptyCell = CELL_MIN_VALUE
	for {
		// fmt.Println("Try setting cell to", *firstEmptyCell)
		if s.solve() {
			// fmt.Println("Found value for cell so that field is valid")
			return true
		}
		if *firstEmptyCell == CELL_MAX_VALUE {
			// fmt.Println("No value for cell found so that field is valid")
			break
		}
		*firstEmptyCell++
	}
	*firstEmptyCell = EMPTY_CELL
	return false
}

func (s *Sudoku) String() string {
	output := ""
	for yi := uint8(0); yi < LENGTH; yi++ {
		for xi := uint8(0); xi < LENGTH; xi++ {
			cell := s.field[s.indexFromXY(xi, yi)]
			if cell != EMPTY_CELL {
				output += strconv.Itoa(int(cell))
			} else {
				output += "_"
			}
			output += " "
		}
		output += "\n"
	}
	return output
}

const (
	CURSOR_UP_CARRIAGE_RET = "\033[F"
)

func (s *Sudoku) printOverwrite() {
	fmt.Println(strings.Repeat(CURSOR_UP_CARRIAGE_RET, 11))
	fmt.Println(s)
	time.Sleep(5 * time.Millisecond)
}

func main() {
	s := Sudoku{
		field: []uint8{
			0, 0, 0, 0, 0, 2, 0, 0, 0,
			0, 6, 2, 0, 0, 0, 5, 0, 4,
			9, 5, 1, 7, 0, 4, 6, 2, 0,
			0, 0, 0, 4, 0, 9, 0, 8, 3,
			7, 8, 6, 0, 2, 3, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 2, 1, 6,
			5, 0, 3, 0, 8, 7, 0, 6, 0,
			0, 0, 0, 0, 0, 5, 0, 3, 7,
			2, 0, 7, 0, 1, 6, 0, 5, 0,
		},
	}
	fmt.Println(s.String())
	start := time.Now()
	solved := s.solve()
	fmt.Println("Solved?", solved)
	fmt.Println("In", time.Since(start))
	fmt.Println("Valid", s.valid())
}