//date: 2023-09-27T17:03:16Z
//url: https://api.github.com/gists/d2b306da13581b97bbbcefaa5b050738
//owner: https://api.github.com/users/distractedm1nd

package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"strconv"
)

type sample struct {
	row int
	col int
}

func randomSample(k int) sample {
	return sample{row: rand.Intn(k * 2), col: rand.Intn(k * 2)}
}

type dataSquare struct {
	k             int
	withheld      bool
	matrix        [][]int
	recoveredRows map[int]bool
	recoveredCols map[int]bool
}

func createDataSquare(k int, withheld bool) dataSquare {
	matrix := make([][]int, 2*k)
	for i := range matrix {
		matrix[i] = make([]int, 2*k)
	}
	return dataSquare{k: k, matrix: matrix, withheld: withheld, recoveredRows: make(map[int]bool), recoveredCols: make(map[int]bool)}
}

func (ds *dataSquare) reset() {
	ds.recoveredRows = make(map[int]bool)
	ds.recoveredCols = make(map[int]bool)
	for m := range ds.matrix {
		for n := range ds.matrix[m] {
			ds.matrix[m][n] = 0
		}
	}
}

func (ds *dataSquare) sample(s sample) {
	row, col := s.row, s.col
	// if the share is outside of the subsquare from (0 -> k + 1), or if the share is the first index, update the matrix
	if !ds.withheld || ((row > ds.k || col > ds.k) || (row == 0 && col == 0)) {
		ds.matrix[row][col] = 1
	}
}

func (ds *dataSquare) tryRecoverRow(row int) {
	rowCount := 0
	for col := range ds.matrix[row] {
		if ds.matrix[row][col] > 0 {
			rowCount++
		}
	}
	if !ds.recoveredRows[row] && rowCount >= ds.k {
		for col := range ds.matrix[row] {
			ds.matrix[row][col] = 1
		}
		ds.recoveredRows[row] = true
	}
}

func (ds *dataSquare) tryRecoverCol(col int) {
	colCount := 0
	for row := range ds.matrix {
		if ds.matrix[row][col] > 0 {
			colCount++
		}
	}
	if !ds.recoveredCols[col] && colCount >= ds.k {
		for row := range ds.matrix {
			ds.matrix[row][col] = 1
		}
		ds.recoveredCols[col] = true
	}
}

func (ds *dataSquare) isImmediatelyRecoverable() bool {
	// fast exit
	if len(ds.recoveredRows) == len(ds.matrix) || len(ds.recoveredCols) == len(ds.matrix) {
		return true
	}

	isRecoverable := true
	for m := range ds.matrix {
		rowCount := 0
		colCount := 0
		for n := range ds.matrix[m] {
			if ds.matrix[m][n] > 0 {
				rowCount++
			}
			if ds.matrix[n][m] > 0 {
				colCount++
			}
		}
		if rowCount < ds.k || colCount < ds.k {
			isRecoverable = false
			break
		}
	}
	return isRecoverable
}

func (ds *dataSquare) recover() bool {
	// Initialize isRecoverable as false
	isRecoverable := false

	// Run the loop an appropriate amount of times to ensure recoverability
	for iter := 0; iter < len(ds.matrix); iter++ {
		// Check if every row/column is recoverable
		for i := 0; i < ds.k*2; i++ {
			ds.tryRecoverRow(i)
			ds.tryRecoverCol(i)
		}

		// Check if the square is recoverable after a few iterations
		isRecoverable = ds.isImmediatelyRecoverable()
		if isRecoverable {
			break
		}
	}
	return isRecoverable
}

func main() {
	flag.Parse()
	if len(os.Args) < 5 {
		fmt.Println("Please provide four command line arguments: k, shares, iter, and lights.")
		os.Exit(1)
	}
	k, err := strconv.Atoi(os.Args[1])
	if err != nil {
		fmt.Println("Error: k should be an integer.")
		os.Exit(1)
	}
	shares, err := strconv.Atoi(os.Args[2])
	if err != nil {
		fmt.Println("Error: shares should be an integer.")
		os.Exit(1)
	}
	iter, err := strconv.Atoi(os.Args[3])
	if err != nil {
		fmt.Println("Error: iter should be an integer.")
		os.Exit(1)
	}
	lights, err := strconv.Atoi(os.Args[4])
	if err != nil {
		fmt.Println("Error: lights should be an integer.")
		os.Exit(1)
	}

	// Create a 2k*2k matrix to represent shares
	ds := createDataSquare(k, true)

	for l := 0; l < lights; l++ {
		count := 0
		for i := 0; i < iter; i++ {
			// Reset the matrix for each iteration
			ds.reset()

			// loop over nodes
			for n := 0; n < l; n++ {
				// elems stores shares for single node
				uniqueSamples := make(map[sample]int)
				//loop until 20 uniq shares are stored
				for len(uniqueSamples) < shares {
					s := randomSample(k)
					uniqueSamples[s]++
					ds.sample(s)
				}
			}

			if ds.recover() {
				count++
			}
		}

		probability := float64(count) / float64(iter)
		fmt.Println(l, probability)
	}
}
