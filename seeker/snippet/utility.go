//date: 2024-12-06T16:56:49Z
//url: https://api.github.com/gists/869d01468bea275be02f9807a30d50aa
//owner: https://api.github.com/users/simpleOperator

package utility

import (
	"log/slog"
	"os"
	"strconv"
)

func NewInputReader() *os.File {
	inputReader, err := os.Open("input.txt")

	if err != nil {
		slog.Error("got an error while trying to open input file", "Error", err)
	}

	return inputReader
}

func AtoiSlice(d []string) []int {
	intSlice := make([]int, len(d))
	for i, e := range d {
		convertedElement, err := strconv.Atoi(e)
		if err != nil {
			slog.Error("error converting string to int", "error", err)
		}
		intSlice[i] = convertedElement
	}
	return intSlice
}
