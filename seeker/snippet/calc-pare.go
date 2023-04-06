//date: 2023-04-06T17:02:10Z
//url: https://api.github.com/gists/9eda28661701d8421d3a458f4567ca4b
//owner: https://api.github.com/users/michaeldavidkelley

package second

import (
	"strconv"
)

func Run() {
}

// calc("1+1") == 2
// calc("2-1+5") == 6
// calc("0-2-3-5") == -10
// calc("1+23") == 24

func Calc(input string) int {

	results := 0

	ex := []rune{}
	paraCount := 0
	for _, r := range input {
		if r == '(' {
			paraCount++
		}

		if r == ')' {
			paraCount--
		}

		if paraCount == 0 {
			//Calc()
		}
	}

	numbers := []int{}
	operators := []rune{}

	num := ""
	for _, r := range input {
		if r == '+' || r == '-' {
			operators = append(operators, r)
			n, _ := strconv.Atoi(num)
			numbers = append(numbers, n)
			num = ""
		} else {
			num = num + string(r)
		}
	}
	n, _ := strconv.Atoi(num)
	numbers = append(numbers, n)

	results = numbers[0]
	for i, o := range operators {
		switch o {
		case '+':
			results += numbers[i+1]
		case '-':
			results -= numbers[i+1]
		}
	}

	return results
}


package second_test

import (
	"ngrok/internal/second"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCalc(t *testing.T) {
	tests := []struct {
		input    string
		expected int
	}{
		{"1+1", 2},
		{"2-1+5", 6},
		{"(0-2)-(3-5)", -10},
		{"0-((2-3)-5)", -10},
		{"1+23", 24},
		{"(1+23)", 24},
		{"215468-23-00000-10", 215435},
	}

	for _, test := range tests {
		actual := second.Calc(test.input)
		//fmt.Println(actual, test.expected)

		assert.Equal(t, test.expected, actual)
	}
}
