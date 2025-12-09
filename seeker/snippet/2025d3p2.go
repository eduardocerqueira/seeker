//date: 2025-12-09T16:56:17Z
//url: https://api.github.com/gists/aeb3d61733f2e04c5293266a7462c42e
//owner: https://api.github.com/users/JonHunt1995

package main

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
)

func findJuiciestBatteryBank(bank string, capacity int) (int, error) {
	digits := strings.Split(bank, "")
	if len(digits) < capacity {
		return 0, fmt.Errorf("battery bank too short")
	}
	curr, next, room := 0, 0, 0
	numDigits := len(digits)
	finalDigits := ""

	for capacity > 0 {
		next = curr + slices.Index(digits[curr:], slices.Max(digits[curr:]))
		room = numDigits - next

		// If not enough digits to the right of largest digit,
		// keep choosing the next largest until there is enough
		// room to guarantee we match the capacity length
		for room < capacity {
			next = curr + slices.Index(digits[curr:next], slices.Max(digits[curr:next]))
			room = numDigits - next
		}

		// We should have the correct number to add at this point
		finalDigits += digits[next]
		curr = next + 1
		capacity--
	}
	// Should have correct digits here
	final, err := strconv.Atoi(finalDigits)
	return final, err
}

func sumMaxJoltage(data string, capacity int) int {
	sum, largest := 0, 0
	banks := strings.Split(data, "\n")

	for _, bank := range banks {
		largest, _ = findJuiciestBatteryBank(bank, capacity)
		sum += largest
	}
	return sum
}
func main() {

	data := `987654321111111
811111111111119
234234234234278
818181911112111`

	fmt.Println(sumMaxJoltage(data, 12))
}
