//date: 2026-01-20T17:06:33Z
//url: https://api.github.com/gists/78db1fac55d2ee7dfeb14035bfc901fe
//owner: https://api.github.com/users/garsonavrilio

package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func fourPow(x int) int {
	return x * x * x * x
}

func sumFourthNonPositive(nums []int, idx int) int {
	if idx == len(nums) {
		return 0
	}
	if nums[idx] <= 0 { //  if <= 0 then it should power by 4
		return fourPow(nums[idx]) + sumFourthNonPositive(nums, idx+1)
	}
	// else ignore
	return sumFourthNonPositive(nums, idx+1)
}

func parseInts(parts []string, result []int, idx int) {
	if idx == len(parts) {
		return
	}
	result[idx], _ = strconv.Atoi(parts[idx])
	parseInts(parts, result, idx+1)
}

func processCases(scanner *bufio.Scanner, cases int, outputs *[]int) {
	if cases == 0 {
		return
	}

	scanner.Scan()
	x, _ := strconv.Atoi(strings.TrimSpace(scanner.Text()))
	// fmt.Println("==========> x: ", x)

	scanner.Scan()
	input := strings.Fields(scanner.Text())
	// fmt.Println("==========> input: ", input)
	if len(input) != x {
		*outputs = append(*outputs, -1)
		processCases(scanner, cases-1, outputs)
		return // return -1 if len parts is mismatch with the input x
	}

	nums := make([]int, x)
	parseInts(input, nums, 0)

	sum := sumFourthNonPositive(nums, 0)
	*outputs = append(*outputs, sum)

	processCases(scanner, cases-1, outputs)
}

func printResults(results []int, idx int) {
	if idx == len(results) {
		return
	}
	fmt.Println(results[idx])
	printResults(results, idx+1)
}

func main() {
	scanner := bufio.NewScanner(os.Stdin)

	scanner.Scan()
	n, _ := strconv.Atoi(strings.TrimSpace(scanner.Text()))
	// fmt.Println("========> n:", n)

	results := []int{}
	processCases(scanner, n, &results)
	printResults(results, 0)
}
