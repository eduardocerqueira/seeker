//date: 2024-05-10T16:57:21Z
//url: https://api.github.com/gists/48ac016e3f85eb767e5bbe3bb9405da6
//owner: https://api.github.com/users/h5law

// Package main provides numerous implementations of functions for calculating
// the nth Fibonacci number - detailing their time and space complexities as
// well as their limitations on how large of n they can actually compute.
// Playground link: https://go.dev/play/p/mW_-C9mRr5u
package main

import (
	"fmt"
	"math"
	"math/big"
	"time"
)

// We are focussed on the tradition Fibonacci Sequence: 0, 1, 1, 2, 3, 5, 8, ...
// Defined as the recurrence relation: F(n) = F(n-1) + F(n-2) and the nth
// Fibonacci number can be represented using Binet's formula:
// F(n) = (Phi^n - Psi^n) / sqrt(5) - this can be used to determine the
// maximum n a function can compute or be used as a constant time approximation
// for n <= 2^64 - 1 (using a uint64).
func main() {
	fmt.Printf("fibRecursive(25): \t\t\t")
	start := time.Now()
	res := fibRecursive(25)
	elapsed := time.Since(start)
	fmt.Printf("%d (%v) // slow\n", res, elapsed)
	start = time.Now()
	elapsed = time.Since(start)
	fmt.Printf("fibMemoMap(91): \t\t\t")
	start = time.Now()
	res = fibMemoMap(91)
	elapsed = time.Since(start)
	fmt.Printf("%d (%v)\n", res, elapsed)
	fmt.Printf("fibMemoSlice(91): \t\t\t")
	start = time.Now()
	res = fibMemoSlice(91)
	elapsed = time.Since(start)
	fmt.Printf("%d (%v)\n", res, elapsed)
	fmt.Printf("fibIterativeArray(91): \t\t\t")
	start = time.Now()
	res = fibIterativeArray(91)
	elapsed = time.Since(start)
	fmt.Printf("%d (%v)\n", res, elapsed)
	fmt.Printf("fibIterativeVars(91): \t\t\t")
	start = time.Now()
	res = fibIterativeVars(91)
	elapsed = time.Since(start)
	fmt.Printf("%d (%v)\n", res, elapsed)
	fmt.Printf("fibIterativeArrayChecked(93): \t\t")
	start = time.Now()
	res2 := fibIterativeArrayChecked(93)
	elapsed = time.Since(start)
	fmt.Printf("%d (%v)\n", res2, elapsed)
	fmt.Printf("fibIterativeVarsChecked(93): \t\t")
	start = time.Now()
	res2 = fibIterativeArrayChecked(93)
	elapsed = time.Since(start)
	fmt.Printf("%d (%v)\n", res2, elapsed)
	fmt.Printf("fibApproximation(93): \t\t\t")
	start = time.Now()
	res2 = fibApproximation(93)
	elapsed = time.Since(start)
	fmt.Printf("%d (%v)\n", res2, elapsed)
	fmt.Printf("fibIterativeVarsUnlimited(500): \t")
	start = time.Now()
	res3 := fibIterativeVarsUnlimited(big.NewInt(500))
	elapsed = time.Since(start)
	fmt.Printf("%d (%v)\n", res3, elapsed)
	fmt.Printf("fibIterativeVarsUnlimitedChecked(500): \t")
	start = time.Now()
	res3 = fibIterativeVarsUnlimitedChecked(big.NewInt(500))
	elapsed = time.Since(start)
	fmt.Printf("%d (%v)\n", res3, elapsed)
}

// Time: ~O(2^n), Space: O(1)
// Fails at n~=63log_1.618(2)+1/2log_1.618(5)~=100 (1.618 approximate for Phi)
// This is due to the maximum value an int can store - after this value for n
// the int value will overflow and won't be able to continue computing further.
func fibRecursive(n int) int {
	if n < 2 {
		return n
	}
	return fibRecursive(n-1) + fibRecursive(n-2)
}

var cacheMap = make(map[int]int, 100)

// Time: O(n), Space: O(n)
// Fails at n~=63log_1.618(2)+1/2log_1.618(5)~=100 (1.618 approximate for Phi)
// This is due to the maximum value an int can store - after this value for n
// the int value will overflow and won't be able to continue computing further.
func fibMemoMap(n int) int {
	if n < 2 {
		return n
	}
	if _, ok := cacheMap[n]; !ok {
		cacheMap[n] = fibMemoMap(n-1) + fibMemoMap(n-2)
	}
	return cacheMap[n]
}

var cacheSlice = make([]int, 101)

// Time: O(n), Space: O(n)
// Fails at n~=63log_1.618(2)+1/2log_1.618(5)~=100 (1.618 approximate for Phi)
// This is due to the maximum value an int can store - after this value for n
// the int value will overflow and won't be able to continue computing further.
func fibMemoSlice(n int) int {
	if n < 2 {
		return n
	}
	if cacheSlice[n] == 0 {
		cacheSlice[n] = fibMemoSlice(n-1) + fibMemoSlice(n-2)
	}
	return cacheSlice[n]
}

// Time: O(n), Space: O(1)
// Fails at n~=63log_1.618(2)+1/2log_1.618(5)~=100 (1.618 approximate for Phi)
// This is due to the maximum value an int can store - after this value for n
// the int value will overflow and won't be able to continue computing further.
// Returns n if n < 0
func fibIterativeArray(n int) int {
	if n < 2 {
		return n
	}
	arr := [2]int{0, 1}
	for i := 2; i <= n; i++ {
		tmp := arr[1]
		arr[1] = arr[0] + arr[1]
		arr[0] = tmp
	}
	return arr[1]
}

// Time: O(n), Space: O(1)
// Fails at n~=63log_1.618(2)+1/2log_1.618(5)~=100 (1.618 approximate for Phi)
// This is due to the maximum value a uint64 can store - after this value for n
// the uint64 value will overflow and won't be able to continue computing further.
func fibIterativeArrayChecked(n uint64) uint64 {
	if n < 2 {
		return n
	}
	arr := [2]uint64{0, 1}
	for i := uint64(2); i <= n; i++ {
		tmp := arr[1]
		arr[1] = arr[0] + arr[1]
		arr[0] = tmp
	}
	return arr[1]
}

// Time: O(n), Space: O(1)
// Fails at n~=63log_1.618(2)+1/2log_1.618(5)~=100 (1.618 approximate for Phi)
// This is due to the maximum value an int can store - after this value for n
// the int value will overflow and won't be able to continue computing further.
// Returns n if n < 0
func fibIterativeVars(n int) int {
	if n < 2 {
		return n
	}
	a, b := 0, 1
	for i := 2; i <= n; i++ {
		a, b = b, a+b
	}
	return b
}

// Time: O(n), Space: O(1)
// Fails at n~=64log_1.618(2)+1/2log_1.618(5)~=100 (1.618 approximate for Phi)
// This is due to the maximum value a uint64 can store - after this value for n
// the uint64 value will overflow and won't be able to continue computing further.
func fibIterativeVarsChecked(n uint64) uint64 {
	if n < 2 {
		return n
	}
	var a, b uint64 = 0, 1
	for i := uint64(2); i <= n; i++ {
		a, b = b, a+b
	}
	return b
}

// Time: O(n), Space: O(n)
// Works indefinitely for any n > 0, returns n if n < 0
func fibIterativeVarsUnlimited(n *big.Int) *big.Int {
	if n.Cmp(big.NewInt(2)) == -1 {
		return n
	}
	a, b := big.NewInt(0), big.NewInt(1)
	for i := big.NewInt(2); i.Cmp(n) < 1; i = i.Add(i, big.NewInt(1)) {
		a, b = b, a.Add(a, b)
	}
	return b
}

// Time: O(n), Space: O(n)
// Works indefinitely for any n > 0, returns nil when n < 0
func fibIterativeVarsUnlimitedChecked(n *big.Int) *big.Int {
	if n.Cmp(big.NewInt((0))) == -1 {
		return nil
	}
	if n.Cmp(big.NewInt(2)) == -1 {
		return n
	}
	a, b := big.NewInt(0), big.NewInt(1)
	for i := big.NewInt(2); i.Cmp(n) < 1; i = i.Add(i, big.NewInt(1)) {
		a, b = b, a.Add(a, b)
	}
	return b
}

// Time: O(1), Space: O(1)
// Fails at n~=64log_1.618(2)+1/2log_1.618(5)~=100 (1.618 approximate for Phi)
// This is due to the maximum value a uint64 can store - after this value for n
// the uint64 value will overflow and won't be able to continue computing further.
func fibApproximation(n uint64) uint64 {
	fn := float64(n)
	psi := (1.0 - math.Sqrt(5.0)) / 2.0
	return uint64(math.Floor(((math.Pow(math.Phi, fn) - math.Pow(psi, fn)) / math.Sqrt(5)) + 0.5))
}