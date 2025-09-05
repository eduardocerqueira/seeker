//date: 2025-09-05T16:50:27Z
//url: https://api.github.com/gists/c808d03352f89d8c591bd72cb2d536f6
//owner: https://api.github.com/users/Andrew4d3

package main

import (
	"errors"
	"fmt"
)

type MyError struct {
	Code   int
	Errors []error
}

func (m MyError) Error() string {
	return errors.Join(m.Errors...).Error()
}

func (m MyError) Unwrap() []error {
	return m.Errors
}

func processErrors() error {
	err1 := errors.New("first error")
	err2 := errors.New("second error")
	err3 := errors.New("third error")

	return MyError{
		Code:   500,
		Errors: []error{err1, err2, err3},
	}
}

func main() {
	err := processErrors()

	// Handle the error based on its unwrap capability
	switch err := err.(type) {
	case interface{ Unwrap() error }:
		fmt.Println("Single wrapped error:", err.Unwrap())
	case interface{ Unwrap() []error }:
		fmt.Println("Multiple wrapped errors:") // Output: Multiple wrapped errors:
		for i, innerErr := range err.Unwrap() {
			fmt.Printf("  %d: %v\n", i+1, innerErr) // Output: 1: first error, 2: second error, 3: third error
		}
	default:
		fmt.Println("No wrapped errors")
	}