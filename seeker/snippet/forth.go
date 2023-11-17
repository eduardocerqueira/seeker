//date: 2023-11-17T16:40:14Z
//url: https://api.github.com/gists/68341cd45d80e6a1a55e5b0d9ffa2084
//owner: https://api.github.com/users/helioproduct

//go:build !solution

package main

import (
	"errors"
	"strconv"
	"strings"
)

type Evaluator struct {
	doubleOperations map[string]bool
	singleOperations map[string]bool
	defines          map[string]string
}

// NewEvaluator creates evaluator.
func NewEvaluator() *Evaluator {
	return &Evaluator{
		doubleOperations: map[string]bool{
			"+": true,
			"-": true,
			"*": true,
			"/": true,
		},

		singleOperations: map[string]bool{
			"dup":  true,
			"over": true,
			"drop": true,
			"swap": true,
		},
		defines: map[string]string{},
	}
}

func (e *Evaluator) Process(row string) ([]int, error) {

	split := strings.Fields(row)

	stack := make([]int, 0)
	words := make([]string, 0)

	defining := false

	for _, word := range split {
		if word == ":" {
			defining = true
		}
		definition, ok := e.defines[strings.ToLower(word)]
		if ok && !defining {
			words = append(words, strings.Fields(definition)...)
		} else {
			words = append(words, word)
		}
	}

	for i := 0; i < len(words); i++ {

		word := words[i]
		word = strings.ToLower(word)

		// new define
		if word == ":" {
			newWord := strings.ToLower(words[i+1])

			if _, err := strconv.Atoi(newWord); err == nil {
				err := errors.New("redefining numbers")
				return stack, err
			}

			i += 2

			var definition string
			for i < len(words) {
				word := strings.ToLower(words[i])
				value, ok := e.defines[word]
				if ok {
					word = value
				}
				if words[i] == ";" {
					break
				}
				definition += word
				definition += " "
				i++
			}
			e.defines[newWord] = definition
		} else if !e.singleOperations[word] && !e.doubleOperations[word] {
			number, err := strconv.Atoi(word)
			if err != nil {
				err := errors.New("not a number: " + word)
				return stack, err
			}
			stack = append(stack, number)

		} else if e.doubleOperations[word] {
			if len(stack) < 2 {
				err := errors.New("not enough arguments for double operations")
				return stack, err
			}
			op2, op1 := stack[len(stack)-1], stack[len(stack)-2]
			stack = stack[:len(stack)-1]
			if word == "+" {
				stack[len(stack)-1] = int(op1 + op2)
			} else if word == "-" {
				stack[len(stack)-1] = int(op1 - op2)
			} else if word == "/" {
				if op2 == 0 {
					err := errors.New("division by zero")
					return stack, err
				}
				stack[len(stack)-1] = int(op1 / op2)
			} else if word == "*" {
				stack[len(stack)-1] = int(op1 * op2)
			}

		} else if e.singleOperations[word] {
			if len(stack) < 1 {
				err := errors.New("not enough arguments to single operations")
				return stack, err
			}
			if word == "drop" {
				stack = stack[:len(stack)-1]
			} else if word == "over" {
				if len(stack) < 2 {
					err := errors.New("not enough numbers for signle operation")
					return stack, err
				}
				stack = append(stack, stack[len(stack)-2])

			} else if word == "dup" {
				stack = append(stack, stack[len(stack)-1])
			} else if word == "swap" {
				if len(stack) < 2 {
					err := errors.New("not enough numbers to swap")
					return stack, err
				}
				first, second := stack[len(stack)-1], stack[len(stack)-2]
				stack[len(stack)-1], stack[len(stack)-2] = second, first
			}
		}
	}
	return stack, nil
}