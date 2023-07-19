//date: 2023-07-19T17:08:13Z
//url: https://api.github.com/gists/9b7e4c13049e37975ef44683ad7d58fb
//owner: https://api.github.com/users/Pomog

// QUEST-06, 6
// Write a program that can take --insert (or -i), --order (or -o) and a string as arguments.
// This program should :
// Insert the string given to the --insert (or -i), into the argument string, if given.
// If the flag --order (or -o) is given, order the string argument (in ASCII order).
// If there are no arguments or if the flag --help (or -h) is given, the options should be printed as in the example.
// The short flag will have two spaces before the (-).
// The explanation of the flag will have a tab followed by a space before the beginning of the sentence (This flag...).
package main

import (
	"fmt"
	"os"
)

var (
	isInsertFlag                    bool
	isOrderFlag                     bool
	insertSrting                    string
	orderString                     string
	shortHelpFlag, longHelpFlag     string = "-h", "--help"
	shortInsertFlag, longInsertFlag string = "-i", "--insert"
	shortOrderFlag, longOrderFlag   string = "-o", "--order"
)

func main() {
	args := os.Args[1:]
	if helpFlagIsPresent(args) {
		printHelp()
		return
	}

	if longInsertFlagIsPresent(args) {
		isInsertFlag = true
		insertSrting = getInsertValue(args, longInsertFlag)
	} else if shortInsertFlagIsPresent(args) {
		isInsertFlag = true
		insertSrting = getInsertValue(args, shortInsertFlag)
	}

	if longOrderFlagIsPresent(args) {
		isOrderFlag = true
		orderString = getOrdertValue(args)
	} else if shortOrderFlagIsPresent(args) {
		isOrderFlag = true
		orderString = getOrdertValue(args)
	}

	if !isOrderFlag && !isInsertFlag {
		fmt.Println(sliceOfStringsToString(args))
		return
	}

	if isOrderFlag && !isInsertFlag {
		fmt.Println(order(orderString))
		return
	}

	if isOrderFlag && isInsertFlag {
		fmt.Println(order(orderString + insertSrting))
		return
	}

	if !isOrderFlag && isInsertFlag {
		result := insertSrting
		if len(args) > 1 {
			for _, arg := range args[1:] {
				result = arg + result
			}
		}
		fmt.Println(result)
		return
	}
}

func helpFlagIsPresent(args []string) bool {
	return checkFlag(args, shortHelpFlag) || checkFlag(args, longHelpFlag) || len(args) == 0
}

func shortInsertFlagIsPresent(args []string) bool {
	return checkFlag(args, shortInsertFlag)
}

func longInsertFlagIsPresent(args []string) bool {
	return checkFlag(args, longInsertFlag)
}

func shortOrderFlagIsPresent(args []string) bool {
	return checkFlag(args, shortOrderFlag)
}

func longOrderFlagIsPresent(args []string) bool {
	return checkFlag(args, longOrderFlag)
}

func getInsertValue(args []string, flag string) string {
	for _, arg := range args {
		for i := 0; i <= len(arg)-len(flag); i++ {
			if arg[i:i+len(flag)] == flag {
				return arg[i+len(flag)+1:]
			}
		}
	}
	return ""
}

func getOrdertValue(args []string) string {
	return args[len(args)-1]
}

func checkFlag(args []string, flag string) bool {
	result := false
	for _, arg := range args {
		for i := 0; i <= len(arg)-len(flag); i++ {
			if arg[i:i+len(flag)] == flag {
				result = true
			}
		}
	}
	return result
}

func sliceOfStringsToString(slice []string) string {
	var result string
	for _, str := range slice {
		result += str
	}
	return result
}

func order(str string) string {
	runes := []rune(str)
	for i := 0; i < len(runes); i++ {
		for j := i; j < len(runes); j++ {
			if runes[i] > runes[j] {
				runes[i], runes[j] = runes[j], runes[i]
			}
		}
	}
	return string(runes)
}

func printHelp() {
	fmt.Println("--insert")
	fmt.Println("  -i")
	fmt.Println("\t This flag inserts the string into the string passed as argument.")
	fmt.Println("--order")
	fmt.Println("  -o")
	fmt.Println("\t This flag will behave like a boolean, if it is called it will order the argument.")
}