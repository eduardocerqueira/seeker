//date: 2022-08-30T17:02:12Z
//url: https://api.github.com/gists/a4961f4babc590de3c796a8c1f83048c
//owner: https://api.github.com/users/varfrog

package main

import (
	"fmt"
	"os"
)

const (
	fileStateOpen int = iota + 1
	fileStateClosed
)

var inputFileState = fileStateClosed
var outputFileState = fileStateClosed

const (
	recoverStrategyPanic = iota + 1
	recoverStrategyOsExit
)

// recoverStrategy specifies whether to call panic() or os.Exit() in recover().
// With recoverStrategyPanic, both opened files are closed when a panic occurs.
// With recoverStrategyOsExit, the process terminates right away without a chance for other recover() fns to get called,
// which in this case will leave the file descriptor of the output file opened.
var recoverStrategy = recoverStrategyPanic

func generateReport() (err error) {
	openOutputFile()
	fmt.Println("generateReport: output file OPENED.")

	defer func() {
		if outputFileState == fileStateOpen {
			closeOutputFile()
			fmt.Println("generateReport: output file CLOSED")
		}
	}()
	defer func() {
		if r := recover(); r != nil {
			fmt.Println(fmt.Sprintf("generateReport: caught panic: %v", r))
			closeOutputFile()
			fmt.Println("generateReport: output file CLOSED")
			err = fmt.Errorf("recovered error: %v", r)
		}
	}()

	readInputFile()
	return
}

func readInputFile() {
	openInputFile()
	fmt.Println("readInputFile: input file OPENED")

	defer func() {
		if inputFileState == fileStateOpen {
			closeInputFile()
			fmt.Println("readInputFile: input file CLOSED")
		}
	}()
	defer func() {
		if r := recover(); r != nil {
			fmt.Println(fmt.Sprintf("readInputFile: caught panic: %v", r))
			closeInputFile()
			fmt.Println("readInputFile: input file CLOSED.")
			if recoverStrategy == recoverStrategyPanic {
				fmt.Println("readInputFile: calling panic again")
				panic(r)
			} else {
				fmt.Println("readInputFile: calling os.Exit()")
				os.Exit(1)
			}
		}
	}()

	panic("deliberate panic")
}

func openOutputFile() {
	outputFileState = fileStateOpen
}

func closeOutputFile() {
	outputFileState = fileStateClosed
}

func openInputFile() {
	inputFileState = fileStateOpen
}

func closeInputFile() {
	inputFileState = fileStateClosed
}

func main() {
	if err := generateReport(); err != nil {
		fmt.Println(fmt.Errorf("main, generate a report: %v", err))
		os.Exit(1)
	}
	fmt.Println("Report generated")
}
