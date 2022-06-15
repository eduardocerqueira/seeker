//date: 2022-06-15T17:02:46Z
//url: https://api.github.com/gists/aa5180ea26350d7d3470b5b769eb450f
//owner: https://api.github.com/users/TheDevtop

package main

/*
	Prog: Read & Apply Regular Expressions
	Vers: 1.0
	Auth: Thijs Haker
*/

import (
	"flag"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"
)

const STDIN string = "/dev/stdin"

func printUsage() {
	fmt.Println("rap: Read & Apply Regular Expressions")
	flag.PrintDefaults()
}

func main() {
	// Declare stuff
	var (
		// Error
		err error

		// Regex pointer
		rex *regexp.Regexp

		// File buffer
		buf []byte

		// Flags
		optRex  *string = flag.String("r", "", "Specifies the regular expression")
		optSubs *string = flag.String("s", "", "Specifies string to substitute")
		optFile *string = flag.String("f", STDIN, "Specifies input stream")
	)

	// Assign usage and parse flags
	flag.Usage = printUsage
	flag.Parse()

	// Attempt to read input stream
	if *optFile == STDIN {
		if buf, err = io.ReadAll(os.Stdin); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	} else {
		if buf, err = os.ReadFile(*optFile); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}

	// Attempt to compile rexMatch
	if rex, err = regexp.Compile(*optRex); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	// If subs is empty, just match string and printout
	if *optSubs == "" {
		strBuf := rex.FindAllString(string(buf), -1)
		fmt.Println(strings.Join(strBuf, "\t"))
		os.Exit(0)
	}

	// Apply replace
	fmt.Println(rex.ReplaceAllString(string(buf), *optSubs))
	os.Exit(0)
}