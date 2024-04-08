//date: 2024-04-08T16:59:02Z
//url: https://api.github.com/gists/8b996f03dca8e6dc90880ba037024dde
//owner: https://api.github.com/users/Sourjaya

package xxd

//import other packages
import (
	"bufio"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"

	flag "github.com/spf13/pflag"
)

//Struct containing flag values as entered in the terminal.
type Flags struct {
	Endian    bool
	GroupSize string
	Length    string
	Columns   string
	Seek      string
	Revert    bool
}

// Struct containing parsed flag values from original values
type ParsedFlags struct {
	IsFile bool
	E      bool
	G      int
	L      int
	S      int
	C      int
	R      bool
}

// Struct to indicate whether a particular flag was used as options
type IsSetFlags struct {
	IsSetG bool
	IsSetL bool
	IsSetC bool
	IsSetS bool
}

// Function to parse flags from the command line.
func NewFlags() (*Flags, *IsSetFlags, []string) {
	flags := new(Flags)
	setFlags := &IsSetFlags{}
	flag.BoolVarP(&flags.Endian, "little-endian", "e", false, "little-endian")
	flag.StringVarP(&flags.GroupSize, "group-size", "g", "2", "group-size")
	flag.StringVarP(&flags.Length, "length", "l", "-1", "length")
	flag.StringVarP(&flags.Columns, "cols", "c", "16", "columns")
	flag.StringVarP(&flags.Seek, "seek", "s", "0", "seek")
	flag.BoolVarP(&flags.Revert, "revert", "r", false, "revert")
	flag.Parse()
	flag.Visit(func(f *flag.Flag) {
		if f.Shorthand == "c" {
			setFlags.IsSetC = true
		}
		if f.Shorthand == "l" {
			setFlags.IsSetL = true
		}
		if f.Shorthand == "g" {
			setFlags.IsSetG = true
		}
		if f.Shorthand == "s" {
			setFlags.IsSetS = true
		}
	})
	args := flag.Args()
	return flags, setFlags, args
}