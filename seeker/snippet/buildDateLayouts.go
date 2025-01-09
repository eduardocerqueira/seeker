//date: 2025-01-09T16:55:33Z
//url: https://api.github.com/gists/727647d032786470a0293ffa6e12c0f5
//owner: https://api.github.com/users/rorycl

// 09 January 2025
package main

import (
	"cmp"
	"fmt"
	"slices"
	"sort"
	"time"
)

var dateLayouts []string = []string{}

var fmtMap map[string]*goVsLettersVsRCL = map[string]*goVsLettersVsRCL{}

type goVsLettersVsRCL struct{ gomessage, letters, rcl bool }

func (g goVsLettersVsRCL) Header() string {
	return fmt.Sprintf("%-7s | %-7s | %-7s", "gomail", "letters", "rcl")
}

func (g goVsLettersVsRCL) String() string {
	x := func(b bool) string {
		if b {
			return "x"
		}
		return " "
	}
	return fmt.Sprintf("%-7s | %-7s | %-7s", x(g.gomessage), x(g.letters), x(g.rcl))
}

// taken from https://go.dev/src/net/mail/message.go line 123
func buildDateLayouts() {

	// Generate layouts based on RFC 5322, section 3.3.
	dows := [...]string{"", "Mon, "}   // day-of-week
	days := [...]string{"2", "02"}     // day = 1*2DIGIT
	years := [...]string{"2006", "06"} // year = 4*DIGIT / 2*DIGIT
	seconds := [...]string{":05", ""}  // second
	// "-0700 (MST)" is not in RFC 5322, but is common.
	zones := [...]string{"-0700", "MST", "UT"} // zone = (("+" / "-") 4DIGIT) / "UT" / "GMT" / ...

	for _, dow := range dows {
		for _, day := range days {
			for _, year := range years {
				for _, second := range seconds {
					for _, zone := range zones {
						s := dow + day + " Jan " + year + " 15:04" + second + " " + zone
						dateLayouts = append(dateLayouts, s)
					}
				}
			}
		}
	}
	dateLayouts = sort.StringSlice(dateLayouts)
}

// taken from github.com/mnako/letters/parsers.go line 31
func lettersLayouts() []string {
	formats := []string{
		time.RFC1123Z,
		"Mon, 2 Jan 2006 15:04:05 -0700",
		time.RFC1123Z + " (MST)",
		"Mon, 2 Jan 2006 15:04:05 -0700 (MST)",
	}
	return sort.StringSlice(formats)
}

// suggestions in https://github.com/mnako/letters/issues/115
func rclLayouts() []string {
	formats := []string{
		time.RFC1123Z,
		"Mon, 2 Jan 2006 15:04:05 -0700",
		time.RFC1123Z + " (MST)",
		"Mon, 2 Jan 2006 15:04:05 -0700 (MST)",
		"2 Jan 2006 15:04:05 -0700 (MST)",
		"2 Jan 2006 15:04:05 -0700",
		"Mon, 2 Jan 2006 15:04:05 MST",
		"2 Jan 2006 15:04 -0700",
		"2 Jan 2006 15:04:05",
		"Mon, 2 Jan 2006 15:04:05 -0700",
	}
	return sort.StringSlice(formats)
}

func buildMap(label string, formats []string) {
	for _, format := range formats {
		lfm, ok := fmtMap[format]
		if !ok {
			fmtMap[format] = &goVsLettersVsRCL{}
			lfm = fmtMap[format]
		}
		switch label {
		case "gomessage":
			lfm.gomessage = true
		case "letters":
			lfm.letters = true
		case "rcl":
			lfm.rcl = true
		default:
			panic(label)
		}
	}
}

// extract keys from fmtMap
func keys() []string {
	keys := []string{}
	for k := range fmtMap {
		keys = append(keys, k)
	}
	lenCmp := func(a, b string) int {
		return cmp.Compare(len(b), len(a))
	}
	slices.SortFunc(keys, lenCmp)
	return keys
}

func main() {
	buildDateLayouts()
	buildMap("gomessage", dateLayouts)
	buildMap("letters", lettersLayouts())
	buildMap("rcl", rclLayouts())
	keys := keys()

	fmt.Printf("%40s : %s\n", "", goVsLettersVsRCL{}.Header())
	for _, k := range keys {
		fmt.Printf("%40s : %s\n", k, fmtMap[k])
	}
}
