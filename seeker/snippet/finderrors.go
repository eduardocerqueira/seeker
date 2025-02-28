//date: 2025-02-28T16:43:41Z
//url: https://api.github.com/gists/b3976953ad0c125e1e5359c915f94b60
//owner: https://api.github.com/users/PXshadow

package main

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
)

var removeStrings = []string{
	",",
	".",
	":",
	"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
}

var removePrefixes = []string{
	"--",           // pass or fail
	"=== ",         // success
	"Called from ", // panic stack
	"Exception",    // runtime exception
}

func main() {
	const initDir = "tests/stdlogs"
	osDirs, err := os.ReadDir(initDir)
	if err != nil {
		panic(err)
	}
	dirs := make([]string, len(osDirs))
	for i := 0; i < len(osDirs); i++ {
		dirs[i] = osDirs[i].Name()
	}

	sort.Strings(dirs)

	//fmt.Println(dirs[0:3])
	//return

	cases := []Case{}

	const target = "hl" // MODIFY
	const ext = "_" + target + ".log"
	for _, dir := range dirs {
		// filter out all files except  with extension .log
		if dir[len(dir)-len(ext):] != ext {
			continue
		}

		b, err := os.ReadFile(filepath.Join(initDir, dir))
		if err != nil {
			panic(err)
		}
		lines := strings.Split(string(b), "\n")
		searchLines := []string{}

		for _, line := range lines {
			skip := false
			for _, prefix := range removePrefixes {
				if len(line) > len(prefix) && line[:len(prefix)] == prefix {
					skip = true
					break
				}
			}
			if skip {
				continue
			}
			//"     |   "
			line = strings.Trim(line, " ")
			line = strings.Trim(line, "|")
			line = strings.Trim(line, " ")
			if line == "" {
				continue
			}
			searchLine := line
			gotStr := "got "
			gotIndex := strings.Index(searchLine, gotStr)
			if gotIndex != -1 {
				searchLine = searchLine[:gotIndex]
			}
			for _, s := range removeStrings {
				searchLine = strings.ReplaceAll(searchLine, s, "")
			}
			if !strings.Contains(searchLine, " ") {
				continue
			}
			// special cases to ignore
			if searchLine == "Called from here" || searchLine == "Exception __skip__" {
				continue
			}
			searchLines = append(searchLines, searchLine)
		}
		cases = append(cases, Case{
			lines:       lines,
			searchLines: searchLines,
			path:        dir,
		})
		//println("add case", len(cases))
		//println(dir)

	}

	//os.Exit(0)

	mapCount := map[string]*CaseCount{}

	for i := range cases {
		for _, searchLine := range cases[i].searchLines {
			for j := range cases {
				if i == j {
					continue
				}
				for _, searchLines2 := range cases[j].searchLines {
					if searchLine != searchLines2 {
						continue
					}
					if c, exists := mapCount[searchLine]; exists {
						//println(cases[i].path, cases[j].path, searchLine)
						c.count++
					} else {
						mapCount[searchLine] = &CaseCount{
							data:  cases[i],
							count: 1,
						}
					}
					break
				}
			}
		}
	}

	maxCount := 0
	const minCount = 10 // MODIFY
	for searchLine, data := range mapCount {
		if maxCount < data.count {
			maxCount = data.count
		}
		if data.count < minCount {
			continue
		}
		_ = searchLine
		println(searchLine)
		println("COUNT:", data.count)
		println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.")
		// let go handle newlines in a cross platform way
		for _, line := range data.data.lines {
			println(line)
		}
		println("==========")
	}
	println("MAX COUNT:", maxCount)
}

type Case struct {
	path        string
	lines       []string
	searchLines []string
}

type CaseCount struct {
	data  Case
	count int
}
