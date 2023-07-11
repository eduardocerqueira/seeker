//date: 2023-07-11T16:41:51Z
//url: https://api.github.com/gists/990d3a29ff77767389dd1e9cc66e7ce2
//owner: https://api.github.com/users/dimitrilw

import (
    // most already have these imported; putting here just in case
    "log" 
    "os"
)

func logTest() {
    // log settings are universal to this package (file, for code challenges); i.e., it will also print
    log.Println("inside logTest")
}

func demoMain(/* code challenge's args here */) /*(result int)*/ {
    log.SetOutput(os.Stdout)
    log.SetFlags(log.Lshortfile)
	log.Println("Hello, world!")
    
    logTest()
}

// approx output:
// some_go_file.go:15: Hello, world!
// some_go_file.go:9: inside logTest

