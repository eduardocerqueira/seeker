//date: 2023-01-04T16:41:33Z
//url: https://api.github.com/gists/d597de551614581d4506f12c411d3f99
//owner: https://api.github.com/users/CeoFred

package main

import "fmt"
import "os"
import "log"


func main() {
	logfile,err := os.Create("log.txt")

	if err != nil {
		fmt.Println("failed to create log file: ", err)
	}
	log.SetOutput(logfile)

	_, err = os.Open("no-file.txt")
	if err != nil {
		fmt.Println("err happened ", err)
		log.Println(err)
	}
}

