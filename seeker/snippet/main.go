//date: 2024-12-24T16:39:54Z
//url: https://api.github.com/gists/08b0ef7ba028aff1a28cee0c760d71e7
//owner: https://api.github.com/users/sguerrini97

package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path"
	"regexp"
	"strings"
	"time"
)

var DATE_FORMAT = "20060102150405"

func extract_digits(input string, max_len int) string {
	re := regexp.MustCompile("(?m)[\\d]+")
	matches := re.FindAllString(input, -1)
	digits_str := strings.Join(matches, "")

	if len(digits_str) > max_len {
		digits_str = digits_str[0:max_len]
	}

	return digits_str
}

func main() {

	pictures_path := flag.String("p", "", "path to the pictures")

	flag.Parse()

	entries, err := os.ReadDir(*pictures_path)
	if err != nil {
		log.Fatalf("Error reading pictures dir: %s\n", err)
	}

	// loop over directory entries
	for _, entry := range entries {
		// skip subdirectories
		if entry.IsDir() {
			continue
		}

		info, err := entry.Info()
		if err != nil {
			log.Fatalf("Error retrieving file info for %s: %s\n", entry.Name(), err)
		}

		// parse date/time from digits in the filename
		date_str := extract_digits(entry.Name(), len(DATE_FORMAT))
		t, err := time.Parse(DATE_FORMAT, date_str)

		if err != nil {
			fmt.Printf("Error parsing time from %s: %s\n", entry.Name(), err)
			continue
		}

		// log date change
		fmt.Printf("\t%s - %s => %s\n", entry.Name(), info.ModTime(), t)

		// set file mtime to new date
		os.Chtimes(path.Join(*pictures_path, entry.Name()), t, t)
	}
}