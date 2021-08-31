//date: 2021-08-31T03:09:14Z
//url: https://api.github.com/gists/abede6259b3b7493c7fe7702fd1818f7
//owner: https://api.github.com/users/chadleeshaw

package main

import (
  "strings"
  "os/exec"
  "fmt"
)

type memory struct {
	MemType         string `json:",omitempty"`
	MemForm         string `json:",omitempty"`
	MemLocation     string `json:",omitempty"`
	MemSize         string `json:",omitempty"`
	MemSpeed        string `json:",omitempty"`
	MemSerial       string `json:",omitempty"`
	MemManufacturer string `json:",omitempty"`
	MemPartNumber   string `json:",omitempty"`
}

func main() {
	memObjects := memInfo()
	fmt.Println(memObjects)
}

func readDmiDecode(hwtype string) []string {
	out, err := exec.Command("dmidecode", "--type", hwtype).Output()
	if err != nil {
		log.Fatal("Unable to gather DmiDecode info: " + err.Error())
	}
	return strings.Split(string(out), "\n\n")
}

func memInfo() []memory {
	var m []memory
	var tempm memory

	blocks := readDmiDecode("memory") // Paragraphs of DMI output, []string

	for _, block := range blocks {
		if strings.Contains(block, "Memory Device") {
			if strings.Contains(block, "No Module Installed") {
				continue
			}
			tempm = memory{}
			lines := strings.Split(block, "\n")
			for _, line := range lines { // Individual lines of paragraph, string
				line = strings.Trim(line, " \t\n\r")
				parts := strings.Split(line, ":")
				if len(parts) == 2 {
					key := strings.TrimSpace(parts[0])
					value := strings.TrimSpace(parts[1])
					switch key {
					case "Size":
						tempm.MemSize = value
					case "Form Factor":
						tempm.MemForm = value
					case "Locator":
						tempm.MemLocation = value
					case "Type":
						tempm.MemType = value
					case "Speed":
						tempm.MemSpeed = value
					case "Serial Number":
						tempm.MemSerial = value
					case "Manufacturer":
						tempm.MemManufacturer = value
					case "Part Number":
						tempm.MemPartNumber = value
					}
				}
			}
			m = append(m, tempm)
		}
	}
	return m
}