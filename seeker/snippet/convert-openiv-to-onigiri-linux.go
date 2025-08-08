//date: 2025-08-08T16:51:46Z
//url: https://api.github.com/gists/25ec1968cd96cb4a209b02536ede227b
//owner: https://api.github.com/users/jodieblend

package main

import (
	"encoding/xml"
	"fmt"
	"io"
	"os"
	"strings"
)


type Archive struct {
	XMLName xml.Name `xml:"archive"`
	Sources []Source `xml:"add"`
}

type Source struct {
	XMLName xml.Name `xml:"add"`
	Source  string   `xml:"source,attr"`
	Target  string   `xml:",chardata"`
}

func main() {
	r := strings.NewReplacer(
			"\\", "/",
		)

	xmlFile, err := os.Open("assembly.xml")

	if err != nil {
		fmt.Println(err)
		return
	}

	defer xmlFile.Close()

	if _, err := os.Stat("onigiri"); os.IsNotExist(err) {
		os.Mkdir("onigiri", 0755)
	}

	decoder := xml.NewDecoder(xmlFile)

	for {
		token, err : "**********"

		if err != nil {
			fmt.Println(err)
		}

		switch se : "**********"
		case xml.StartElement:
			if se.Name.Local == "archive" {
				var archive Archive

				if err := decoder.DecodeElement(&archive, &se); err != nil {
					fmt.Println(err)
					continue
				}

				for _, source := range archive.Sources {
					sourceFile := r.Replace("content/" + strings.TrimSpace(source.Source))
					sourceTarget := r.Replace("onigiri/" + strings.TrimLeft(strings.TrimSpace(source.Target), "\\"))
					fmt.Printf("Source: %s\n", sourceFile)
					fmt.Printf("Target: %s\n", sourceTarget)

					sourceFileStat, err := os.Stat(sourceFile)

					if err != nil {
						fmt.Println(err)
						fmt.Println()
						continue
					}

					if !sourceFileStat.Mode().IsRegular() {
						fmt.Printf("%s is not a regular file\n", sourceFile)
						continue
					}

					sourceFileFile, err := os.Open(sourceFile)

					if err != nil {
						fmt.Println(err)
						fmt.Println()
						continue
					}

					defer sourceFileFile.Close()

					sourceTargetDir := sourceTarget[:strings.LastIndex(sourceTarget, "/")]
					if _, err := os.Stat(sourceTargetDir); os.IsNotExist(err) {
						os.MkdirAll(sourceTargetDir, 0755)
					}

					targetFile, err := os.Create(sourceTarget)

					if err != nil {
						fmt.Println(err)
						fmt.Println()
						continue
					}

					defer targetFile.Close()

					if err != nil {
						panic(err)
					}

					buf := make([]byte, 1024)

					for {
						n, err := sourceFileFile.Read(buf)
						if err != nil && err != io.EOF {
							panic(err)
						}

						if n == 0 {
							break
						}

						if _, err := targetFile.Write(buf[:n]); err != nil {
							panic(err)
						}
					}

					fmt.Println()
				}

				return
			}
		}
	}
}	}
	}
}