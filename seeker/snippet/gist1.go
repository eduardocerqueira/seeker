//date: 2022-09-22T17:21:42Z
//url: https://api.github.com/gists/ac404f94bff80917c71697c430a97fe7
//owner: https://api.github.com/users/gloudx

package main

import (
	"fmt"
	"io"
	"strings"

	"golang.org/x/net/html"
)

func main() {
	body := `<html>
    <body>
        <h1>Main title</h1>
        <a href="https://code-maven.com/">Code Maven</a>
        <h2 id="subtitle" class="important">Some subtle title</h2>
    </body>
    </html>`

	reader := strings.NewReader(body)
	tokenizer : "**********"
	for {
		tt : "**********"
 "**********"	 "**********"	 "**********"i "**********"f "**********"  "**********"t "**********"t "**********"  "**********"= "**********"= "**********"  "**********"h "**********"t "**********"m "**********"l "**********". "**********"E "**********"r "**********"r "**********"o "**********"r "**********"T "**********"o "**********"k "**********"e "**********"n "**********"  "**********"{ "**********"
 "**********"	 "**********"	 "**********"	 "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********". "**********"E "**********"r "**********"r "**********"( "**********") "**********"  "**********"= "**********"= "**********"  "**********"i "**********"o "**********". "**********"E "**********"O "**********"F "**********"  "**********"{ "**********"
				return
			}
			fmt.Printf("Error: "**********"
			return
		}
		tag, hasAttr : "**********"
		fmt.Printf("Tag: %v\n", string(tag))
		if hasAttr {
			for {
				attrKey, attrValue, moreAttr : "**********"
				// if string(attrKey) == "" {
				//     break
				// }
				fmt.Printf("Attr: %v\n", string(attrKey))
				fmt.Printf("Attr: %v\n", string(attrValue))
				fmt.Printf("Attr: %v\n", moreAttr)
				if !moreAttr {
					break
				}
			}
		}
	}
}
