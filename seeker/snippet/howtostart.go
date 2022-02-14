//date: 2022-02-14T16:57:38Z
//url: https://api.github.com/gists/7838d2767e731a2cf7efe18c72efb64a
//owner: https://api.github.com/users/shoobyban

package main

import 	(
	"time"
	"log"
	"github.com/blackfireio/go-blackfire"
)

func main() {
	ender := blackfire.EnableNow()
	r, err := blackfire.NewServeMux("_")
	if err != nil {
		log.Fatal(err)
	}
	go log.Panic(http.ListenAndServe(":8999", r))
	defer ender.End()
  
  	// do your bits here
	time.Sleep(100000)
	// if you have issues with ender.End(), go to http://localhost:8999/_/dashboard and click end
}