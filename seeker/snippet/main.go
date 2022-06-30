//date: 2022-06-30T20:58:47Z
//url: https://api.github.com/gists/6a31fd3dcd94df2107962248751ef496
//owner: https://api.github.com/users/nealarch01

package main

import (
	"fmt"
	"net/http"
	"log"
	"encoding/json"
)

func Greet(w http.ResponseWriter, req *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	

	type GreetResponse struct {
		Status int
		Message string
	}

	var greetResponse GreetResponse

	greetResponse.Status = 200
	greetResponse.Message = "OK"

	jsonData, err := json.Marshal(greetResponse)

	w.Write(jsonData)

	if err != nil {
		log.Fatalln("Greet marshal error")
		return
	}

}

func main() {
	http.HandleFunc("/greet", Greet)

	port := ":2000"

	fmt.Print("Listening at https://127.0.0.1", port, "\n")

	err := http.ListenAndServeTLS(port, "server.crt", "server.key", nil)

	if err != nil {
		log.Fatalln("ListenAndServeTLS:", err)
	}

}