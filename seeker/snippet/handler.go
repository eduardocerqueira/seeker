//date: 2022-03-22T16:55:24Z
//url: https://api.github.com/gists/3bdfee96e0ed68e249f078335ae08905
//owner: https://api.github.com/users/carlosm27

package main

import (
    "encoding/json"
    "fmt"
    "io/ioutil"
    "net/http"

    "github.com/gorilla/mux"
)

var groceries = []Grocery{
    {Name: "Almod Milk", Quantity: 2},
    {Name: "Apple", Quantity: 6},
}

func AllGroceries(w http.ResponseWriter, r *http.Request) {

    fmt.Println("Endpoint hit: returnAllGroceries")
    json.NewEncoder(w).Encode(groceries)
}