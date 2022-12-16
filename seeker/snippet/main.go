//date: 2022-12-16T17:03:16Z
//url: https://api.github.com/gists/b4c53584f06f1cbee0538ef2e7b7635f
//owner: https://api.github.com/users/Vostbur

package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
)

func main() {
	data, err := ioutil.ReadFile("./data-20190514T0100.json")
	if err != nil {
		fmt.Print(err)
	}

	type GlobalIds []struct {
		Id int `json:"global_id"`
	}
	var globalIds GlobalIds

	err = json.Unmarshal(data, &globalIds)
	if err != nil {
		fmt.Print(err)
	}

	var sum int
	for _, i := range globalIds {
		sum += i.Id
	}

	fmt.Print(sum)
}