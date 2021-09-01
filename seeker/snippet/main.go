//date: 2021-09-01T17:03:56Z
//url: https://api.github.com/gists/543aef91187a0e68fe0e306c63b0f9ed
//owner: https://api.github.com/users/takoikatakotako

package main

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"fmt"
	"log"
)

type Pokemon struct {
	Id   int    `json:"id"`
	Name string `json:"name"`
}

func main() {
	// pokemon.json を読み込む
	pokemonJsonFile, err := os.Open("pokemon.json")

	// pokemon.json の読み込みに失敗した場合
	if err != nil {
		log.Fatal(err)
	}

	// defer で pokemonJsonFile を閉じる
	defer pokemonJsonFile.Close()

	// pokemonJsonFile を読み込みパースする
	pokemonByteValue, _ := ioutil.ReadAll(pokemonJsonFile)
	var pokemon Pokemon
	json.Unmarshal(pokemonByteValue, &pokemon)

	fmt.Println(pokemon.Id, pokemon.Name)	// 143 Snorlax
}
