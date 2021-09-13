//date: 2021-09-13T17:08:28Z
//url: https://api.github.com/gists/eaba3ab68c44bb243b1bca1af774ebb8
//owner: https://api.github.com/users/ibilalkayy

package main

import "fmt"

type User struct {
	Title, Body string
}

func main() {
	var users = []User{
		{Title: "Old title", Body: "Old Body"},
	}

	users = append(users, User{Title: "New title", Body: "New Body"})

	fmt.Println(users)
}