//date: 2023-02-14T16:49:36Z
//url: https://api.github.com/gists/3fc0268cb395eda620d55d29ba87d355
//owner: https://api.github.com/users/himcc

package main

import (
	"database/sql"
	"fmt"
	"log"

	_ "github.com/mattn/go-sqlite3"
)

func main() {
	fmt.Println("KKKK")

	db, err := sql.Open("sqlite3", "./foo.db")
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	rows, err := db.Query(".tables")
	if err != nil {
		log.Fatal(err)
	}
	defer rows.Close()
	for rows.Next() {
		var n string
		err = rows.Scan(&n)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(n)
	}
}