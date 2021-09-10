//date: 2021-09-10T16:48:08Z
//url: https://api.github.com/gists/eb19cfdcef48414df883cb9db19bbb4a
//owner: https://api.github.com/users/batuhannoz

package main

import (
	"database/sql"
	"fmt"
	"log"

	"github.com/go-sql-driver/mysql"
)

var db *sql.DB

type User struct{
	id int64
	name string
	surname string
	phoneNumber string
	createDate []uint8
}

func main() {
	dbConnection()

	users, err := usersByName("Batuhan")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Users found: %v\n", users)
	userID, err := userByID(1)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("User found: %v\n", userID)
	userInfo, err := addUser(User{
		name:  "Batuhan",
		surname: "Özdemir",
		phoneNumber: "+905301111111",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("ID of added user: %v\n", userInfo)
	user := User{
		name:  "Canberk",
		surname: "Özdemir",
		phoneNumber: "+905301111111",
		id: 6,
	}
	updateUserInfo, err := updateUser(user)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("ID of updated user: %v\n", updateUserInfo)
}

func dbConnection(){
	cfg := mysql.Config{
		User:   "batuhan",
		Passwd: "batuhan1",
		Net:    "tcp",
		Addr:   "127.0.0.1:3306",
		DBName: "sample",
	}
	var err error
	db, err = sql.Open("mysql", cfg.FormatDSN())
	if err != nil {
		log.Fatal(err)
	}
	pingErr := db.Ping()
	if pingErr != nil {
		log.Fatal(pingErr)
	}
	fmt.Println("Connected!")
}

func usersByName(name string) ([]User, error) {
	var users []User

	rows, err := db.Query("SELECT * FROM user WHERE name = ?", name)
	if err != nil {
		return nil, fmt.Errorf("UserByName %q: %v", name, err)
	}
	defer rows.Close()
	for rows.Next() {
		var usr User
		if err := rows.Scan(&usr.id, &usr.name, &usr.surname, &usr.phoneNumber, &usr.createDate); err != nil {
			return nil, fmt.Errorf("UserByName %q: %v", name, err)
		}
		users = append(users, usr)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("UserByName %q: %v", name, err)
	}
	return users, nil
}

func userByID(id int64) (User, error) {
	var user User
	row := db.QueryRow("SELECT * FROM user WHERE id = ?", id)
	if err := row.Scan(&user.id, &user.name, &user.surname, &user.phoneNumber, &user.createDate); err != nil {
		if err == sql.ErrNoRows {
			return user, fmt.Errorf("userById %d: no such user", id)
		}
		return user, fmt.Errorf("userById %d: %v", id, err)
	}
	return user, nil
}

func addUser(user User) (int64, error) {
	result, err := db.Exec("INSERT INTO user (name, surname, phone_number) VALUES (?, ?, ?)", user.name, user.surname, user.phoneNumber)
	if err != nil {
		return 0, fmt.Errorf("addUser: %v", err)
	}
	id, err := result.LastInsertId()

	if err != nil {
		return 0, fmt.Errorf("addUser: %v", err)
	}
	return id, nil
}

func updateUser(user User) (int64, error) {
	result, err := db.Exec("UPDATE user SET name = ?, surname = ?, phone_number = ? WHERE id = ?;", user.name, user.surname, user.phoneNumber, user.id)
	if err != nil {
		return 0, fmt.Errorf("update user: %v", err)
	}
	id, err := result.LastInsertId()
	if err != nil {
		return 0, fmt.Errorf("update user: %v", err)
	}
	return id, nil
}

