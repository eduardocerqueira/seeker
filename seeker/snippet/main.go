//date: 2024-11-18T17:03:40Z
//url: https://api.github.com/gists/e3b9e81591f0a74093d73465fdaae9e9
//owner: https://api.github.com/users/SKumarSpace

package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"time"
)

// Custom type to wrap sql.NullTime
type NullableTime struct {
	sql.NullTime
}

// UnmarshalJSON is a custom unmarshaler for NullableTime
func (nt *NullableTime) UnmarshalJSON(b []byte) error {
	// Check if the value is null
	if string(b) == "null" {
		nt.Valid = false
		return nil
	}

	// Otherwise, parse the time string
	t, err := time.Parse(`"2006-01-02T15:04:05Z"`, string(b))
	if err != nil {
		return err
	}

	nt.Time = t
	nt.Valid = true
	return nil
}

type MyStruct struct {
	Name string       `json:"name"`
	Time NullableTime `json:"time"`
}

func main() {
	data := `{"name": "example", "time": "2023-10-01T12:00:00Z"}`
	var myStruct MyStruct

	err := json.Unmarshal([]byte(data), &myStruct)
	if err != nil {
		fmt.Println("Error unmarshalling JSON:", err)
		return
	}

	if myStruct.Time.Valid {
		fmt.Println("Name:", myStruct.Name)
		fmt.Println("Time:", myStruct.Time.Time)
	} else {
		fmt.Println("Name:", myStruct.Name)
		fmt.Println("Time is null")
	}
}
