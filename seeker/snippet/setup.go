//date: 2022-09-05T17:11:41Z
//url: https://api.github.com/gists/3fd5a00e2ffb7a4a63c94e0a87d1da45
//owner: https://api.github.com/users/BetterProgramming

package config
import (
 "database/sql"
_ "github.com/go-sql-driver/mysql"
)
var DB *sql.DB
func ConnectToDb(config string) error {
 database, err := sql.Open("mysql", config)
 if err != nil {
  return err
 }
 DB = database
 return nil
}