//date: 2021-10-04T16:58:55Z
//url: https://api.github.com/gists/3a4b227b86f6c67c6c91d5bfa2783a1f
//owner: https://api.github.com/users/aminjam

package main

import (
	"database/sql"

	"github.com/jackc/pgx"
	"github.com/jackc/pgx/stdlib"
)

func main() {
	conStr := "postgres://diego:diego_pw@localhost/"
	config, err := pgx.ParseConnectionString(conStr)
	checkError(err)

	driverConfig := &stdlib.DriverConfig{ConnConfig: config}
	stdlib.RegisterDriverConfig(driverConfig)
	revisedConStr := driverConfig.ConnectionString(conStr)
	db, err := sql.Open("pgx", revisedConStr)
	checkError(err)
	checkError(db.Ping())

}

func checkError(err error) {
	if err != nil {
		panic(err)
	}
}
