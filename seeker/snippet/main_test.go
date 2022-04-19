//date: 2022-04-19T17:07:32Z
//url: https://api.github.com/gists/f382e49b648945640fec17b1f4f89be8
//owner: https://api.github.com/users/sipliy15310

package psql_test

import (
    "fmt"
    "github.com/adrianbrad/psqldocker"
    "github.com/adrianbrad/psqltest"
    "log"
    "os"
    "testing"
)

func TestMain(m *testing.M) {
    // psql connection parameters.
    const (
        usr           = "usr"
        password      = "pass"
        dbName        = "tst"
        containerName = "psql_docker_tests"
    )

    // run a new psql docker container.
    c, err := psqldocker.NewContainer(
        usr,
        password,
        dbName,
        psqldocker.WithContainerName(containerName),
        psqldocker.WithSql(`
        CREATE TABLE users(
            user_id UUID PRIMARY KEY,
            email VARCHAR NOT NULL
        );`),
    )
    if err != nil {
        log.Fatalf("err while creating new psql container: %s", err)
    }

    // exit code
    var ret int

    defer func() {
        // close the psql container
        err = c.Close()
        if err != nil {
            log.Printf("err while tearing down db container: %s", err)
        }

        // exit with the code provided by executing m.Run().
        os.Exit(ret)
    }()

    // compose the psql dsn.
    dsn := fmt.Sprintf(
        "user=%s "+
            "password=%s "+
            "dbname=%s "+
            "host=localhost "+
            "port=%s "+
            "sslmode=disable",
        usr,
        password,
        dbName,
        c.Port(),
    )

    // register the psql container connection details
    // in order to be able to spawn new database connections
    // in an isolated transaction.
    psqltest.Register(dsn)

    // run the package tests.
    ret = m.Run()
}