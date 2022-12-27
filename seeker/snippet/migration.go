//date: 2022-12-27T16:55:57Z
//url: https://api.github.com/gists/7d48d101b0cdebbbccad01b44d6e5232
//owner: https://api.github.com/users/albuquerque53

package db

import (
	"database/sql"
	"strings"

	_ "github.com/go-sql-driver/mysql"

	"github.com/golang-migrate/migrate"
	_mysql "github.com/golang-migrate/migrate/database/mysql"
)

type Migration struct {
	Migrate *migrate.Migrate
}

func (m *Migration) Up() error {
	err := m.Migrate.Up()

	if err != nil && err != migrate.ErrNoChange {
		return err
	}

	return nil
}

func (m *Migration) Down() error {
	return m.Migrate.Down()
}

func RunMigration(dbConn *sql.DB, migrationsFolderLocation string) (*Migration, error) {
	dataPath := []string{}
	dataPath = append(dataPath, "file://")
	dataPath = append(dataPath, migrationsFolderLocation)

	pathToMigrate := strings.Join(dataPath, "")

	driver, err := _mysql.WithInstance(dbConn, &_mysql.Config{})
	if err != nil {
		return nil, err
	}

	m, err := migrate.NewWithDatabaseInstance(pathToMigrate, "mysql", driver)
	if err != nil {
		return nil, err
	}

	return &Migration{Migrate: m}, nil
}