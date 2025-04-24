//date: 2025-04-24T16:56:16Z
//url: https://api.github.com/gists/ab5337b4b0d678b3d2cdb90cfc03cb90
//owner: https://api.github.com/users/dschofie

package main

import (
	"context"
	"log"

	"github.com/ClickHouse/clickhouse-go/v2"
)

type row struct {
	Col1 Inner
}

type Inner struct {
	MyVal string `json:"my_val,omitempty"`
}

func AppendStruct() error {
	conn, err := clickhouse.Open(&clickhouse.Options{})
	if err != nil {
		return err
	}

	ctx := context.Background()
	defer func() {
		conn.Exec(ctx, "DROP TABLE example")
	}()
	if err := conn.Exec(ctx, `DROP TABLE IF EXISTS example`); err != nil {
		return err
	}
	if err := conn.Exec(ctx, `
		CREATE TABLE example (
			Col1 Tuple(my_val String)
		) Engine = Memory
		`); err != nil {
		return err
	}

	batch, err := conn.PrepareBatch(context.Background(), "INSERT INTO example")
	if err != nil {
		return err
	}

	err = batch.AppendStruct(&row{
		Col1: Inner{
			MyVal: "value",
		},
	})
	if err != nil {
		return err
	}
	return batch.Send()
}

func main() {
	err := AppendStruct()
	if err != nil {
		log.Fatal(err)
	}
}
