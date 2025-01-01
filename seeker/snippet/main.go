//date: 2025-01-01T17:05:44Z
//url: https://api.github.com/gists/8a4e77af2bbb15f3e9b160c1803d22f7
//owner: https://api.github.com/users/nikolaymatrosov

package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"time"

	"github.com/ydb-platform/ydb-go-genproto/protos/Ydb"
	environ "github.com/ydb-platform/ydb-go-sdk-auth-environ"
	"github.com/ydb-platform/ydb-go-sdk/v3"
	"github.com/ydb-platform/ydb-go-sdk/v3/query"
	"github.com/ydb-platform/ydb-go-sdk/v3/retry"
	"github.com/ydb-platform/ydb-go-sdk/v3/sugar"
	"github.com/ydb-platform/ydb-go-sdk/v3/table/types"
)

var connectionString = flag.String("ydb", os.Getenv("YDB_CONNECTION_STRING"), "")

func main() {

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	flag.Parse()

	db, err := ydb.Open(ctx, *connectionString,
		environ.WithEnvironCredentials(),
	)
	if err != nil {
		panic(fmt.Errorf("connect error: %w", err))
	}
	defer func() { _ = db.Close(ctx) }()

	qc := db.Query()
	err = recreateTable(ctx, qc)
	if err != nil {
		panic(fmt.Errorf("recreate tables error: %w", err))
	}
	err = qc.Do(ctx, insertFlash)
	if err != nil {
		panic(fmt.Errorf("inser Flash error: %w", err))
	}
	err = retry.Retry(ctx, func(ctx context.Context) error {
		err := qc.Do(ctx, insert)
		if err != nil && ydb.IsOperationError(err, Ydb.StatusIds_PRECONDITION_FAILED) {
			return retry.RetryableError(err)
		}
		return nil
	})

	if err != nil {
		panic(fmt.Errorf("select simple error: %w", err))
	}
}

func insertFlash(ctx context.Context, s query.Session) error {
	users := []struct {
		Id    int
		Name  string
		Email string
	}{
		{Id: 2, Name: "The Flash", Email: "barry.allen@ccpd.gov"},
	}

	var data []types.Value
	for _, user := range users {
		data = append(data, types.StructValue(
			types.StructFieldValue("id", types.Int32Value(int32(user.Id))),
			types.StructFieldValue("name", types.UTF8Value(user.Name)),
			types.StructFieldValue("email", types.UTF8Value(user.Email)),
		))
	}

	err := s.Exec(ctx, fmt.Sprintf(`
		DECLARE $data AS List<Struct<
			id: Int32,
			name: Utf8,
			email: Utf8
		>>;
		
		REPLACE INTO %s
		SELECT
			id,
			name,
			email
		FROM AS_TABLE($data)
		RETURNING *;`, "`users`"),
		query.WithParameters(
			ydb.ParamsBuilder().
				Param("$data").
				BeginList().AddItems(data...).EndList().
				Build(),
		),
	)
	return err
}

func insert(ctx context.Context, s query.Session) (err error) {

	users := []struct {
		Name  string
		Email string
	}{
		{Name: "Batman", Email: "bruce@wayne.com"},
		{Name: "Green Arrow", Email: "oliver@queen-industries.com"},
		{Name: "Superman", Email: "clark.kent@dailyplanet.com"},
	}

	var data []types.Value
	for _, user := range users {
		data = append(data, types.StructValue(
			types.StructFieldValue("name", types.UTF8Value(user.Name)),
			types.StructFieldValue("email", types.UTF8Value(user.Email)),
		))
	}

	result, err := s.Query(ctx, fmt.Sprintf(`
		DECLARE $data AS List<Struct<
			name: Utf8,
			email: Utf8
		>>;
		
		REPLACE INTO %s
		SELECT
			name,
			email
		FROM AS_TABLE($data)
		RETURNING *;`, "`users`"),
		query.WithParameters(
			ydb.ParamsBuilder().
				Param("$data").
				BeginList().AddItems(data...).EndList().
				Build(),
		),
	)
	if err != nil {
		return err
	}

	defer func() {
		_ = result.Close(ctx)
	}()

	for {
		resultSet, err := result.NextResultSet(ctx)
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}

			return err
		}
		type info struct {
			ID    string `sql:"id"`
			Name  string `sql:"name"`
			Email string `sql:"email"`
		}
		for row, err := range sugar.UnmarshalRows[info](
			resultSet.Rows(ctx),
			query.WithScanStructAllowMissingFieldsInStruct(),
		) {
			if err != nil {
				return err
			}

			log.Printf("id: %v", row)
		}
	}

	return nil
}

func recreateTable(ctx context.Context, c query.Client) error {
	err := c.Exec(ctx, fmt.Sprintf(`DROP TABLE %s;`, "`users`"),
		query.WithTxControl(query.NoTx()),
	)
	if err != nil {
		return err
	}
	err = c.Exec(ctx, fmt.Sprintf(`
		CREATE TABLE %s (
			id Serial,
			name Utf8,
			email Utf8,
			PRIMARY KEY (id)
		);`, "`users`"),
		query.WithTxControl(query.NoTx()),
	)
	return err
}
