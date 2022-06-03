//date: 2022-06-03T16:58:16Z
//url: https://api.github.com/gists/c17096c24ea89592628a00c8347fa8d1
//owner: https://api.github.com/users/yyforyongyu

package main

import (
	"fmt"

	"github.com/lightningnetwork/lnd/kvdb"
)

var (
	openChannelBucket = []byte("open-chan-bucket")
)

func testOpen() error {
	// Point this to your db path.
	dbPath := "channel.db"

	db, err := kvdb.Open(
		kvdb.BoltBackendName, dbPath, true, kvdb.DefaultDBTimeout,
	)
	if err != nil {
		return fmt.Errorf("cannot open: %v", err)
	}
	defer db.Close()

	return kvdb.View(db, func(tx kvdb.RTx) error {
		rootBucket := tx.ReadBucket(openChannelBucket)
		if rootBucket == nil {
			return fmt.Errorf("empty root bucket")
		}

		return rootBucket.ForEach(func(k, _ []byte) error {
			fmt.Printf("found key: %x\n", k)
			return nil
		})
	}, func() {})
}

func main() {
	err := testOpen()
	fmt.Printf("err is: %v\n", err)
}