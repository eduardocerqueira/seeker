//date: 2026-02-25T17:52:07Z
//url: https://api.github.com/gists/946294a5cfdf298f6c4b2ad045a11ae0
//owner: https://api.github.com/users/saniales

package mongo_test

import (
	"context"
	"fmt"
	mongo "memongo-test"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tryvium-travels/memongo"
)

var mongoServer *memongo.Server

func TestMain(m *testing.M) {
	var err error
	mongoServer, err = memongo.StartWithOptions(&memongo.Options{MongoVersion: "8.0.0", ShouldUseReplica: true})
	if err != nil {
		panic(err)
	}
	defer mongoServer.Stop()

	os.Setenv("MONGO_URI", mongoServer.URI())

	os.Exit(m.Run())
}

func setup() *mongo.Storage {
	db := mongo.NewDB(mongoServer.URI(), memongo.RandomDatabase())

	storage := mongo.NewStorage(db)

	return storage
}

func TestCreateDocument(t *testing.T) {
	i := 1
	count := 1000

	for {
		t.Run("OK", func(t *testing.T) {
			storage := setup()

			document := mongo.Document{
				Name: fmt.Sprintf("Test %d", i),
			}

			err := storage.CreateDocument(context.TODO(), &document)
			require.NoError(t, err)
		})

		if i == count {
			break
		}

		i++
	}
}
