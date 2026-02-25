//date: 2026-02-25T17:52:07Z
//url: https://api.github.com/gists/946294a5cfdf298f6c4b2ad045a11ae0
//owner: https://api.github.com/users/saniales

package mongo

import (
	"context"
	"time"

	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.mongodb.org/mongo-driver/mongo/readpref"
)

// Storage represents the database connection.
type Storage struct {
	db        *mongo.Database
	documents *mongo.Collection
}

// NewDB initializes the database connection
func NewDB(uri string, dbName string) *mongo.Database {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	client, err := mongo.Connect(ctx, options.Client().ApplyURI(uri))
	if err != nil {
		panic(err)
	}

	// test connection
	err = client.Ping(ctx, readpref.Primary())
	if err != nil {
		panic(err)
	}

	return client.Database(dbName)
}

// NewStorage returns a new instance of Storage
func NewStorage(db *mongo.Database) *Storage {

	s := Storage{
		db:        db,
		documents: db.Collection("documents"),
	}

	return &s
}

func (s *Storage) CreateDocument(ctx context.Context, document *Document) error {
	document.ID = ""

	if result, err := s.documents.InsertOne(ctx, document); err == nil {
		document.ID = result.InsertedID.(primitive.ObjectID).Hex()
	} else {
		return err
	}

	return nil
}
