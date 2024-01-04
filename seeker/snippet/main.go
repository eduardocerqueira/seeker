//date: 2024-01-04T16:49:37Z
//url: https://api.github.com/gists/7bdd0f83a8ff54852b45180c3174899d
//owner: https://api.github.com/users/maxsei

package main

import (
	"context"
	"database/sql"
	_ "embed"
	"fmt"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"github.com/uptrace/bun"
	"github.com/uptrace/bun/dialect/sqlitedialect"
)

var schema string = `
CREATE TABLE IF NOT EXISTS authors (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
) STRICT;
CREATE TABLE IF NOT EXISTS books (
    id INTEGER PRIMARY KEY,
    title TEXT UNIQUE NOT NULL,
    -- published_ts_ms INTEGER NOT NULL,
    author_id INTEGER NOT NULL,
    FOREIGN KEY (author_id) REFERENCES authors(id)
) STRICT;
CREATE TABLE IF NOT EXISTS chapters (
    id INTEGER PRIMARY KEY,
    title TEXT UNIQUE NOT NULL,
    book_id INTEGER NOT NULL,
    FOREIGN KEY (book_id) REFERENCES books(id)
) STRICT;
`

// Data model
type Author struct {
	id    int
	name  string
	books []Book
}
type Book struct {
	id       int
	age      *time.Time
	title    string
	chapters []Chapter
}
type Chapter struct {
	id    int
	title string
}

// Query Specifiers
type QueryFields struct {
	authors QueryAuthors
}
type QueryAuthors struct {
	id    bool
	name  bool
	books QueryBooks
}
type QueryBooks struct {
	id       bool
	title    bool
	chapters QueryChapters
}
type QueryChapters struct {
	id    bool
	title bool
}

func main() {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		panic(err)
	}
	defer db.Close()
	if _, err := db.Exec(schema); err != nil {
		panic(err)
	}
	b := bun.NewDB(db, sqlitedialect.New())
	var expected = []Author{
		{id: 0, name: "Ava", books: []Book{
			{id: 0, title: "A Symphony of Stardust", chapters: []Chapter{
				{id: 0, title: "Harmony in the Cosmos"},
				{id: 1, title: "Meteor Meltdown"},
				{id: 2, title: "Stellar Serenade"},
				{id: 3, title: "Galactic Rhapsody"},
			}},
		}},
		{id: 1, name: "Max", books: []Book{
			{id: 1, title: "Alchemy of the Mind", chapters: []Chapter{
				{id: 4, title: "The Alchemist's Labyrinth"},
				{id: 5, title: "Cognizance Concoctions"},
				{id: 6, title: "Mindscape Metamorphosis"},
				{id: 7, title: "Enigmatic Elixirs"},
				{id: 8, title: "The Thought Transmutation"},
				{id: 9, title: "Psyche's Alchemical Dance"},
			}},
			{id: 2, title: "Astral Ascendance", chapters: []Chapter{
				{id: 10, title: "Ascension's Prelude"},
				{id: 11, title: "Voyage to the Astral Plane"},
				{id: 12, title: "Celestial Elevation"},
				{id: 13, title: "Starlight Sojourn"},
				{id: 14, title: "The Ethereal Apex"},
			}},
		}},
		{id: 3, name: "Gus", books: []Book{
			{id: 3, title: "Serenade of the Spheres", chapters: []Chapter{
				{id: 15, title: "Celestial Harmony"},
				{id: 16, title: "Orbiting Overture"},
				{id: 17, title: "Spheres' Serenade"},
				{id: 18, title: "Galactic Melodia"},
				{id: 19, title: "Cosmic Crescendo"},
				{id: 20, title: "Symphony of Celestial Strings"},
			}},
		}},
	}
	// Inserts
	for _, author := range expected {
		if _, err := b.NewInsert().Table("authors").Model(&map[string]any{
			"id":   author.id,
			"name": author.name,
		}).Exec(context.Background()); err != nil {
			panic(err)
		}
		for _, book := range author.books {
			if _, err := b.NewInsert().Table("books").Model(&map[string]any{
				"id":        book.id,
				"title":     book.title,
				"author_id": author.id,
			}).Exec(context.Background()); err != nil {
				panic(err)
			}
			for _, chapter := range book.chapters {
				if _, err := b.NewInsert().Table("chapters").Model(&map[string]any{
					"id":      chapter.id,
					"title":   chapter.title,
					"book_id": book.id,
				}).Exec(context.Background()); err != nil {
					panic(err)
				}
			}
		}
	}

	// Fields that we might want to query
	fields := QueryFields{authors: QueryAuthors{
		id:   true,
		name: true,
		books: QueryBooks{
			id:    true,
			title: true,
			chapters: QueryChapters{
				id:    true,
				title: true,
			},
		},
	}}

	var actual []Author
	{
		expr := b.NewSelect().Table("authors")
		// Dest is a slice of pointers that will be scanned into the query once it
		// is built.
		var dest []any
		// Map column takes a variable and assigns it as a row that will be scanned
		// from the query in the order it was "mapped".
		mapCol := func(k string, x any) {
			expr = expr.ColumnExpr(fmt.Sprintf(`%s as "%s"`, k, k))
			dest = append(dest, x)
		}
		// Flattened list of all possible values in the data model that are nullable
		// for because values in sql can always be null.
		var (
			// Authors
			authorsRel  sql.NullInt64
			authorsName sql.NullString
			// Books
			booksRel   sql.NullInt64
			booksTitle sql.NullString
			// Chapters
			chaptersRel   sql.NullInt64
			chaptersTitle sql.NullString
		)

		// Authors
		if fields.authors.id {
			mapCol("authors.id", &authorsRel)
		}
		if fields.authors.name {
			mapCol("authors.name", &authorsName)
		}
		// Books
		if fields.authors.books.id ||
			fields.authors.books.title {
			expr = expr.Join(`LEFT JOIN books ON books.author_id = authors.id`)
			mapCol("books.id", &booksRel)
			if fields.authors.books.title {
				mapCol("books.title", &booksTitle)
			}
		}
		// Chapters
		if fields.authors.books.chapters.id ||
			fields.authors.books.chapters.title {
			expr = expr.Join(`LEFT JOIN chapters ON chapters.book_id = books.id`)
			mapCol("chapters.id", &chaptersRel)
			if fields.authors.books.chapters.title {
				mapCol("chapters.title", &chaptersTitle)
			}
		}

		// Do query.
		fmt.Println(expr)
		rows, err := db.Query(expr.String())
		if err != nil {
			panic(err)
		}
		defer rows.Close()

		// Scan rows.
		for rows.Next() {
			if err := rows.Scan(dest...); err != nil {
				panic(err)
			}

			// Denormalize rows into the nested data model
			// XXX: How can this be DRY'd up/ coupled with the query building logic?
			if authorsRel.Valid {
				// XXX: The top level of records is aliased as a pointer because of the
				// repeated pattern below...
				group := &actual
				var author *Author
				for i := range *group {
					if (*group)[i].id == int(authorsRel.Int64) {
						author = &(*group)[i]
						break
					}
				}
				if author == nil {
					*group = append(*group, Author{})
					author = &(*group)[len(*group)-1]
				}
				author.id = int(authorsRel.Int64)
				author.name = authorsName.String
				if booksRel.Valid {
					group := &author.books
					var book *Book
					for i := range *group {
						if (*group)[i].id == int(booksRel.Int64) {
							book = &(*group)[i]
							break
						}
					}
					if book == nil {
						*group = append(*group, Book{})
						book = &(*group)[len(*group)-1]
					}
					book.id = int(booksRel.Int64)
					book.title = booksTitle.String
					if chaptersRel.Valid {
						group := &book.chapters
						var chapter *Chapter
						for i := range *group {
							if (*group)[i].id == int(chaptersRel.Int64) {
								chapter = &(*group)[i]
								break
							}
						}
						if chapter == nil {
							*group = append(*group, Chapter{})
							chapter = &(*group)[len(*group)-1]
						}
						chapter.id = int(chaptersRel.Int64)
						chapter.title = chaptersTitle.String
					}
				}
			}
		}
	}
	fmt.Println(actual)
}
