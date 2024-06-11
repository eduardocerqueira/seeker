//date: 2024-06-11T17:02:27Z
//url: https://api.github.com/gists/00a3c3ba997a40df4c0615a7207dd393
//owner: https://api.github.com/users/docsallover

type Book struct {
  Title  string
  Author string
  Year   int
  Pages  int `json:"num_pages"` // Example struct tag for JSON encoding
}

func main() {
  book1 := Book{Title: "The Lord of the Rings", Author: "J.R.R. Tolkien", Year: 1954, Pages: 1178}
  fmt.Println("Book Title:", book1.Title)
}