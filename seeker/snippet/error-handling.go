//date: 2024-06-14T16:51:28Z
//url: https://api.github.com/gists/c7e22512842fcf40bf28cdd3f3aa986c
//owner: https://api.github.com/users/docsallover

func GetUser(id int) (*User, error) {
  user, err := db.Query("SELECT * FROM users WHERE id = ?", id)
  if err != nil {
    return nil, err
  }
  // ... process user data
  return user, nil
}

func main() {
  user, err := GetUser(10)
  if err != nil {
    fmt.Println("Error getting user:", err)
    // Handle the error (e.g., use a default user object)
    user = &User{ID: 10, Name: "Unknown"}
  }
  fmt.Println("User:", user.Name)
}