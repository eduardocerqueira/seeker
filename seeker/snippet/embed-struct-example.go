//date: 2024-06-11T17:11:04Z
//url: https://api.github.com/gists/0fc13cef723f917bdc2a0a318a5d0fa7
//owner: https://api.github.com/users/docsallover

type User struct {
  ID       string
  Username string
  Email    string
}

type Profile struct {
  FirstName string
  LastName  string
  Bio       string
}

type UserProfile struct {
  User    // Embedded user struct
  *Profile // Anonymous embedding of profile struct
}