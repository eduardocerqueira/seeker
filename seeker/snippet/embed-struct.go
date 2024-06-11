//date: 2024-06-11T17:05:07Z
//url: https://api.github.com/gists/c86cb37d8b9b1071c362b81338615770
//owner: https://api.github.com/users/docsallover

type Person struct {
  Name string
  Age  int
}

type Employee struct {
  Person   // Embedded struct
  EmployeeID string
  Department string
}