//date: 2024-06-11T17:07:23Z
//url: https://api.github.com/gists/78795619f838cd3e832a8bb47d454acb
//owner: https://api.github.com/users/docsallover

emp1 := Employee{
  Person:   Person{Name: "Alice", Age: 30},
  EmployeeID: "12345",
  Department: "Engineering",
}

fmt.Println(emp1.Name)  // Output: Alice (promoted from Person)