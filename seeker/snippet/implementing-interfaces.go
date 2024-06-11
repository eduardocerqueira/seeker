//date: 2024-06-11T16:44:54Z
//url: https://api.github.com/gists/c81e68dba3fe3d070459ba738a79fca8
//owner: https://api.github.com/users/docsallover

type Dog struct {}

func (d Dog) Eat() {
  fmt.Println("Dog is eating")
}

func (d Dog) Speak() string {  // Implementation matches the interface signature
  return "Woof!"
}

// Similar implementations for other types like Cat and Bird