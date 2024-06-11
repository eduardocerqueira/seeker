//date: 2024-06-11T16:47:07Z
//url: https://api.github.com/gists/f3bc9db35aa4ea7fc0d3d8870d32c7ab
//owner: https://api.github.com/users/docsallover

func makeAnimalSound(a Animal) {
  a.Speak()  // Call the Speak() method on any type implementing Animal
}

func main() {
  dog := Dog{}
  makeAnimalSound(dog) // Output: Dog is eating (from Eat() in Dog)
                         //        Woof! (from Speak() in Dog)

  // Similar calls for Cat and Bird types using the same function
}