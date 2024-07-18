//date: 2024-07-18T16:45:48Z
//url: https://api.github.com/gists/0ade80bfa276b63466c942e83890acdb
//owner: https://api.github.com/users/docsallover

class Animal {
  String name;
  int age;

  public void eat() {
    System.out.println(name + " is eating.");
  }
}

class Dog extends Animal {
  String breed;

  public void bark() {
    System.out.println(name + " is barking!");
  }
}

public class Main {
  public static void main(String[] args) {
    Dog myDog = new Dog();
    myDog.name = "Buddy";
    myDog.age = 3;
    myDog.breed = "Labrador";

    myDog.eat(); // Inherited from Animal
    myDog.bark(); // Specific to Dog class
  }
}