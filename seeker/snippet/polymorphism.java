//date: 2024-07-18T16:53:09Z
//url: https://api.github.com/gists/36cde6ed6bd33096f39a6a5ac4ddc087
//owner: https://api.github.com/users/docsallover

class Animal {
    public void makeSound() {
        System.out.println("Animal makes a sound");
    }
}

class Dog extends Animal {
    @Override
    public void makeSound() {
        System.out.println("Dog barks");
    }
}

class Cat extends Animal {
    @Override
    public void makeSound() {
        System.out.println("Cat meows");
    }
}

public class Main {
    public static void main(String[] args) {
        Animal animal = new Dog(); // Polymorphic reference
        animal.makeSound(); // Output: Dog barks
    }
}