//date: 2024-07-18T16:59:01Z
//url: https://api.github.com/gists/6450d882c3b115dc4852a33d2db75f17
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
        Animal animal1 = new Dog();
        Animal animal2 = new Cat();
        
        animal1.makeSound(); // Output: Dog barks
        animal2.makeSound(); // Output: Cat meows
    }
}