//date: 2026-01-09T17:16:38Z
//url: https://api.github.com/gists/f07db2dcdbf07070174855da4efd5d7f
//owner: https://api.github.com/users/Intyrrot14

// Task1.java
public class Task1 {
    public static void main(String[] args) {
        Student student = new Student("Tamirlan", 20, "S12345");
        Teacher teacher = new Teacher("Togzan", 45, "T98765");

        System.out.println("Student Info:");
        student.displayInfo();

        System.out.println("\nTeacher Info:");
        teacher.displayInfo();

        System.out.println("\nUsing Polymorphism:");
        Person[] people = { student, teacher };
        for (Person p : people) {
            p.displayInfo(); // dynamic dispatch
        }
    }
}

// Base class
class Person {
    private String name;
    private int age;

    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }

    public String getName() { return name; }
    public int getAge() { return age; }

    public void displayInfo() {
        System.out.println("Name: " + name + ", Age: " + age);
    }
}

// Derived class
class Student extends Person {
    private String studentId;

    public Student(String name, int age, String studentId) {
        super(name, age);
        this.studentId = studentId;
    }

    @Override
    public void displayInfo() {
        System.out.println("Name: " + getName() + ", Age: " + getAge() + ", Student ID: " + studentId);
    }
}

// Derived class
class Teacher extends Person {
    private String employeeId;

    public Teacher(String name, int age, String employeeId) {
        super(name, age);
        this.employeeId = employeeId;
    }

    @Override
    public void displayInfo() {
        System.out.println("Name: " + getName() + ", Age: " + getAge() + ", Employee ID: " + employeeId);
    }
}
