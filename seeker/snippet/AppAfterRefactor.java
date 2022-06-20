//date: 2022-06-20T17:14:09Z
//url: https://api.github.com/gists/851e6cb9d28f12119bfcf52ba12d5712
//owner: https://api.github.com/users/BartekCK

import java.util.ArrayList;

interface Averagable {
    public double calculateAverage();
}

class Student implements Averagable {
    private double averageMark;

    public Student(double averageMark) {
        this.averageMark = averageMark;
    }

    public double calculateAverage() {
        return averageMark;
    }
}

class Group extends ArrayList<Student> implements Averagable {
    private String groupName;

    public Group(String groupName) {
        this.groupName = groupName;
    }

    public double calculateAverage() {
        double sum = 0;
        for (Student student : this) {
            sum += student.calculateAverage();
        }
        return sum / this.size();
    }
}

class AppAfterRefactor {

    public static double calculateAverage(Averagable averagable) {
        return averagable.calculateAverage();
    }

    public static void main(String[] args) {
        Student s1 = new Student(5);
        Student s2 = new Student(4);

        Group g1 = new Group("A");
        g1.add(s1);
        g1.add(s2);

        System.out.println(calculateAverage(s1));
        System.out.println(calculateAverage(g1));
    }
}