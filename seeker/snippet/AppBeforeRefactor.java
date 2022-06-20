//date: 2022-06-20T17:14:09Z
//url: https://api.github.com/gists/851e6cb9d28f12119bfcf52ba12d5712
//owner: https://api.github.com/users/BartekCK

import java.util.ArrayList;

class Student {
    private double averageMark;

    public Student(double averageMark) {
        this.averageMark = averageMark;
    }

    public double getAverageMark() {
        return averageMark;
    }
}

class Group extends ArrayList<Student> {
    private String groupName;

    public Group(String groupName) {
        this.groupName = groupName;
    }
}

class AppBeforeRefactor {

    public static double getStudentAverageMark(Student student) {
        return student.getAverageMark();
    }

    public static double getGroupAverageMark(Group group) {
        double sum = 0;
        for (Student s : group) {
            sum += s.getAverageMark();
        }
        return sum / group.size();
    }

    public static void main(String[] args) {
        Student s1 = new Student(5);
        Student s2 = new Student(4);

        Group g1 = new Group("A");
        g1.add(s1);
        g1.add(s2);

        System.out.println(getStudentAverageMark(s1));
        System.out.println(getGroupAverageMark(g1));
    }
}