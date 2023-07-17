//date: 2023-07-17T16:45:58Z
//url: https://api.github.com/gists/8749f94ee7040eae436df092ac82ed2d
//owner: https://api.github.com/users/yash8917

package Practice;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

class Student implements Comparable<Student>{
    public int id;
    public String name;
    public  int age;

    public Student(int id, String name, int age) {
        this.id = id;
        this.name = name;
        this.age = age;
    }

    @Override
    public String toString() {
        return "Student{" +
                "id=" + id +
                ", name='" + name + '\'' +
                ", age=" + age +
                '}';
    }

    @Override
    public int compareTo(Student otherObject) {
        if(age > otherObject.age){
            return 1;
        }else if(age < otherObject.age){
            return -1;
        }else{
            return 0;
        }
//        return name.compareTo(otherObject.name); // we can perform it that way also if the type of is -> Wrapper Class(Integer, String etc.)
    }
}

class sortByName implements Comparator<Student>{

    @Override
    public int compare(Student o1, Student o2) {
        return o1.name.compareTo(o2.name);
    }
}


public class Dem1 {
    public static void main(String[] args) {

        Student obj1 = new Student(2,"Arjun",120);
        Student obj2= new Student(1,"Yudhister",150);
        Student obj3 = new Student(4,"Nakul",111);
        Student obj4 = new Student(3,"Bheem",116);


        ArrayList<Student> student = new ArrayList<>();
        student.add(obj1);
        student.add(obj2);
        student.add(obj3);
        student.add(obj4);
        System.out.println("Before sorting");
        
        // print with the help of Lamda & forEach
        student.forEach( name -> System.out.println(name));

        System.out.println("After sorting ");

        Collections.sort(student);
        Collections.sort(student, new sortByName());

        student.forEach(n -> System.out.println(n));

    }

}
