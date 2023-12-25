//date: 2023-12-25T16:41:21Z
//url: https://api.github.com/gists/c3190719558d2b073090526830b3ed84
//owner: https://api.github.com/users/elifdev

package org.example;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args)
    {
        Instructor instructor1 = new Instructor();
        instructor1.setUserFirstName("Engin");
        instructor1.setUserLastName("Demiroğ");
        instructor1.setEmail("engindemirog@gmail.com");
        instructor1.setPassword("123456789");
        instructor1.setImage("ed.jpg");
        instructor1.setCv("loremimpus1");

        Instructor instructor2 = new Instructor();
        instructor2.setUserFirstName("Halit Enes");
        instructor2.setUserLastName("Kalaycı");
        instructor2.setEmail("heneskalayci@gmail.com");
        instructor2.setPassword("987654321");
        instructor2.setImage("hek.jpg");
        instructor2.setCv("loremimpus2");

        Lessons lessons1 = new Lessons();
        lessons1.setLessonName("JAVA");
        lessons1.setInstructorName(instructor1.getUserFirstName() + ' ' + instructor1.getUserLastName());
        lessons1.setDesc("Java Dersi");
        lessons1.setPrice("Ücretsiz");
        lessons1.setImage("java.jpg");
        lessons1.setCategories("Programlama");
        lessons1.setVideos("java.mp4");
        lessons1.setHomeworks("Java Homework");

        Lessons lessons2 = new Lessons();
        lessons2.setLessonName("Python");
        lessons2.setInstructorName(instructor2.getUserFirstName() + ' ' + instructor2.getUserLastName());
        lessons2.setDesc("Python Dersi");
        lessons2.setPrice("20 TL");
        lessons2.setImage("python.jpg");
        lessons2.setCategories("Programlama");
        lessons2.setVideos("python.mp4");
        lessons2.setHomeworks("Python Homework");

        Student student1 = new Student();
        student1.setUserFirstName("John");
        student1.setUserLastName("Doe");
        student1.setPassword("password");
        student1.setEmail("johndoe@gmail.com");
        student1.setImage("jd.jpg");
        student1.setAddress("x street y apart");
        student1.setCreditCard("1111 2222 3333 4444");
        student1.setPhone("05333333333");

        System.out.println(student1.getUserFirstName() + " " + student1.getUserLastName());
        System.out.println(lessons1.getLessonName() + " || " + lessons1.getDesc() + " || " + lessons1.getInstructorName()
                + " || " + lessons1.getPrice() + " || " + lessons1.getImage());
        System.out.println(lessons2.getLessonName() + " || " + lessons2.getDesc() + " || " + lessons2.getInstructorName()
                + " || " + lessons2.getPrice() + " || " + lessons2.getImage());

    }
}