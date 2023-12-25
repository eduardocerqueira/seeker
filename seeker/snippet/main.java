//date: 2023-12-25T16:52:42Z
//url: https://api.github.com/gists/69f02aebc99cd720728555aadb385d3b
//owner: https://api.github.com/users/fatihakin0

package org.example;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) {

        //Instructors

        Instructors instructor1 = new Instructors();
        instructor1.setId(1);
        instructor1.setFirstName("Engin");
        instructor1.setLastName("Demirog");
        instructor1.setPhone("05167237612");
        instructor1.setMail("engindemirog@gmail.com");
        instructor1.setPassword("2176387126");


        Instructors instructor2 = new Instructors();
        instructor1.setId(2);
        instructor2.setFirstName("Halit Enes");
        instructor2.setLastName("Kalaycı");
        instructor2.setPhone("05167231233");
        instructor2.setMail("haliteneskalycı@gmail.com");
        instructor1.setPassword("98572489");

        //Students

        Students student1 = new Students("Fatih","Akın","05761237821","fatihakın@gmail.com","7123312",1);
        Students student2 = new Students("Aziz Fatih","Yağız","05816236712","azizyağız@gmail.com","512431",2);

        //Titles

        Titles title1 = new Titles("Javascript", "Engin Demiroğ", "1.5 ay sürecek javascript kampı",0,"image1.jpg");
        Titles title2 = new Titles("Python ve Selenium", "Halit Enes Kalaycı", "Python ve Selenium yazılım geliştirici kampı",0,"image2.jpg");

        //Categories

        Categories category1 = new Categories("Programlama",1);
        Categories category2 = new Categories("Eğitim",2);

        /*
        Dataları toplu bir şekilde almak.

         Students[] student= {student1,student2};
        for (Students ogrenci : student) {
            System.out.println(ogrenci.firstName);

       */
        }