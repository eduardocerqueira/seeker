//date: 2024-11-01T16:44:48Z
//url: https://api.github.com/gists/335c5d5e662439db8b8cad7938f9b751
//owner: https://api.github.com/users/nadim1044

/// Let's learn about the Static keyword of Java programming.
/// Static keyword is use for Memory management in Java.
/// When you declare anything(variable or method) static in java then it get memory only once.
/// It is belongs to class instead of Object of class.
/// Want to know how??? let's see throw the code.
public class Main {
    
    public static void main(String[] args) {
        // uncomment this for static variable
        // Counter class have two variable count and countStatic
        // count is set to 0 and increment by 1 each time with new instance and getting initialise each time.
        // But static variable countStatic not initialise each time because it is static and gets memory only once.
        Counter one =new Counter();
        Counter two =new Counter();
        Counter three =new Counter();


        // Same as variable static methods are also belongs to class instead of object
        Student nadim = new Student(1,"Nadim");
        Student ashfaq = new Student(1,"Asfaq");
        nadim.display();
        ashfaq.display();
        Student.changeCollegeName();
        nadim.display();
        ashfaq.display();

        // Output
        //1 ::: 1
        //1 ::: 2
        //1 ::: 3
        //1 Nadim LDRP-ITR
        //1 Asfaq LDRP-ITR
        //1 Nadim IIT
        //1 Asfaq IIT

    }
}

///Java Program to demonstrate the use of static variable
class Student{
    int rollNo;//instance variable
    String name;
    static String college ="LDRP-ITR";//static variable

    static  void changeCollegeName() {
        college = "IIT";
        // Non-static field 'name' cannot be referenced from a static context
        // name = ""
    }
    Student(int r, String n){
        rollNo = r;
        name = n;
    }
    //method to display the values
    void display (){System.out.println(rollNo+" "+name+" "+college);}
}

//Java Program to demonstrate the use of an instance variable
//which get memory each time when we create an object of the class.
class Counter{
    int count=0;//will get memory each time when the instance is created
    static  int countStatic = 0;
    Counter(){
        count++;//incrementing value
        countStatic++;//incrementing value
        System.out.println(count+" ::: "+countStatic);
    }
}