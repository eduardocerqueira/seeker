//date: 2022-06-21T17:08:20Z
//url: https://api.github.com/gists/33a484f2c50bae5fc8f399a2c0b313eb
//owner: https://api.github.com/users/KEMOBARRA

public class Student extends Person{
  // Field variable
  Stirng course;
  
  
  // Constructor
  public Student(String name, double salary, String course){
    super(name,salary);
    this.course = course;
  }
  
  // getter for the course 
  
  public String getCourse(){
    return course;
  }
  
  
  public String toString (){
    return super.toString() + "\n" + " The course is : " + course;
  }
}
  
  