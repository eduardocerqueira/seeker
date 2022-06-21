//date: 2022-06-21T17:08:20Z
//url: https://api.github.com/gists/33a484f2c50bae5fc8f399a2c0b313eb
//owner: https://api.github.com/users/KEMOBARRA

public class Supervisor extends Person{
  
  // Field varialbe
    double salaryBonus;
  
  // constructor
  
  public Supervisor(String name, double salary, double salaryBonus){
    super(name,salary);
    this.salaryBonus = salaryBonus;
  }
  
  // getter for the salary bonus and the to toString
  public double getSalaryBonus(){
    return salaryBonus;
  }
}
  
  pubic String toString (){
    return super.toString() + "\n" + " salary bonus is :  " + salaryBonus;
  }
}