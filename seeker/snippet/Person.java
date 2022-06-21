//date: 2022-06-21T17:08:20Z
//url: https://api.github.com/gists/33a484f2c50bae5fc8f399a2c0b313eb
//owner: https://api.github.com/users/KEMOBARRA

publc class Person{
  // Field variables
  String name;
  double salary;
  
  // constructor
  public Person(String name, double salary){
    this.name = name;
    this.salary = salary;
    }
    
    // Getter for the constructor variables
    public String getName(){
      return name;
      }
      
      public double getSalary(){
        return salary;
        }
        
        public String toString(){
        return "name is " + name  +  "\n " + " salary is : " + salary;
		}
}