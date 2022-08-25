//date: 2022-08-25T15:31:59Z
//url: https://api.github.com/gists/5fce9f3de630c355ca6843b203bef333
//owner: https://api.github.com/users/DhavalDalal

public class Employee {
  private final Integer id;
  private final String name;
  
  public Employee(final Integer id, final String name) {
    this.id = id;
    this.name = name;
  }
  public String toString() { 
    return String.format("Employee(%d, %s)", id, name); 
  }
}