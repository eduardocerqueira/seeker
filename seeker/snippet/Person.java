//date: 2022-04-12T16:51:43Z
//url: https://api.github.com/gists/a60e3672c40c3d575d5c43299cd743bc
//owner: https://api.github.com/users/jeyakeerthanan

public class Person{
  private Name name;
  private Address address;
  
  public Person(Person otherPerson){
    this.name=new Name(otherPerson.name);
    this.address=new Address(otherPerson.address);
  }
}