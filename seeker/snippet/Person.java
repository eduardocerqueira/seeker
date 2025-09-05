//date: 2025-09-05T16:57:33Z
//url: https://api.github.com/gists/ec2f142b8982bf91c4d0082df27bb3d4
//owner: https://api.github.com/users/HitotsumeGod

import java.util.ArrayList;

public class Person {

    private final ArrayList<Person> children = new ArrayList<>();
    private String firstName;
    private String middleName;
    private String lastName;
    private int age;
    
    public Person(String firstName, String lastName, String middleName, int age) {
        
      	this.firstName = firstName;
        this.middleName = middleName;
        this.lastName = lastName;
        this.age = age;
        
    }
    
    public String getFirstName() { return firstName; }
    
    public String getMiddleName() { return middleName; }
    
    public String getLastName() { return lastName; }
        
    public int getAge() { return age; }
    
    public int getNumberOfChildren() { return children.size(); }

    public boolean hasChildren() { return children.isEmpty(); }
    
    public void procreate(Person person) { children.add(person); }
    
    public Person getChild(String firstName, String lastName) {
        
	    for (Person p : children)
		    if (p.getFirstName.equals(firstName) && p.getLastName.equals(lastName))
	    		return p;
		return null;
			
    }
    
    public Person getChild(String firstName, String middleName, String lastName) {
        
    	for (Person p : children)
	    	if (p.getFirstName.equals(firstName) && p.getMiddleName.equals(middleName) && p.getLastName.equals(lastName))
				return p;
		return null;
        
    }
    
    public Person getChild(int index) { return children.get(index); }
	
}