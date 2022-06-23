//date: 2022-06-23T17:03:12Z
//url: https://api.github.com/gists/6b4bbdaf9327c18d05ae292d8918fa20
//owner: https://api.github.com/users/harshani2427

public class StudentTelescope2 {
	private String id;
	private String name;
	private Boolean degree;
	
	public StudentTelescope2(String id,String name,Boolean degree) {
		this.id = id;
		this.name = name;
		this.degree = degree;
	}
	
	
	public StudentTelescope2(String id,String name ) {
		this(id,name,null);
	
	}


	public StudentTelescope2(String id) {
		this(id,null);
	}

	@Override
	public String toString() {
		return "StudentTelescope1 [id=" + id + ", name=" + name + ", degree=" + degree + "]";
	}

	
}