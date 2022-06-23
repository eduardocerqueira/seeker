//date: 2022-06-23T17:00:41Z
//url: https://api.github.com/gists/0abffc22ad4c0ab249805f60d8654153
//owner: https://api.github.com/users/harshani2427

public class StudentTelescope1 {
	private String id;
	private String name;
	private Boolean degree;
	
	
	public StudentTelescope1(String id) {
		this.id = id;
	}
	
	public StudentTelescope1(String id,String name ) {
		this(id);
		this.name = name;
		
	}

	public StudentTelescope1(String id,String name,Boolean degree) {
		this(id, name);
		this.degree = degree;
	}

	@Override
	public String toString() {
		return "StudentTelescope1 [id=" + id + ", name=" + name + ", degree=" + degree + "]";
	}

	
}