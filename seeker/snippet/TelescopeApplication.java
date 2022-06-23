//date: 2022-06-23T17:05:52Z
//url: https://api.github.com/gists/b8920bba215dffb953e2211e61223d87
//owner: https://api.github.com/users/harshani2427

public class TelescopeApplication {

	public static void main(String[] args) {
		
		StudentTelescope1 s1 = new StudentTelescope1("1");
		System.out.println(s1);  //StudentTelescope1 [id=1, name=null, degree=null]
		
		StudentTelescope2 s2 = new StudentTelescope2("2", "Sachini");
		System.out.println(s2);  //StudentTelescope1 [id=2, name=Sachini, degree=null]
	}

}