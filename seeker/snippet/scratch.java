//date: 2022-09-29T17:16:33Z
//url: https://api.github.com/gists/6fe23925a621b7fcd66ab0e969917ceb
//owner: https://api.github.com/users/fida10

public class scratch {

	public static void main(String[] args) {
		System.out.println(new scratch().defangIPaddr("1.1.1.1"));

		System.out.println(new scratch().sum(5, 6));
		System.out.println(new scratch().size("Hello"));
	}
	public String defangIPaddr(String address) {
		String answer = address.replace(".", "[.]");
		return answer;
	}

	public int sum(int a, int b){
		int c = a + b;
		return c;
	}

	public boolean size(String someString){
		if(someString.length() > 5){
			return true;
		} else {
			return false;
		}
	}
}