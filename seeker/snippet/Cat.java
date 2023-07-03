//date: 2023-07-03T16:58:20Z
//url: https://api.github.com/gists/7757de49e4eb3f8c96e3e3756ddf02f9
//owner: https://api.github.com/users/hguerrero

public class Cat {

	private final String name;

	public Cat(String name) {
		this.name = name;
	}

	public String getName() {
		return name;
	}

	public String say() {
		return name + " mjaus";
	}
}