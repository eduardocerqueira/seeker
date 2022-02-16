//date: 2022-02-16T17:00:41Z
//url: https://api.github.com/gists/38475f76638f256af8f4a548a7fd20ef
//owner: https://api.github.com/users/Saifuddin-Shaikh

@Scope(scopeName = "singleton", proxyMode = ScopedProxyMode.NO)
@Component
@ToString
public class ScopeBeanExample {

	private String name;

	public ScopeBeanExample() {
		super();
		this.name = "Rock";
	}

	public String getName() {
		return name;
	}
}
