//date: 2023-03-01T16:53:48Z
//url: https://api.github.com/gists/e4e522e385de4255c58a0eb8ce90a3c6
//owner: https://api.github.com/users/arikamat

public class TestMethods{
	public static void main(String[] args) {
		MyMethods m = new MyMethods();
		System.out.println(m.volume(2, 3, 4));
		Die d1 = new Die();
		Die d2 = new Die();
		d1.roll();
		d2.roll();
		System.out.println(m.avgFaceValues(d1, d2));
	}
}