//date: 2021-10-15T16:55:35Z
//url: https://api.github.com/gists/d53c527499c6c7c6fd4a6119eaba1a75
//owner: https://api.github.com/users/mitulvaghamshi

public class Pattern {
	public static void main(String[] args) {
		int i;
		System.out.print("Enter pattern size: ");
		int n = new java.util.Scanner(System.in).nextInt();
		for (i = 0; i <= n; i++) print(i, n);
		for (i = n - 1; i >= 0; i--) print(i, n);
	}

	private void print(int i, int n) {
		boolean star = true;
		for (int b = n; b > i; b--) 
		System.out.print("  ");
		if (i % 2 == 0) 
		for (int j = i - 1; j > 0; j--) 
		System.out.printf("%s   ", (star = !star) ? "-" : "*");
		System.out.println();
	}

	private void swastic() {
		String c = "###############", c1 = "#####";
		for (int i = 1; i < 16; i++) {
			if (i < 4) System.out.printf("%s%20s\t\n", c1, c);
			else if (i > 3 && i < 7) System.out.printf("%s%10s\t\t\n", c1, c1);
			else if (i > 6 && i < 10) System.out.printf("%s%s%s\t\n", c1, c1, c);
			else if (i > 9 && i < 13) System.out.printf("%15s%10s\t\n", c1, c1);
			else if (i > 12 && i < 16) System.out.printf("%s%10s\t\n", c, c1);
		}
	}
}
