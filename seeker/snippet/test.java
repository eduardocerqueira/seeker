//date: 2022-09-13T17:16:47Z
//url: https://api.github.com/gists/459d9cc2d8b954048085e2661cd400bf
//owner: https://api.github.com/users/progeon

import java.util.Scanner;

public class Main
{
	public static void main(String[] args)
	{
		Double	deposit,
						price;

		Scanner scanner = new Scanner(System.in);

		System.out.println("Prog Academy Exchange");
		System.out.println("Введите курс BTC/USDT");
		price = scanner.nextDouble();

		System.out.println("Сколько средств ($) готовы вложить?");
		deposit = scanner.nextDouble();

		System.out.println("Можно приобрести " + deposit / price + " BTC");
	}
}