//date: 2022-09-23T17:05:48Z
//url: https://api.github.com/gists/61b7f91c02f1068d8e0f8774a35e31c7
//owner: https://api.github.com/users/ZtubaAcat


import java.util.Scanner;

public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		double tutar, kdv = 0.18;
		Scanner keyboard = new Scanner (System.in);
		
		System.out.print("Ücret Tutarını Giriniz : ");
		tutar = keyboard.nextDouble();
		
		if (tutar>=0 && tutar <= 100) {
			kdv = 0.18;
		}
		else if(tutar > 1000) {
			kdv = 0.8;
		}
		
		System.out.println("KDV'siz tutar " +tutar);
		double nkdv = tutar*kdv;
		tutar = nkdv+tutar;
		System.out.println("KDV'li tutar = " +tutar);
		
		System.out.println("KDV tutarı " +nkdv);
		
		

	}

}