//date: 2026-03-04T17:23:42Z
//url: https://api.github.com/gists/528df9ddca8490af5d6e0e295547cc57
//owner: https://api.github.com/users/klikolio

import java.util.ArrayList;
import java.util.Scanner;

public class Main {

	public static void main(String[] args) {
		Scanner scanner = new Scanner(System.in);
		
		System.out.print("Masukkan jumlah barang: ");
		int jumlah_barang = scanner.nextInt();
		scanner.nextLine();
		
		ArrayList<String> daftar_nama = new ArrayList<>();
		ArrayList<Integer> daftar_stok = new ArrayList<>();
		ArrayList<Double> daftar_harga = new ArrayList<>();
		
		for (int no_barang = 0; no_barang < jumlah_barang; no_barang++) {
			System.out.println("Barang ke-" + (no_barang + 1));
			
			System.out.print("Masukkan nama barang: ");
			String nama = scanner.nextLine();
			
			System.out.print("Masukkan stok barang: ");
			int stok = scanner.nextInt();
			
			System.out.print("Masukkan harga barang: ");
			double harga = scanner.nextDouble();
			scanner.nextLine();
			
			daftar_nama.add(nama);
			daftar_stok.add(stok);
			daftar_harga.add(harga);
		}
		
		System.out.println("== Data Inventaris ==");
		System.out.println("Nama | Stok | Harga");
		
		for (int no_barang = 0; no_barang < jumlah_barang; no_barang++) {
			System.out.print(daftar_nama.get(no_barang));
			System.out.print(" | ");
			System.out.print(daftar_stok.get(no_barang));
			System.out.print(" | ");
			System.out.print(daftar_harga.get(no_barang));
			System.out.println("");
		}
		
		scanner.close();
	}

}
