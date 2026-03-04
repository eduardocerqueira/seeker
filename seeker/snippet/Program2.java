//date: 2026-03-04T17:23:42Z
//url: https://api.github.com/gists/528df9ddca8490af5d6e0e295547cc57
//owner: https://api.github.com/users/klikolio

import java.util.Scanner;
import java.util.ArrayList;

public class Main {
    public static void main(String[] args) {
    	ArrayList<String> daftarNama = new ArrayList<>();
        ArrayList<Integer> daftarStok = new ArrayList<>();
        ArrayList<Double> daftarHarga = new ArrayList<>();
        
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Masukkan jumlah data: ");
        int n = scanner.nextInt();
        scanner.nextLine();        
        
        for (int i = 0; i < n; i++) {
            System.out.println("Barang ke-" + (i + 1));
            
            System.out.print("Nama barang: ");
            String nama = scanner.nextLine();
            daftarNama.add(nama);
            
            System.out.print("Stok barang: ");
            int stok = scanner.nextInt();
            daftarStok.add(stok);
            
            System.out.print("Harga barang: ");
            double harga = scanner.nextDouble();
            daftarHarga.add(harga);
            scanner.nextLine();
        }
        
        System.out.println("-- DATA INVENTARIS --");
        System.out.println("NAMA | STOK | HARGA");
        
        for (int i = 0; i < n; i++) {
            System.out.println(daftarNama.get(i) + " | " + daftarStok.get(i) + " | " + daftarHarga.get(i));
        }
        
        scanner.close();
    }
}