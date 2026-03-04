//date: 2026-03-04T17:23:42Z
//url: https://api.github.com/gists/528df9ddca8490af5d6e0e295547cc57
//owner: https://api.github.com/users/klikolio

import java.util.ArrayList;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Jumlah data: ");
        int jumlah = scanner.nextInt();
        scanner.nextLine();

        ArrayList<String> listNama = new ArrayList<>();
        ArrayList<Integer> listUmur = new ArrayList<>();
        ArrayList<String> listAlamat = new ArrayList<>();
        
        for (int i = 0; i < jumlah; i++) {
            System.out.println("Data ke-" + (i + 1));
            
            System.out.print("Nama: ");
            String nama = scanner.nextLine();
            listNama.add(nama);
            
            System.out.print("Umur: ");
            int umur = scanner.nextInt();
            listUmur.add(umur);
            scanner.nextLine();
            
            System.out.print("Alamat: ");
            String alamat = scanner.nextLine();
            listAlamat.add(alamat);
        }

        System.out.println("Data Hasil");
        System.out.println("Nama | Umur | Alamat");
        for (int i = 0; i < jumlah; i++) {
            System.out.println(listNama.get(i) + " | " + listUmur.get(i) + " | " + listAlamat.get(i));
        }
        
        scanner.close();
    }
}