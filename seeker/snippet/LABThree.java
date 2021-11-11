//date: 2021-11-11T17:03:36Z
//url: https://api.github.com/gists/4014eb4f540c282d5e1b728984a50f8f
//owner: https://api.github.com/users/burak20101


package labthree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;


public class LABThree {
    
    static Random r  =  new Random();
    static ArrayList<String> cities = new ArrayList<String>();  //sehirleri arraylist metodlarını kullanabilmek için cities arraylistini olusturdum
    static String[] citiesInList = {"Istanbul", "Roma", "Paris","Madrid", "New York", "Moskova", "Londra" , "Pekin"};   //şehirlerin listede bulunması
    static final int SECILECEKSEHIRSAYISI = 2;
    static final String Istanbul = "Istanbul";  //d bolumu için istanbul gerekli oldugundan ve listedeki yazılıs seklini buraya atayarak saglamlastırmak istedim
    
    
    
    public static void main(String[] args) {
        addCitiesInArrayList(); //listedeki elemanların arrayliste eklenmesi
        
        String[] citiesForTravelPlan = new String[citiesInList.length/2];   //listedeki elemanların yarısı kadarı seçileceği için listenin uzunlugunu ona gore ayarladım
        
        addCitiesTravelPlan(citiesForTravelPlan);   //secilecek şehirlerin travel plan listesine eklenmesi
        
        SehirCifti[] sehirCiftleri = new SehirCifti[findTheCombinationCount(citiesForTravelPlan.length, SECILECEKSEHIRSAYISI)]; // secilen sehirlerden kac tane cift cıkabilecegini kombinasyonla hesaplayarak o sayıya gore liste uzunlugu olusturdum
        
        addCitiesInSehirCiftleri(sehirCiftleri, citiesForTravelPlan);   //secilen sehirleri sehir ciftlerine kombinasyonlayarak ekleyen kısım
        
        printTheSehirCiftleri(sehirCiftleri);   //sehir ciftlerinin ozelliklerinin yazdırılması
                
        printTheSehirCiftleriWhichHasIstanbul(sehirCiftleri);   //sehir ciftlerinde istanbul bulunanların yazdırılması
    }
    
    public static void addCitiesInArrayList(){  // şehirleri arrayliste ekleyen metod
        cities.addAll(Arrays.asList(citiesInList));
    }
    
    public static void addCitiesTravelPlan(String[] citiesForTravelPlan){   //sehirleri random metodu sayesınde rastgele secıp listeye ekleyip sonrasında arraylistten silen metod
        
        for(int a = 0; a<citiesForTravelPlan.length;a++){
            
            int indexOfCity =r.nextInt(cities.size());
            
            citiesForTravelPlan[a] = cities.get(indexOfCity);
            cities.remove(indexOfCity);
            
        }
  
    }
    
    public static int findTheNumbersFactorial(int number){  //gelen sayının faktöriyelini bulan metod
        
        int returnNumber = 1;
        
        for(int a = number; a>0 ; a--){
            returnNumber*=a;
        }
        
        return returnNumber;
        
    }
    
    public static int findTheCombinationCount(int numbersOfCities , int selectionAmount){   //gelen sayının kombinasyonunu bulan metod
        
        return ((findTheNumbersFactorial(numbersOfCities))/(findTheNumbersFactorial(numbersOfCities- selectionAmount)*findTheNumbersFactorial(selectionAmount)));
    }
    
    public static void addCitiesInSehirCiftleri(SehirCifti[] sehirCiftleri , String[] citiesForTravelPlan){ //sehirleri ciftler olacak sekilde gruplayıp kombinasyon yardımıyla listeye ekleyen metod
       
        int number = 0;
        
        for(int a = 0; a<citiesForTravelPlan.length-1 ; a++){
            
            for(int b = a+1; b<citiesForTravelPlan.length;b++){
                
                sehirCiftleri[number] = new SehirCifti(citiesForTravelPlan[a], citiesForTravelPlan[b]);
                number++;
            }
        }
    }
    
    
    public static void printTheSehirCiftleri(SehirCifti[] sehirCiftleri){   //sehir ciftlerini yazdıran metod
        
        System.out.println("-------BUTUN SEHIR CIFTLERININ BILGILERI-------");
        for(int a = 0 ; a<sehirCiftleri.length ; a++){
            System.out.println(sehirCiftleri[a].toString());
        
        }
    }
    
    public static void printTheSehirCiftleriWhichHasIstanbul (SehirCifti[] sehirCiftleri){  //ıcınde ıstanbul olan sehir ciftlerini yazdıran metod boolean ifade for dongusu ıcınde true ya donusmezse sehir ciftlerinde istanbul yoktur
        
        System.out.println("\n\n-------ICINDE ISTANBUL OLAN BUTUN SEHIR CIFTLERININ BILGILERI-------");
        boolean inList = false;
        
        for(int a = 0 ; a<sehirCiftleri.length;a++){
            if(sehirCiftleri[a].getFirstCity().equals(Istanbul) || sehirCiftleri[a].getSecondCity().equals(Istanbul)){
                System.out.println(sehirCiftleri[a].toString());
                inList = true;
            }
        }
 
        if(!inList){
            System.out.println("Şehir Çiftleri İçerisinde İstanbul Bulunmamaktadır...");
        }
    } 
}
