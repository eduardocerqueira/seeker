//date: 2021-11-11T17:03:36Z
//url: https://api.github.com/gists/4014eb4f540c282d5e1b728984a50f8f
//owner: https://api.github.com/users/burak20101

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package labthree;

import java.util.Random;

/**
 *
 * @author Mahmut
 */


public class SehirCifti {   //SehirCifti classı
    private String firstCity;   //olusacak nesnenin özellikleri
    private String secondCity;
    private int point;
    
    Random r  = new Random();   //random classı
    
    public SehirCifti(String firstCity, String secondCity){ //main constructor
        this.firstCity = firstCity;
        this.secondCity = secondCity;
        this.point = r.nextInt(101);
    }

    public String getFirstCity() {  //ozelliklere ulasabilmek için getter ve setterlar tanımladım
        return firstCity;
    }

    public void setFirstCity(String firstCity) {
        this.firstCity = firstCity;
    }

    public String getSecondCity() {
        return secondCity;
    }

    public void setSecondCity(String secondCity) {
        this.secondCity = secondCity;
    }

    public int getPoint() {
        return point;
    }

    public void setPoint(int point) {
        this.point = point;
    }

    @Override
    public String toString() {  //tostring metodu sayesınde olusan nesnelerin özelliklerini tek metodla yazdırabiliyorum
        return "Birinci Sehir: "+ getFirstCity() + " İkinci Sehir: " + getSecondCity() + " Puan: "+ getPoint();
    }  
}
