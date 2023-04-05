//date: 2023-04-05T16:53:45Z
//url: https://api.github.com/gists/897fe9f26a5b5f8ac4e588ecbb80589a
//owner: https://api.github.com/users/Chiquinho-Coder

import java.util.Scanner;
public class Main {


   public static void main(String[] args) {

  double temp;
  String msg;
  Scanner ler = new Scanner(System.in);
     
     
     System.out.println("Informe sua temperatura");
       temp = ler.nextDouble();
       msg = febreSN(temp);

     System.out.println("Sua situaçao: "+msg);

     



   }
public static String febreSN(double n1){
String febreSN = "";

  if(n1<36){
    febreSN = "Hipotermia";
  }
  if((n1>=36)&&(n1<37.6)){
    febreSN = "Normal";
  }
  if((n1>=37.6)&&(n1<39.6)){
    febreSN = "Febre";
  }
  if((n1>=39.6)&&(n1<41)){
    febreSN = "Febre alta";
  }
  if(n1>=41){
    febreSN = "Hipertermia";
  }
return febreSN;
  }
  
}