//date: 2024-05-24T17:08:03Z
//url: https://api.github.com/gists/42f5a22b56110d95858a705a36f32f91
//owner: https://api.github.com/users/Rachel519

package com.generation;

public class Codigo4 {

    Scanner s = new Scanner();
    
    System.out.print("Turno del jugador 1 (introduzca piedra, papel o tijeras): ");
    String j1 = s.nextLine();
    
    System.out.print("Turno del jugador 1 (introduzca piedra, papel o tijeras): ");
    Scanner s2 = new Scanner();
    String j2 = s.nextLine();
    
    if (j1 == j2)) {
      System.out.println("Empate");
    } else {
      int g = 2;
      switch(j1) {
        case "piedra":
          if (j2 == "tijeras") {
            g = 1;
          }

        case "papel":
          if (j2 == "piedra") {
            g = 1;
          
        case "tijera":
          if (j2.equals("papel")) {
            g = 1;
          }
          break;
        default:
      }
      System.out.println("Gana el jugador " + g);
    }
  
  
}