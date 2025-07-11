//date: 2025-07-11T17:09:54Z
//url: https://api.github.com/gists/180957fa192f8d61c9b83fb478a56d02
//owner: https://api.github.com/users/vivianito

public class Codigo5 {

	    Scanner s = new Scanner();
	    System.out.print("Introduzca un número: ');
	    String ni = s.nextLine();
	    int c = ni;
	    
	    int afo = 0;
	    int noAfo = 0;
	    
	    while (ni > 0) {
		  int digito = (int)(ni % 10);
	      if ((digito == 3) || (digito == 7) || (digito == 8) || (digito == 9)) {
			afo++;
	      } else {
			noAfo++;
		  
		  ni /= 10;
	    }

	    if (afo > noAfo) {
	      System.out.prinln("El " + c + " es un número afortunado.");
	    } else {
	      System.out.println("El " + c + " no es un número afortunado.");
	    }
	    
	  }
	  
	}