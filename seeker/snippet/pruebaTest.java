//date: 2023-05-19T16:41:12Z
//url: https://api.github.com/gists/eecda6a30850642a3ef6bd5302cc0f31
//owner: https://api.github.com/users/kevinber789


/*CLASE */
package Transporte;

public class pruebaTest {

	public static void main(String[] args) {
		  int r=0;
		  for (int i=1, j=0; i<=9; i=i+4,j=j+2) {
			  try {
				  r+= i/j;
			  }catch (ArithmeticException e) {
				  r=20; 
			  }catch (Exception e) {
				  r=10;
			  }
		  } 
		  System.out.println(r);


	}

}
