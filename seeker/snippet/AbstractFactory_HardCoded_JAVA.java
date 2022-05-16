//date: 2022-05-16T17:07:17Z
//url: https://api.github.com/gists/ec6a00270f770ff3a1d8b162bb35062c
//owner: https://api.github.com/users/skipsleep

import java.util.*;
public class abstr {
    //product interface for bgcolor
	public static interface BGCOLOUR {
	////operations for bgcolour	
	}

	//concrete products
	public static class light_bgcolour implements BGCOLOUR {
	 ////implementing operations for light-theme bgcolour
	}

	public static class dark_bgcolour implements BGCOLOUR {
 	////implementing operations for dark-theme bgcolour
	}

	//product interface for text
	public static interface TEXT {
	////operations for text
	}

	//concrete products (TEXT)
	public static class light_txt implements TEXT {
	////implementing operations for light-theme text
	}

	public static class dark_txt implements TEXT {
	////implementing operations for dark-theme text
	}

	/*
Client- instantiates theme-based parameterized products
*/
public static class Client{
    
        BGCOLOUR bgcolour;
        TEXT text;

        // paramterized constructors for theme-based
        // instantiation of concrete products
    public Client (String theme){
            if (theme.equals("Light")) {

                this.bgcolour = new light_bgcolour();
                this.text = new light_txt();
            }
            else {
                 System.out.println("andar2\n");
                this.bgcolour = new dark_bgcolour();
                this.text = new dark_txt();
            }
        }
		//// operations using bgcolour and text
	/*
 	this class gets COUPLED with all the 4 concrete products.
 	And also hard to modify in future extensions!
	*/
    }

	public static void main(String[] args) {
        Scanner sc = new Scanner (System.in);

		String input_theme = sc.nextLine();

        Client client = new Client(input_theme);
    }
	}
        



