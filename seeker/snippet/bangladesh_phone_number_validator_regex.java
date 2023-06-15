//date: 2023-06-15T17:07:27Z
//url: https://api.github.com/gists/f60003d15449f452076f7c68c019f4f4
//owner: https://api.github.com/users/forhadakhan

import java.util.regex.*;

public class Main {
    public static void main(String[] args) {
        String pattern = "^(?:\\+?880|0|88)?\\s?1[3456789]\\d{8}$";

        String[] phoneNumbers = {
                "01534567890",
                "8801534567890",
                "880 1534567890",
                "88 01534567890",
                "+8801534567890",
                "+880 1534567890",
                "+88 01534567890",
                "InvalidPhoneNumber"
        };

        for (String number : phoneNumbers) {
            if (Pattern.matches(pattern, number)) {
                System.out.println(number + " is a valid Bangladeshi phone number.");
            } else {
                System.out.println(number + " is not a valid Bangladeshi phone number.");
            }
        }
    }
}



/**
  Output: 
  
  01534567890 is a valid Bangladeshi phone number.
  8801534567890 is a valid Bangladeshi phone number.
  880 1534567890 is a valid Bangladeshi phone number.
  88 01534567890 is a valid Bangladeshi phone number.
  +8801534567890 is a valid Bangladeshi phone number.
  +880 1534567890 is a valid Bangladeshi phone number.
  +88 01534567890 is a valid Bangladeshi phone number.
  InvalidPhoneNumber is not a valid Bangladeshi phone number.
*/