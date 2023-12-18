//date: 2023-12-18T17:09:54Z
//url: https://api.github.com/gists/905f9c8f07e81e2d38abfb98b6aec570
//owner: https://api.github.com/users/AndHov09

// Method to convert a Roman numeral to an integer
public static int romanToInteger(String s) 
{
    // Map to store the values of Roman numerals
    Map<Character, Integer> romanToIntegerMap = new HashMap<Character, Integer>() 
    {
      {
        put('I', 1);
        put('V', 5);
        put('X', 10);
        put('L', 50);
        put('C', 100);
        put('D', 500);
        put('M', 1000);
      }
    };

    // Initialize the sum with the value of the last character in the Roman numeral
    int sum = romanToIntegerMap.get(s.charAt(s.length() - 1));

    // Iterate through the Roman numeral from right to left
    for (int i = s.length() - 2; i >= 0; --i) 
    {
        // If the value of the current character is less than the value of the next character, subtract it
        if (romanToIntegerMap.get(s.charAt(i)) < romanToIntegerMap.get(s.charAt(i + 1))) 
        {
            sum -= romanToIntegerMap.get(s.charAt(i));
        } else {
            // Otherwise, add its value to the sum
            sum += romanToIntegerMap.get(s.charAt(i));
        }
    }

    // Return the final sum
    return sum;
}