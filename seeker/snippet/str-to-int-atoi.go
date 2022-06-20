//date: 2022-06-20T17:08:00Z
//url: https://api.github.com/gists/4ace5da06c9de8008cf2d324c84db41a
//owner: https://api.github.com/users/jcksnvllxr80

/*
Implement the myAtoi(string s) function, which converts a string to a 32-bit 
signed integer (similar to C/C++'s atoi function).

The algorithm for myAtoi(string s) is as follows:

Read in and ignore any leading whitespace.
Check if the next character (if not already at the end of the string) is '-' or '+'. 
Read this character in if it is either. This determines if the final result is negative 
or positive respectively. Assume the result is positive if neither is present.
Read in next the characters until the next non-digit character or the end of the input is 
reached. The rest of the string is ignored.
Convert these digits into an integer (i.e. "123" -> 123, "0032" -> 32). If no digits were read, 
then the integer is 0. Change the sign as necessary (from step 2).
If the integer is out of the 32-bit signed integer range [-231, 231 - 1], 
then clamp the integer so that it remains in the range. Specifically, integers 
less than -231 should be clamped to -231, and integers greater than 231 - 1 should be 
clamped to 231 - 1.
Return the integer as the final result.
Note:

Only the space character ' ' is considered a whitespace character.
Do not ignore any characters other than the leading whitespace or the rest of the string after the digits.

Constraints:
0 <= s.length <= 200
s consists of English letters (lower-case and upper-case), digits (0-9), ' ', '+', '-', and '.'.

Runtime: 9 ms, faster than 8.87% of Go online submissions for String to Integer (atoi).
Memory Usage: 2.3 MB, less than 34.03% of Go online submissions for String to Integer (atoi).
*/
func myAtoi(s string) int {
    sLen := len(s);
    if sLen == 0 {return 0}
    
    s = strings.Trim(s, " ");
    sLen = len(s);
    if sLen == 0 {return 0}
    
    pos := s[0] == '+' || s[0] >= '0' && s[0] <= '9';
    if s[0] == '+' || s[0] == '-' {
        s = s[1:]
    }

    sLen = len(s);
    intStr := "";
    for i := 0; i < sLen; i++ {
        if s[i] < '0' || s[i] > '9' {
            break;
        } else {
            intStr += string(s[i]);
        }
    }
    
    if len(intStr) == 0 {return 0}
    
    outInt, err := strconv.Atoi(intStr)
    if err != nil {
        if pos {
            return int(math.Pow(2,31) - 1);
        } else {
            return int(- math.Pow(2,31));
        }
    } else {
        if pos {
            return int(math.Min(float64(outInt), math.Pow(2,31) - 1));            
        } else {
            return int(math.Max(float64(-1 * outInt), - math.Pow(2,31)));
        }
    }
}