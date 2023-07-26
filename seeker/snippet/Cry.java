//date: 2023-07-26T16:48:49Z
//url: https://api.github.com/gists/63a111b673ec238638bd06f9d11d005f
//owner: https://api.github.com/users/SwarabRaul

1)Lexicographically Palindromic String

import java.util.*;
public class Main {
 
    static char MAX_CHAR = 26;
    static void countFreq(String str, int freq[], int len)
    {
        for (int i = 0; i < len; i++)
        {
            freq[str.charAt(i) - 'a']++;
        }
    }
    static boolean canMakePalindrome(int freq[], int len)
    {
        int count_odd = 0;
        for (int i = 0; i < MAX_CHAR; i++)
        {
            if (freq[i] % 2 != 0)
            {
                count_odd++;
            }
        }
        if (len % 2 == 0)
        {
            if (count_odd > 0)
            {
                return false;
            }
            else
            {
                return true;
            }
        }
        if (count_odd != 1)
        {
            return false;
        }
 
        return true;
    }
    static String findOddAndRemoveItsFreq(int freq[])
    {
        String odd_str = "";
        for (int i = 0; i < MAX_CHAR; i++)
        {
            if (freq[i] % 2 != 0)
            {
                freq[i]--;
                odd_str = odd_str + (char) (i + 'a');
                return odd_str;
            }
        }
        return odd_str;
    }
    static String findPalindromicString(String str)
    {
        int len = str.length();
        int freq[] = new int[MAX_CHAR];
        countFreq(str, freq, len);
 
        if (!canMakePalindrome(freq, len))
        {
            return "No Palindromic String";
        }
        String odd_str = findOddAndRemoveItsFreq(freq);
 
        String front_str = "", rear_str = " ";
         for (int i = 0; i < MAX_CHAR; i++)
        {
            String temp = "";
            if (freq[i] != 0)
            {
                char ch = (char) (i + 'a');
                for (int j = 1; j <= freq[i] / 2; j++)
                {
                    temp = temp + ch;
                }
                front_str = front_str + temp;
                rear_str = temp + rear_str;
            }
        }
 
        return (front_str + odd_str + rear_str);
    }
 
    public static void main(String[] args)
    {
        String str = "malayalam";
        System.out.println(findPalindromicString(str));
    }
}





2)Simple Seive
import java.util.*;

class Main{
    public static void main(String ar[]){
        Scanner sc =  new Scanner(System.in);
        int n = sc.nextInt();
        boolean[] bool = new boolean[n+1];
        for(int i=0;i<n;i++){
            bool[i] = true;
        }
        for(int i=2;i<Math.sqrt(n);i++){
            if(bool[i]==true){
                for(int j=i*i;j<n;j+=i){
                    bool[j] = false;
                }
            }
        }
        for(int i=0;i<n;i++){
            if(bool[i]==true){
                System.out.println(i);
            }
        }
    }
}




















3) Incremental Seive
import java.util.Vector;
import static java.lang.Math.sqrt;
import static java.lang.Math.floor;

class Main
{
	
	static void simpleSieve(int limit, Vector<Integer> prime)
	{
		
		boolean mark[] = new boolean[limit+1];
		
		for (int i = 0; i < mark.length; i++)
			mark[i] = true;
	
		for (int p=2; p*p<limit; p++)
		{
			if (mark[p] == true)
			{
				for (int i=p*p; i<limit; i+=p)
					mark[i] = false;
			}
		}
	
		for (int p=2; p<limit; p++)
		{
			if (mark[p] == true)
			{
				prime.add(p);
				System.out.print(p + " ");
			}
		}
	}
	
	static void segmentedSieve(int n)
	{
		
		int limit = (int) (floor(sqrt(n))+1);
		Vector<Integer> prime = new Vector<>();
		simpleSieve(limit, prime);
	
		
		int low = limit;
		int high = 2*limit;
	
		
		while (low < n)
		{
			if (high >= n)
				high = n;

			
			boolean mark[] = new boolean[limit+1];
			
			for (int i = 0; i < mark.length; i++)
				mark[i] = true;
	
			
			for (int i = 0; i < prime.size(); i++)
			{
				
				int loLim = (int) (floor(low/prime.get(i)) * prime.get(i));
				if (loLim < low)
					loLim += prime.get(i);
	
			
				for (int j=loLim; j<high; j+=prime.get(i))
					mark[j-low] = false;
			}
	
			for (int i = low; i<high; i++)
				if (mark[i - low] == true)
					System.out.print(i + " ");
	
			low = low + limit;
			high = high + limit;
		}
	}
	
	public static void main(String args[])
	{
		int n = 100;
		System.out.println("Primes smaller than " + n + ":");
		segmentedSieve(n);
	}
}



























4)Eular Phi
import java.util.Scanner;

class Main{
    public static void main(String ar[]){
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int result = n;
        for(int i=2;i*i<=n;i++){
            if(n%i==0){
                while(n%i==0){
                    n /= i;
                }
                result -= result/i;
            }
        }
        if(n>1){
            result -= result/n;
        }
        System.out.println(result);
    }
}




















5)Strobrogroammtic Nymber

import java.util.*;

class Main {

// definition function
static ArrayList<String> numdef(int n, int length)
{
	ArrayList<String> result = new ArrayList<String>();
	if (n == 0)
	return result;
	if (n == 1) {
	result.add("1");
	result.add("0");
	result.add("8");
	return result;
	}

	ArrayList<String> middles = numdef(n - 2, length);

	for (String middle : middles) {
	if (n != length)
		result.add("0" + middle + "0");

	result.add("8" + middle + "8");
	result.add("1" + middle + "1");
	result.add("9" + middle + "6");
	result.add("6" + middle + "9");
	}
	return result;
}

// strobogrammatic function
static ArrayList<String> strobogrammatic_num(int n)
{
	ArrayList<String> result = numdef(n, n);
	return result;
}

// Driver Code
public static void main(String[] args)
{
	// Print all Strobogrammatic
	// number for n = 3
	for (String num : (strobogrammatic_num(3)))
	System.out.print(num + " ");
}
}

// This code is contributed by phasing17

















6)Remainder theoram
import java.util.*;

class Main{

    static int remainder(int num[], int rem[],int n){
        int x = 1;
        while(true){
            int j;
            for(j=0;j<n;j++){
                if(x%num[j]!=rem[j])
                    break;
            }
            if(j==n){
                return x;
            }
            x++;
        }
    }

    public static void main(String ar[]){
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();

        int a[] = new int[n];
        int m[] = new int[n];
        
        for(int i=0;i<n;i++){
            a[i] = sc.nextInt();
        }
        for(int i=0;i<n;i++){
            m[i] = sc.nextInt();
        }

        System.out.println(remainder(a,m,n));

    }
}


















7)Alice Apple
import java.util.*;

class Main{
    public static void main(String ar[]){
        Scanner sc = new Scanner(System.in);
        int m = sc.nextInt();
        int k = sc.nextInt();
        int n = sc.nextInt();
        int s = sc.nextInt();
        int w = sc.nextInt();
        int e = sc.nextInt();

        if(k*s >= m){
            System.out.println(m);
        }else if (m<= (k*s + w + e)){
            System.out.println(s*k + (m-s*k)*k);
        }else{
            System.out.println(-1);
        }

    }
}













8)Binary Pallindrome
import java.util.*;

class Main{
    public static void main(String ar[]){
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        String bin = Integer.toBinaryString(n);
        
        int j = bin.length()-1;
        int i = 0;
        int flag = 1;
        while(i<j){
            if(bin.charAt(i) != bin.charAt(j)){
                flag = 0;

            }
            i++;
            j--;
        }
        System.out.println(flag);
    }
}















9)Booth Algo
import java.util.Scanner;

public class Main {
    public static int boothMultiplication(int m, int r) {
        int a = m;
        int b = r;

        int product = 0;
        int count = Integer.toBinaryString(b).length();
        int addBit = 0;

        for (int i = 0; i < count; i++) {
            int lastBit = b & 1;

            if (lastBit == 1 && addBit == 0) {
                product += a;
            } else if (lastBit == 0 && addBit == 1) {
                product -= a;
            }

            addBit = lastBit;

            a <<= 1;
            b >>= 1;
        }

        return product;
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.print("Enter the first number: ");
        int m = scanner.nextInt();

        System.out.print("Enter the second number: ");
        int r = scanner.nextInt();

        int result = boothMultiplication(m, r);
        System.out.println("The result of multiplication is: " + result);
    }
}


















10) Euclid Algo
import java.util.Scanner;

class Main{

    static int gcd(int a,int b){
        if(a==0)
            return b;
        return gcd(b%a,a);
    }

    public static void main(String ar[]){
        Scanner sc = new Scanner(System.in);
        int n1 = sc.nextInt();
        int n2 = sc.nextInt();
        System.out.println(gcd(n1,n2));
    }
}



















11) Karatsuba Algo
import java.util.Scanner;
import java.math.*;

class Main{

    static BigInteger karatsuba(BigInteger x, BigInteger y){
        int n = Math.max(x.bitLength(),y.bitLength());
        if(n<1000){
            return x.multiply(y);
        }
        int half = (n+32)/64;
        BigInteger mask = BigInteger.ONE.shiftLeft(n).multiply(BigInteger.ONE);

        BigInteger xLow = x.and(mask);
        BigInteger yLow = y.and(mask);
        BigInteger xHigh = x.shiftRight(half);
        BigInteger yHigh = y.shiftRight(half);

        BigInteger z0 = karatsuba(xLow,yLow);
        BigInteger z1 = karatsuba(xLow.add(xHigh),yLow.add(yHigh));
        BigInteger z2 = karatsuba(xHigh,yHigh);

        BigInteger result = z2.shiftLeft(2*half).add(z1.subtract(z0).subtract(z2).shiftLeft(half)).add(z0);
        return result;

    }

    public static void main(String ar[]){
        Scanner sc = new Scanner(System.in);

        BigInteger x = sc.nextBigInteger();
        BigInteger y = sc.nextBigInteger();

        System.out.println(karatsuba(x,y));

    }
}
















12)Flipping Bit
import java.util.*;

class Main
{

	static int flipBit(int a)
	{
		/* If all bits are l, binary representation
		of 'a' has all 1s */
		if (~a == 0)
		{
			return 8 * sizeof();
		}

		int currLen = 0, prevLen = 0, maxLen = 0;
		while (a != 0)
		{
			// If Current bit is a 1
			// then increment currLen++
			if ((a & 1) == 1)
			{
				currLen++;
			}
			
			// If Current bit is a 0 then
			// check next bit of a
			else if ((a & 1) == 0)
			{
				/* Update prevLen to 0 (if next bit is 0)
				or currLen (if next bit is 1). */
				prevLen = (a & 2) == 0 ? 0 : currLen;

				// If two consecutively bits are 0
				// then currLen also will be 0.
				currLen = 0;
			}

			// Update maxLen if required
			maxLen = Math.max(prevLen + currLen, maxLen);

			// Remove last bit (Right shift)
			a >>= 1;
		}

		// We can always have a sequence of
		// at least one 1, this is flipped bit
		return maxLen + 1;
	}

	static byte sizeof()
	{
		byte sizeOfInteger = 8;
		return sizeOfInteger;
	}
	
	// Driver code
	public static void main(String[] args)
	{
		// input 1
		System.out.println(flipBit(13));

		// input 2
		System.out.println(flipBit(1775));

		// input 3
		System.out.println(flipBit(15));
	}
}

















13)Swap 2 Nibbles in a byte
import java.util.*;

class Main{
    public static void main(String ar[]){
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int a = ((n&0xF0) >> 4) | ((n&0x0F) << 4);
        System.out.println(a);
    }
}







14)Bock Swap Algo:
import java.util.*;

class Main{
    public static void main(String ar[]){
        Scanner sc = new Scanner(System.in);
        
        int n = sc.nextInt();
        int arr[] = new int[n];
        for(int i=0;i<n;i++){
            arr[i] = sc.nextInt();
        }

        int k = sc.nextInt();
        int temp[] = new int[k];
        for(int i=0;i<k;i++){
            temp[i] = arr[i];
        }
        for(int i=0;i<n-k;i++){
            arr[i] = arr[i+k];
        }
        for(int i=0;i<k;i++){
            arr[i+n-k] = temp[i];
        }
        for(int i=0;i<n;i++){
            System.out.print(arr[i] + " ");
        }

    }
}

















15)Max Product Sub Array:
import java.util.*;

class MaxProductSubarray{
    public static void main(String ar[]){
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int arr[] = new int[n];

        for(int i=0;i<n;i++){
            arr[i] = sc.nextInt();
        }

        int p=1;
        int s=1;
        int ans=0;
        for(int i=0;i<n;i++){
            if(p==0){
                p = 1;
            }
            if(s==0){
                s=1;
            }
            p *= arr[i];
            s *= arr[n-i-1];
            ans = Math.max(ans,Math.max(p,s));
        }
        System.out.println(ans);
        

    }
}















16)Hourglass:
import java.util.Scanner;

class Hourglass{
    public static void main(String ar[]){
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int m = sc.nextInt();
        int arr[][] = new int[n][m];

        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                arr[i][j] = sc.nextInt();
            }
        }

        int ans = 0;
        for(int i=0;i<n-2;i++){
            for(int j=0;j<m-2;j++){
                int temp = arr[i][j] + arr[i][j+1] + arr[i][j+2] + arr[i+1][j+1] + arr[i+2][j] + arr[i+2][j+1] + arr[i+2][j+2];
                ans = Math.max(ans,temp);
            }
        }
        System.out.println(ans);

    }
}
















17)MaxEquilSum

import java.util.Scanner;

class MaxEquilibSum{
    public static void main(String ar[]){
        Scanner sc = new Scanner(System.in);

        int arr[] = {-2,5,3,3,2,6,-4,2};
        int n = arr.length;

        int pre[] = new int[n];
        int suf[] = new int[n];
        int ans = -100;

        pre[0] = arr[0];
        for(int i=1;i<n;i++){
            pre[i] = pre[i-1] + arr[i];
        }
        suf[n-1] = arr[n-1];
        if(suf[n-1] == pre[n-1]){
            ans = Math.max(ans,pre[n-1]);
        }
        for(int i=n-2;i>=0;i--){
            suf[i] = suf[i+1] + arr[i];
            if(suf[i]==pre[i]){
                ans = Math.max(ans,pre[n-1]);
            }
        }

        System.out.println(ans);

    }
}


















18)Leaders In an array:
// This code is self-made and not provided by ethnus team. Time complexity of this code is more than theirs;
import java.util.Scanner;

class Leader{
    public static void main(String ar[]){
        // Scanner sc = new Scanner(System.in);
        // int n = sc.nextInt();
        // int arr[] = new int[n];
        int arr[] = {10,2,3,8,5,4,6,1};
        int n = 8;
        // for(int i=0;i<n;i++){
        //     arr[i] = sc.nextInt();
        // }
        System.out.println("Leaders");
        int i=0, j=n-1;
        while(i<n){
            if(arr[i]>arr[j]){
                j--;
            }else if (arr[i]<arr[j]){
                i++;
            }else if(arr[i]==arr[j]){
                System.out.println(arr[i]);
                j=n-1;
                i++;
            }
        }
    }
}















19)Majority Element

// Solution is not by ethnus but by me. Time complexity might be different
import java.util.Scanner;

class MajorityElement{
    public static void main(String ar[]){
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int arr[] = new int[n];
        int max = -100;
        for(int i=0;i<n;i++){
            arr[i] = sc.nextInt();
            max = Math.max(max,arr[i]);
        }
        int temp[] = new int[max+1];
        for(int i=0;i<max;i++){
            temp[i] = 0;
        }

        for(int i=0;i<n;i++){
            temp[arr[i]]++;
        }
        int t = 0;
        int ele = 0;
        for(int i=0;i<max;i++){
            t = Math.max(t,temp[i]);
        }
        for(int i=0;i<max;i++){
            if(t==temp[i]){
                ele = arr[i];
            }
        }
        System.out.println(ele);

    }
}












20)SelectionSort:
import java.util.Scanner;

class SelectionSort{

    static void sort_arr(int arr[],int n){
        for(int i=0;i<n-1;i++){
            int min = i;
            for(int j=i+1;j<n;j++){
                if(arr[j]<arr[min]){
                    min = j;
                }
            }
            int temp = arr[i];
            arr[i] = arr[min];
            arr[min] = temp;
        }
    }

    public static void main(String ar[]){
        int arr[] = {10,9,8,7,6,5,4,3,2,1};
        int n = 10;
        sort_arr(arr,n);
        for(int i=0;i<n;i++){
            System.out.print(arr[i]+" ");
        }
    }
}


















21)Hyphen to the beginning:
import java.util.*;

public class Hiphen{
    
    static void moveSpaceInFront(char str[])
    {
        int i = str.length-1;
        for (int j = i; j >= 0; j--)
            if (str[j] != '-')
            {
                char c = str[i];
                str[i] = str[j];
                str[j] = c;
                i--;
            }
    }   
 
    public static void main(String[] args)
    {
        Scanner sc=new Scanner(System.in);
        System.out.println("Enter String:");
        String s1=sc.next();
        char str[] = s1.toCharArray();
        moveSpaceInFront(str);
        System.out.println(String.valueOf(str));
    }
}
    




22)Manacher's Algo:
// Java program to implement Manacher's Algorithm
import java.util.*;

class Main
{
	static void findLongestPalindromicString(String text)
	{
		int N = text.length();
		if (N == 0)
			return;
		N = 2 * N + 1; // Position count
		int[] L = new int[N + 1]; // LPS Length Array
		L[0] = 0;
		L[1] = 1;
		int C = 1; // centerPosition
		int R = 2; // centerRightPosition
		int i = 0; // currentRightPosition
		int iMirror; // currentLeftPosition
		int maxLPSLength = 0;
		int maxLPSCenterPosition = 0;
		int start = -1;
		int end = -1;
		int diff = -1;

		// Uncomment it to print LPS Length array
		// printf("%d %d ", L[0], L[1]);
		for (i = 2; i < N; i++)
		{

			// get currentLeftPosition iMirror
			// for currentRightPosition i
			iMirror = 2 * C - i;
			L[i] = 0;
			diff = R - i;

			// If currentRightPosition i is within
			// centerRightPosition R
			if (diff > 0)
				L[i] = Math.min(L[iMirror], diff);

			// Attempt to expand palindrome centered at
			// currentRightPosition i. Here for odd positions,
			// we compare characters and if match then
			// increment LPS Length by ONE. If even position,
			// we just increment LPS by ONE without
			// any character comparison
			while (((i + L[i]) + 1 < N && (i - L[i]) > 0) &&
							(((i + L[i] + 1) % 2 == 0) ||
						(text.charAt((i + L[i] + 1) / 2) ==
						text.charAt((i - L[i] - 1) / 2))))
			{
				L[i]++;
			}

			if (L[i] > maxLPSLength) // Track maxLPSLength
			{
				maxLPSLength = L[i];
				maxLPSCenterPosition = i;
			}

			// If palindrome centered at currentRightPosition i
			// expand beyond centerRightPosition R,
			// adjust centerPosition C based on expanded palindrome.
			if (i + L[i] > R)
			{
				C = i;
				R = i + L[i];
			}

			// Uncomment it to print LPS Length array
			// printf("%d ", L[i]);
		}

		start = (maxLPSCenterPosition - maxLPSLength) / 2;
		end = start + maxLPSLength - 1;
		System.out.printf("LPS of string is %s : ", text);
		for (i = start; i <= end; i++)
			System.out.print(text.charAt(i));
		System.out.println();
	}

	// Driver Code
	public static void main(String[] args)
	{
		String text = "babcbabcbaccba";
		findLongestPalindromicString(text);

		text = "abaaba";
		findLongestPalindromicString(text);

		text = "abababa";
		findLongestPalindromicString(text);

		text = "abcbabcbabcba";
		findLongestPalindromicString(text);

		text = "forgeeksskeegfor";
		findLongestPalindromicString(text);

		text = "caba";
		findLongestPalindromicString(text);

		text = "abacdfgdcaba";
		findLongestPalindromicString(text);

		text = "abacdfgdcabba";
		findLongestPalindromicString(text);

		text = "abacdedcaba";
		findLongestPalindromicString(text);
	}
}












23)Permutations of substring:
// Java program to print all permutations of a string
// in sorted order.
import java.io.*;
import java.util.*;

class Main {

// Calculating factorial of a number
static int factorial(int n) {
	int f = 1;
	for (int i = 1; i <= n; i++)
	f = f * i;
	return f;
}

// Method to print the array
static void print(char[] temp) {
	for (int i = 0; i < temp.length; i++)
	System.out.print(temp[i]);
	System.out.println();
}

// Method to find total number of permutations
static int calculateTotal(char[] temp, int n) {
	int f = factorial(n);

	// Building HashMap to store frequencies of
	// all characters.
	HashMap<Character, Integer> hm =
					new HashMap<Character, Integer>();
	for (int i = 0; i < temp.length; i++) {
	if (hm.containsKey(temp[i]))
		hm.put(temp[i], hm.get(temp[i]) + 1);
	else
		hm.put(temp[i], 1);
	}

	// Traversing hashmap and finding duplicate elements.
	for (Map.Entry e : hm.entrySet()) {
	Integer x = (Integer)e.getValue();
	if (x > 1) {
		int temp5 = factorial(x);
		f = f / temp5;
	}
	}
	return f;
}

static void nextPermutation(char[] temp) {

	// Start traversing from the end and
	// find position 'i-1' of the first character
	// which is greater than its successor.
	int i;
	for (i = temp.length - 1; i > 0; i--)
	if (temp[i] > temp[i - 1])
		break;

	// Finding smallest character after 'i-1' and
	// greater than temp[i-1]
	int min = i;
	int j, x = temp[i - 1];
	for (j = i + 1; j < temp.length; j++)
	if ((temp[j] < temp[min]) && (temp[j] > x))
		min = j;

	// Swapping the above found characters.
	char temp_to_swap;
	temp_to_swap = temp[i - 1];
	temp[i - 1] = temp[min];
	temp[min] = temp_to_swap;

	// Sort all digits from position next to 'i-1'
	// to end of the string.
	Arrays.sort(temp, i, temp.length);

	// Print the String
	print(temp);
}

static void printAllPermutations(String s) {

	// Sorting String
	char temp[] = s.toCharArray();
	Arrays.sort(temp);

	// Print first permutation
	print(temp);

	// Finding the total permutations
	int total = calculateTotal(temp, temp.length);
	for (int i = 1; i < total; i++)
	nextPermutation(temp);
}

// Driver Code
public static void main(String[] args) {
	String s = "AAB";
	printAllPermutations(s);
}
}






24)Manevuring a cave prob:
import java.util.*;
class Main {
    static int numberOfPaths(int m, int n){
        if (m == 1 || n == 1)
            return 1;
    return numberOfPaths(m-1,n)+numberOfPaths(m,n-1);
}

public static void main(String args[])
{
System.out.println(numberOfPaths(2, 2));
}
}




25)Josephus Trap:
import java.util.Scanner;

class Main{

    static int josephusSolve(int n,int k){
        if(n==1){
            return 1;
        }
        return (josephusSolve(n-1,k)+ k-1)% n+1;
    }

    public static void main(String args[]){
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int k = sc.nextInt();
        System.out.println(josephusSolve(n,k));
    }
}











26)Rat in a maze:
class RatInMaze {

    static boolean isSafe(int maze[][], int x, int y, int n) {
        return (x < n && x >= 0 && y < n && y >= 0 && maze[x][y] == 1);
    }

    static boolean solveMazeUtil(int maze[][], int x, int y, int sol[][], int n) {
        if (x == n - 1 && y == n - 1) {
            sol[x][y] = 1;
            return true;
        }
        if (isSafe(maze, x, y, n) == true) {
            sol[x][y] = 1;
            if (solveMazeUtil(maze, x + 1, y, sol, n)) {
                return true;
            }
            if (solveMazeUtil(maze, x, y + 1, sol, n)) {
                return true;
            }
            sol[x][y] = 0;
            return false;
        }
        return false;
    }

    static void solveMaze(int maze[][], int n) {
        int sol[][] = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                sol[i][j] = 0;
            }
        }

        if (solveMazeUtil(maze, 0, 0, sol, n) == false) {
            System.out.println("No Solution");
            return;
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
                System.out.print(" " + sol[i][j] + " ");
            System.out.println();
        }
    }

    public static void main(String arg[]) {
        int maze[][] = {
            { 1, 0, 0, 0 },
            { 1, 1, 0, 1 },
            { 0, 1, 0, 0 },
            { 1, 1, 1, 1 },
        };

        solveMaze(maze, 4);

    }
}









27)Nqueens:
import java.util.Scanner;

class NQueen {

    static void soln(int board[][], int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                System.out.print(board[i][j] + " ");
            }
            System.out.println();
        }
    }

    static boolean isSafe(int board[][], int row, int column, int n) {
        int i, j;
        for (i = 0; i < column; i++) {
            if (board[row][i] == 1) {
                return false;
            }
        }
        for (i = row, j = column; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] == 1) {
                return false;
            }
        }
        for (i = row, j = column; i < n && j >= 0; i++, j--) {
            if (board[i][j] == 1) {
                return false;
            }
        }
        return true;
    }

    static void solveNQueen(int board[][], int column, int n) {
        if (column >= n) {
            soln(board, n);
            return;
        }
        for (int i = 0; i < n; i++) {
            if (isSafe(board, i, column, n)) {
                board[i][column] = 1;
                solveNQueen(board, column + 1, n);
                board[i][column] = 0;
            }
        }
    }

    public static void main(String ar[]) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter the value of n: ");
        int n = sc.nextInt();
        int board[][] = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                board[i][j] = 0;
            }
        }

        solveNQueen(board, 0, n);
    }
}










28)Hamiltonian Cycle:
import java.util.*;
class Main
{
	final int V = 5;
	int path[];

	/* A utility function to check if the vertex v can be
	added at index 'pos'in the Hamiltonian Cycle
	constructed so far (stored in 'path[]') */
	boolean isSafe(int v, int graph[][], int path[], int pos)
	{
		/* Check if this vertex is an adjacent vertex of
		the previously added vertex. */
		if (graph[path[pos - 1]][v] == 0)
			return false;

		/* Check if the vertex has already been included.
		This step can be optimized by creating an array
		of size V */
		for (int i = 0; i < pos; i++)
			if (path[i] == v)
				return false;

		return true;
	}

	/* A recursive utility function to solve hamiltonian
	cycle problem */
	boolean hamCycleUtil(int graph[][], int path[], int pos)
	{
		/* base case: If all vertices are included in
		Hamiltonian Cycle */
		if (pos == V)
		{
			// And if there is an edge from the last included
			// vertex to the first vertex
			if (graph[path[pos - 1]][path[0]] == 1)
				return true;
			else
				return false;
		}

		// Try different vertices as a next candidate in
		// Hamiltonian Cycle. We don't try for 0 as we
		// included 0 as starting point in hamCycle()
		for (int v = 1; v < V; v++)
		{
			/* Check if this vertex can be added to Hamiltonian
			Cycle */
			if (isSafe(v, graph, path, pos))
			{
				path[pos] = v;

				/* recur to construct rest of the path */
				if (hamCycleUtil(graph, path, pos + 1) == true)
					return true;

				/* If adding vertex v doesn't lead to a solution,
				then remove it */
				path[pos] = -1;
			}
		}

		/* If no vertex can be added to Hamiltonian Cycle
		constructed so far, then return false */
		return false;
	}

	/* This function solves the Hamiltonian Cycle problem using
	Backtracking. It mainly uses hamCycleUtil() to solve the
	problem. It returns false if there is no Hamiltonian Cycle
	possible, otherwise return true and prints the path.
	Please note that there may be more than one solutions,
	this function prints one of the feasible solutions. */
	int hamCycle(int graph[][])
	{
		path = new int[V];
		for (int i = 0; i < V; i++)
			path[i] = -1;

		/* Let us put vertex 0 as the first vertex in the path.
		If there is a Hamiltonian Cycle, then the path can be
		started from any point of the cycle as the graph is
		undirected */
		path[0] = 0;
		if (hamCycleUtil(graph, path, 1) == false)
		{
			System.out.println("\nSolution does not exist");
			return 0;
		}

		printSolution(path);
		return 1;
	}

	/* A utility function to print solution */
	void printSolution(int path[])
	{
		System.out.println("Solution Exists: Following" +
						" is one Hamiltonian Cycle");
		for (int i = 0; i < V; i++)
			System.out.print(" " + path[i] + " ");

		// Let us print the first vertex again to show the
		// complete cycle
		System.out.println(" " + path[0] + " ");
	}

	// driver program to test above function
	public static void main(String args[])
	{
		HamiltonianCycle hamiltonian =
								new HamiltonianCycle();
		/* Let us create the following graph
		(0)--(1)--(2)
			| / \ |
			| / \ |
			| /	 \ |
		(3)-------(4) */
		int graph1[][] = {{0, 1, 0, 1, 0},
			{1, 0, 1, 1, 1},
			{0, 1, 0, 0, 1},
			{1, 1, 0, 0, 1},
			{0, 1, 1, 1, 0},
		};

		// Print the solution
		hamiltonian.hamCycle(graph1);

		/* Let us create the following graph
		(0)--(1)--(2)
			| / \ |
			| / \ |
			| /	 \ |
		(3)	 (4) */
		int graph2[][] = {{0, 1, 0, 1, 0},
			{1, 0, 1, 1, 1},
			{0, 1, 0, 0, 1},
			{1, 1, 0, 0, 0},
			{0, 1, 1, 0, 0},
		};

		// Print the solution
		hamiltonian.hamCycle(graph2);
	}
}













29)Combination:
import java.util.*;
public class Main{
    static int fact(int number){
        int f=1;
        int j=1;
        while(j<=number){
            f=f*j;
            j++;
        }
    return f;
}
public static void main(String args[]) {

    List<Integer> numbers=new ArrayList<Integer>();
	numbers.add(9);
    numbers.add(12);
    numbers.add(19);
    numbers.add(61);
    numbers.add(19);
    int n=numbers.size();
    int r=2;
    int result;
    result=fact(n)/(fact(r)*fact(n-r));
    System.out.println("The combination value for the numbers list is: "+result);
    }
}













30)Weighted Substring:
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main {
    static int[] values = new int[26];

    public static void main(String[] args) 
    {
        insertValues();
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        List<Character> s = new ArrayList<>();
        formedString(s, n);
    }

    static void insertValues() 
    {
        values[0] = 1;
        int prev = 1;
        for (int i = 1; i < 26; i++) 
        {
            values[i] = (i + 1) * prev + prev;
            prev = values[i];
        }
    }

    static void formedString(List<Character> s, int k) 
    {
        int low = 0;
        int high = 25;
        while (k != 0) 
        {
            int ind = findFloor(k, low, high);
            s.add((char) (ind + 'A'));
            k = k - values[ind];
        }
        for (int i = s.size() - 1; i >= 0; i--)
            System.out.print(s.get(i));
    }

    static int findFloor(int k, int low, int high) 
    {
        int ans = -1;
        while (low <= high) 
        {
            int mid = (low + high) / 2;
            if (values[mid] <= k) 
            {
                ans = mid;
                low = mid + 1;
            }
            else {
                high = mid - 1;
            }
        }
        return ans;
    }
}
















31)Warnsdoff Algo:
public class KnightTour 
{  
    private static final int[] ROW_MOVES = {2, 1, -1, -2, -2, -1, 1, 2};  
    private static final int[] COL_MOVES = {1, 2, 2, 1, -1, -2, -2, -1};  
  
    public static boolean isSafe(int[][] board, int row, int col, int N) 
    {  
        return (row >= 0 && row < N && col >= 0 && col < N && board[row][col] == -1);  
    }  
 
    public static void printSolution(int[][] board) 
    {  
        int N = board.length;  
        for (int[] row : board) 
        {  
            for (int cell : row) 
            {  
                System.out.print(cell + " ");  
            }  
            System.out.println();  
        }  
        System.out.println();  
    }  
  
    public static boolean solveKnightTour(int N) 
    {  
        int[][] board = new int[N][N];  
  
        // Initialize the board with -1 (unvisited)  
        for (int i = 0; i< N; i++) 
        {  
            for (int j = 0; j < N; j++) 
            {  
                board[i][j] = -1;  
            }  
        }  
  
        // Start the knight's tour from the top-left corner (0, 0)  
        board[0][0] = 0;  
  
        // Recursive function to find a solution  
        if (solveUtil(board, 0, 0, 1, N)) 
        {  
            printSolution(board);  
            return true;  
        } 
        else
        {  
            System.out.println("No solution exists.");  
            return false;  
        }  
    }  
  
    public static boolean solveUtil(int[][] board, int row, int col, int moveCount, int N) 
    {  
        if (moveCount == N * N) 
        {  
            return true;  
        }  
  
        // Try all next moves from the current position  
        for (int i = 0; i< 8; i++) 
        {  
            int nextRow = row + ROW_MOVES[i];  
            int nextCol = col + COL_MOVES[i];  
  
            if (isSafe(board, nextRow, nextCol, N)) {  
                board[nextRow][nextCol] = moveCount;  
  
                // Recur for the next move  
                if (solveUtil(board, nextRow, nextCol, moveCount + 1, N)) 
                {  
                    return true;  
                }  
  
                // Backtrack: if the move doesn't lead to a solution, undo it  
                board[nextRow][nextCol] = -1;  
            }  
        }  
  
        // If none of the moves work, the problem has no solution  
        return false;  
    }  
  
    public static void main(String[] args) 
    {  
        int N = 8;  // Board size  
        solveKnightTour(N);  
    }  
}