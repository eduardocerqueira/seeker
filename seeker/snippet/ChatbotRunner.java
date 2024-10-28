//date: 2024-10-28T16:48:53Z
//url: https://api.github.com/gists/56a86c170c17a8d8f273c6718e2118cb
//owner: https://api.github.com/users/MDavisEA

import java.util.Scanner;

/**
 * A simple class to run the Chatbot class.
 * @author Laurie White
 * @version April 2012
 */
public class ChatbotRunner
{

	/**
	 * Create a Chatbot, give it user input, and print its replies.
	 */
	public static void main(String[] args)
	{
		Chatbot cb = new Chatbot();
		
		System.out.println (cb.getGreeting());
		Scanner in = new Scanner (System.in);
    
    
    // Oh no, this code is bummy, fix it so it only needs one in.nextLine()
		String statement = in.nextLine();
		
		while (!statement.equals("Bye"))
		{
			System.out.println (cb.getResponse(statement));
			statement = in.nextLine();
		}
	}

}