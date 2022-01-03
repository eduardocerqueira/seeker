//date: 2022-01-03T16:57:49Z
//url: https://api.github.com/gists/a13bef6d9f496e700f227f13e62334d5
//owner: https://api.github.com/users/peytonh03

public class InvalidCardSuitException extends Exception {

	private char suitIdentifier = '?';

	public InvalidCardSuitException (char invalidSuit) {

		suitIdentifier = invalidSuit;

		System.out.println("Invalid suit" + " " + invalidSuit);
	}

	private InvalidCardSuitException() {
		System.out.println("Invalid suit");
	}
	
	public String toString(){

		return ("Attempted to create card with invalid suit argument" + " " + this.suitIdentifier);

	}
	
	public char getSuitDesignator() {
		
		return suitIdentifier;
	}
} //End class