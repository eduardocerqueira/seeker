//date: 2022-01-03T16:57:49Z
//url: https://api.github.com/gists/a13bef6d9f496e700f227f13e62334d5
//owner: https://api.github.com/users/peytonh03

public class InvalidCardValueException extends Exception
{
	private int valueIdentifier = 0;

	public InvalidCardValueException(int invalidValue) {

		valueIdentifier = invalidValue;

		System.out.println("Invalid value " + invalidValue);
	}

	private InvalidCardValueException() {


		System.out.println("Invalid value");
	}

	public String toString() {
		

		return ("Attempted to create card with invalid suit argument" + " " + this.valueIdentifier);
	}

	public int getValue() {
		
		return valueIdentifier;
	}

} //End class