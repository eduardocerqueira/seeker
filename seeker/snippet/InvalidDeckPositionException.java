//date: 2022-01-03T16:57:49Z
//url: https://api.github.com/gists/a13bef6d9f496e700f227f13e62334d5
//owner: https://api.github.com/users/peytonh03

public class InvalidDeckPositionException extends Exception {

	private int positionIdentifier = 0;

	public InvalidDeckPositionException(int inValidPosition) {

		positionIdentifier = inValidPosition;

		System.out.println("Invalid Position" + inValidPosition);

	}

	private InvalidDeckPositionException() {
		System.out.println("Invalid Position");
	}

	public String toString() {

		return ("Attempted to get a card from a position not in Deck" + " " + this.positionIdentifier);
	}

	public int getPositionValue() {
		return positionIdentifier;
	}
} //End class