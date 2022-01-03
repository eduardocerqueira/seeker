//date: 2022-01-03T16:57:49Z
//url: https://api.github.com/gists/a13bef6d9f496e700f227f13e62334d5
//owner: https://api.github.com/users/peytonh03

import java.io.Serializable;

public class Hand implements Serializable
{

	private Card[] theHand = new Card[12];

	private int numberOfCards = 0;

	// Calculates the total of a hand and also decides whether ace is 1 or 11
	public int calculateTotal() {
		int total =0;
		boolean aceFlag = false;
		for (int i = 0; i < numberOfCards; i++) {
			int value = theHand[i].getValue();
			if (value > 10) {
				value = 10;
			} else if ( value == 1) {
				aceFlag = true;
			}
			total += value;
		}
		if (aceFlag && total + 10 <= 21) {
			total += 10;
		}
		return total;
	}
	
	public String toString(){
		return this.toString(false, false);
	}
	
	public String toString(boolean isDealer, boolean hideHoleCard){
		String str = "";
		int total =0;
		boolean aceFlag = false;
		String aceString = "";
		for (int i = 0; i < numberOfCards; i++) {
			if ( isDealer && hideHoleCard && i == 0) {
				str = " Showing";
			} else {
				int value = theHand[i].getValue();
				String valueName;
				if (value > 10) {
					valueName = theHand[i].getValueName().substring(0, 1);
				} else if ( value == 1 ){
					valueName = "A";
				} else {
					valueName = Integer.toString(value);
				}
						str += " " +valueName + theHand[i].getSuitDesignator();
				if (value > 10) {
					value = 10;
				} else if ( value == 1) {
					aceFlag = true;
				}
				total += value;
			}
		}
		if (aceFlag && total + 10 <= 21) {
			aceString = " or "+ (total + 10);
		}
		if ( hideHoleCard) {
			return str;
		} else {
			return str+ " totals "+ total + aceString;
		}
		
	}
	
	public void addCard(Card card) {
		theHand[numberOfCards++] = card;
	}
	
	public void clearHand() {
		numberOfCards = 0;
	}
	
	public boolean dealerPeek() {
		int value = theHand[1].getValue();
		return value == 1 || value >= 10;
	}
} //End class