//date: 2025-05-06T16:51:02Z
//url: https://api.github.com/gists/b4defc58679f558a8b36288b32b55997
//owner: https://api.github.com/users/mschandr77

package game;

/**
 * Initializes an array of String objects.
 * Draws the ascii ship.
 * TODO: Update the class documentation to briefly describe
 *       the changes in this version.
 * 
 * @author Foothill College, Marvish Chandra
 */
public class GuessOrSink {
	/**
	 * TODO: Provide a brief description of the purpose of the main() method.
	 *
	 * @param args not used
	 */
	public static void main(String[] args) {
		// TODO: Add a welcome message of your own.
		System.out.println("Welcome to Guess or Sink!");
		// TODO: Initializes an array with four different words.
		final String[] wordsToGuess = {"ocean", "submarine", "ship", "torpedo"};
		// TODO: Select an element from the wordsToGuess array and store it in the variable word.
		//       This will be the word that we will be working with and display at the end.
		String word = wordsToGuess[2];
		// Holds a row of the ASCII representation of your ship (including smoke, funnel, hull, deck, etc.)
		// Note that asciiShipFunnel is composed of 3 rows but only represents one variable.
		// And that asciiShipDeck is composed of 2 rows but only represents one variable.
		// TODO: Replace the initials BM with your initials.
		final String asciiShipSmoke = "         O  O  O\n";
		final String asciiShipFunnel = "                 O\n" +
				"                __|__\n" +
				"                || ||_____\n";
		final String asciiShipHull = "                || ||    |\n";
		final String asciiShipDeck = "     --------------------------\n" +
				"      \\   O   O   O   O  MC  /\n";
		final String asciiWater = "^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^\n" +
				"   ^^^   ^^^   ^^^   ^^^   ^";

		// TODO: Define a String array object called asciiShip to hold the ascii representation of the ship.
		String[] asciiShip;
		// TODO: Initialize the variable size to hold an index for each letter in the word.
		int[] size = new int[word.length()];
		for (int i = 0; i < word.length(); i++) {
			size[i] = i;
		}

		System.out.println("Selected word: " + word);
		System.out.println("Word length: " + word.length());
		System.out.println("Size array initialized with indices: ");
		for (int i = 0; i < size.length; i++) {
			System.out.print(size[i] + " ");
		}
		System.out.println("\n");

		// TODO: Define the variable rowToDraw to be an int set to the value 0.
		// This will represent the index that our loop structure starts drawing at. 
		int rowToDraw = 0;
		// TODO: Initialize asciiShip with our final String ascii variables given in the starter code above.
		//       After each index that you update, increment rowToDraw	.
		// Note: Each index in the array will hold one row of the ship.
		//       The first two indices will hold the ship smoke and ship funnel.
		//       The last index will hold the ship deck. Use the discussion to discuss how to calculate this!
		//       The rest of the indices will hold the ship hull. See below on using a for-loop to 
		//       initialize this.
		asciiShip[rowToDraw] = asciiShipSmoke;
		rowToDraw++;

		asciiShip[rowToDraw] = asciiShipFunnel;
		rowToDraw++;

		asciiShip[rowToDraw] = asciiShipDeck;
		rowToDraw++;
		for (int i = 0; i < heightOfShipHull; i++) {
			asciiShip[rowToDraw++] = "   \\_____/";
		}

		// TODO: Define the height of the ship hull to be the length of the word minus three.
		int heightOfShipHull = word.length() - 3;
	}

	// TODO: Initialize the remaining indices of the array of String objects for the ascii ship.
	//       How will you calculate the indices (i.e. rows) that will hold the hull?
	//       Use the discussion forum to experiment and discuss.
	// TODO: Check your work. Can you display the ship using one word? then another?
	// TODO: Use a for loop to display the ship. Then display the water.
	public static void displayShip(String shipWord) {
		int shipHullHeight = shipWord.length() - 3;
		System.out.println("Building a ship for the word: " + shipWord);
		System.out.println("Ship hull height: " + shipHullHeight);

		// Simulate ship structure
		System.out.println("    ~~~~~~");
		System.out.println("   |  []  |"); // Smoke stack
		System.out.println("   |______|"); // Funnel
		for (int i = 0; i < shipHullHeight; i++) {
			System.out.println("   |      |"); // Hull lines
		}
		System.out.println("   |______|"); // Ship deck
		System.out.println(" ~~~~~~~~~~~~~~"); // Water
		// TODO: At the end display the word.
		System.out.println("The word for this ship was: " + shipWord);
	}
}
