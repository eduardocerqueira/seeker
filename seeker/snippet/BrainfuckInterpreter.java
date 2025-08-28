//date: 2025-08-28T17:04:25Z
//url: https://api.github.com/gists/abb43ce6cdd0320282a3d72314d8c01a
//owner: https://api.github.com/users/DrDrunkenstien-10

import java.util.Scanner;

public class BrainfuckInterpreter {
	private static final int CELL_COUNT = 30000;

	public static void main(String[] args) {
		Scanner scanner = new Scanner(System.in);
		
		System.out.println("Enter the brainfuck code:");

		String brainfuckCode = scanner.nextLine();
		
		int[] cells = new int[CELL_COUNT];
		
		int currentPointer = 0;

		for(int i = 0; i < brainfuckCode.length(); i++) {
			char instruction = brainfuckCode.charAt(i);

			switch(instruction) {
				case '+':
				cells[currentPointer] = (cells[currentPointer] + 1) % 256;
				break;

				case '-':
				cells[currentPointer] = (cells[currentPointer] - 1 + 256) % 256;
				break;

				case '.':
				System.out.print((char) cells[currentPointer]);
				break;
				
				case '>':
				currentPointer++;
				break;

				case '<':
				currentPointer--;
				break;

				default:
				throw new IllegalArgumentException("Invalid instruction: " + instruction);
			}
		}

		scanner.close();
	}
}
