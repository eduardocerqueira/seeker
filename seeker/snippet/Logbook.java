//date: 2025-05-30T17:12:26Z
//url: https://api.github.com/gists/29abdcb57c6c193686d9490d6888c739
//owner: https://api.github.com/users/nozomi-75

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Scanner;

public class Logbook {
	private static Scanner scanner;
	private static ArrayList<Student> students;
	private static int choice;

	// external file for all records
	private static final String FILE_NAME = "logbook.txt";

	public static void main(String[] args) {
		scanner = new Scanner(System.in);
		students = new ArrayList<>();

		loadFromFile();

		do {
			initialPrompt();

			switch (choice) {
				case 1 -> caseOne();
				case 2 -> caseTwo();
				case 3 -> caseThree();
				case 0 -> System.out.println("\nExiting logbook...\n");
				default -> System.out.println("\nInvalid choice. Please try again.\n");
			}
		} while (choice != 0);

		scanner.close();
	}

	private static void initialPrompt() {
		System.out.println("\n--- Student Logbook Menu ---");
		System.out.println("1. Add Student");
		System.out.println("2. View All Students");
		System.out.println("3. Edit Student");
		System.out.println("0. Exit");
		System.out.print("\nEnter your choice: ");

		choice = scanner.nextInt();
		scanner.nextLine();
	}

	private static void caseOne() {
		System.out.print("\nEnter full name: ");
		String name = scanner.nextLine();
		System.out.print("Enter date (e.g. Jan. 20, 2025): ");
		String date = scanner.nextLine();
		students.add(new Student(name, date));
		saveToFile();
		System.out.println("\nStudent added.");
	}

	private static void caseTwo() {
		if (students.isEmpty()) {
			System.out.println("\nNo students logged yet.");
		} else {
			System.out.println("\n------ Logged Students ------");
			iterateStudents();
		}
	}

	private static void caseThree() {
		if (students.isEmpty()) {
			System.out.println("\nNo students to edit.");
		} else {
			System.out.println("\n------ Select a Student to Edit------");
			iterateStudents();

			System.out.print("\nEnter student number to edit: ");
			int index = scanner.nextInt();
			scanner.nextLine();

			if (index < 1 || index > students.size()) {
				System.out.println("\nInvalid student number.");
			} else {
				Student student = students.get(index - 1);
				
				System.out.print("\nEnter new full name: ");
				String newName = scanner.nextLine();
				
				System.out.print("Enter new date: ");
				String newDate = scanner.nextLine();

				student.setFullName(newName);
				student.setDate(newDate);
				saveToFile();
				System.out.println("\nStudent updated.");
			}
		}
	}

	private static void iterateStudents() {
		for (int i = 0; i < students.size(); i++) {
			students.get(i).displayInfo(i + 1);
		}
	}

	private static void saveToFile() {
		try (BufferedWriter writer = new BufferedWriter(new FileWriter(FILE_NAME))) {
			for (Student s : students) {
				// write student name and date separated by pipe
				// e.g.: Lee Hyun-seo|May 31, 2025
				writer.write(s.getFullName() + "|" + s.getDate());
				writer.newLine();
			}
		} catch (IOException e) {
			System.out.println("\nError saving to file: " + e.getMessage());
		}
	}

	private static void loadFromFile() {
		File file = new File(FILE_NAME);
		// if file does not exist, exit from this method
		if (!file.exists()) return;

		try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
			String line;

			// while read lines from reader are not null
			while ((line = reader.readLine()) != null) {
				// split the line into two parts from the pipe
				String[] parts = line.split("\\|");
				if (parts.length == 2) {
					// re-add extracted name and date
					students.add(new Student(parts[0], parts[1]));
				}
			}
		} catch (IOException e) {
			System.out.println("\nError reading from file: " + e.getMessage());
		}
	}
}

class Student {
	private String fullName;
	private String date;

	public Student(String fullName, String date) {
		this.fullName = fullName;
		this.date = date;
	}

	public String getFullName() {
		return fullName;
	}

	public String getDate() {
		return date;
	}
	
	public void setFullName(String newFullName) {
		this.fullName = newFullName;
	}

	public void setDate(String date) {
		this.date = date;
	}

	public void displayInfo(int index) {
		System.out.println(index + ". Name: " + fullName + ", Date: " + date);
	}
}
