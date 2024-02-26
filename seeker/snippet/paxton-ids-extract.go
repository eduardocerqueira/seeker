//date: 2024-02-26T16:54:30Z
//url: https://api.github.com/gists/de911705e943de5687f02e77c95e150e
//owner: https://api.github.com/users/spbkaizo

package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
)

// padStringToLength takes a string and pads it with leading zeros until it reaches a total length of 10.
func padStringToLength(s string) string {
	for len(s) < 10 {
		s = "0" + s
	}
	return s
}

// formatHexWithColons takes a hexadecimal string and formats it so that every two characters are separated by a colon.
func formatHexWithColons(hexStr string) string {
	var sb strings.Builder // Using strings.Builder for efficient string concatenation

	// Iterate over the hex string in steps of 2 to process each pair of characters
	for i := 0; i < len(hexStr); i += 2 {
		sb.WriteString(hexStr[i : i+2]) // Append the pair of characters
		if i < len(hexStr)-2 {
			sb.WriteString(":") // Append a colon after each pair except the last one
		}
	}

	return sb.String()
}

// writeToFile writes the given string to a file, appending a newline after the string.
func writeToFile(filename, text string) error {
	// Open the file in append mode. If it doesn't exist, create it with permissions 0666.
	file, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0666)
	if err != nil {
		return err
	}
	defer file.Close()

	// Write the text followed by a newline
	_, err = file.WriteString(text + "\n")
	return err
}

func main() {
	// Define a command-line flag for the CSV file path
	csvFilePath := flag.String("file", "", "Path to the CSV file")
	flag.Parse()

	// Check if the file path was provided
	if *csvFilePath == "" {
		fmt.Println("CSV file path must be provided")
		os.Exit(1) // Exit if no file path is provided
	}

	// Open the CSV file
	file, err := os.Open(*csvFilePath)
	if err != nil {
		fmt.Printf("Error opening CSV file: %v\n", err)
		os.Exit(1) // Exit if the file cannot be opened
	}
	defer file.Close()

	// Create a new CSV reader from the file
	r := csv.NewReader(file)

	// Reading the header line to identify the "Card Number" column index
	headers, err := r.Read()
	if err != nil {
		fmt.Printf("Error reading CSV: %v\n", err)
		return
	}

	cardNumberIndex := -1
	for i, header := range headers {
		if header == "Card Number" {
			cardNumberIndex = i
			break
		}
	}

	if cardNumberIndex == -1 {
		fmt.Println("Column 'Card Number' not found")
		return
	}

	// Reading the rest of the rows and extracting the "Card Number" column
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			fmt.Printf("Error reading CSV row: %v\n", err)
			return
		}

		cardNumber := record[cardNumberIndex]
		// Skip if card number is unset, doesn't start with '6', or doesn't contain '#'
		if cardNumber == "" || !strings.HasPrefix(cardNumber, "6") || !strings.Contains(cardNumber, "#") {
			continue
		}

		// Extract the number after '#' and convert it to decimal
		parts := strings.Split(cardNumber, "#")
		if len(parts) < 2 {
			continue // Skip if there's no number after '#'
		}
		decimalNumber, err := strconv.Atoi(parts[1])
		if err != nil {
			fmt.Printf("Error converting number to integer: %v\n", err)
			continue
		}

		// Convert the decimal number to hexadecimal and print
		hexNumber := strconv.FormatInt(int64(decimalNumber), 16)
		paddedHexNumber := padStringToLength(hexNumber)
		formattedHexNumber := formatHexWithColons(paddedHexNumber)
		log.Printf("card id: %v hex: %v padded: %v formatted: %v", decimalNumber, hexNumber, paddedHexNumber, formattedHexNumber)
		// Write the formatted hex number to "targets.txt"
		err = writeToFile("targets.txt", paddedHexNumber)
		if err != nil {
			fmt.Printf("Error writing to file: %v\n", err)
		}

	}
	fmt.Println("id's written to targets.txt")
}
