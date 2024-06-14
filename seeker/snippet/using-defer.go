//date: 2024-06-14T16:50:14Z
//url: https://api.github.com/gists/ab1ddc41a6430d90c48ae635b066d1d5
//owner: https://api.github.com/users/docsallover

file, err := os.Open("data.txt")
defer file.Close()  // Close the file even if there's an error

if err != nil {
  fmt.Println("Error opening file:", err)
  // Handle the error but the file will still be closed
}

// Use data from the file