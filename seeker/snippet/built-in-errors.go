//date: 2024-06-12T16:48:02Z
//url: https://api.github.com/gists/6926292395867c6700fd28811baa4aa5
//owner: https://api.github.com/users/docsallover

_, err := os.Open("missing_file.txt")
if err != nil {
  fmt.Println("Error:", err.Error()) // Output: Error: open missing_file.txt: no such file or directory
}