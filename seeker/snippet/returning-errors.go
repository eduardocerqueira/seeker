//date: 2024-06-14T16:48:09Z
//url: https://api.github.com/gists/edea01f8738a895ebd5bcfb56de9288a
//owner: https://api.github.com/users/docsallover

func ReadData(filename string) ([]byte, error) {
  data, err := os.ReadFile(filename)
  if err != nil {
    return nil, err // Return the error to the caller
  }
  return data, nil
}

func main() {
  data, err := ReadData("data.txt")
  if err != nil {
    fmt.Println("Error reading data:", err)
    // Handle the error (e.g., exit or use default data)
  } else {
    fmt.Println(string(data))
  }
}