//date: 2023-05-01T16:59:55Z
//url: https://api.github.com/gists/d41ea57ee9c2b601b3c7321237566049
//owner: https://api.github.com/users/benkyriakou

package main

import ("fmt"; "time")

func saveTime(timestamp int32) {
  fmt.Printf("%+d\n", timestamp)
}

func main() {
  // Go handles the epochalypse problem by making timestamps signed 64-bit ints.
  var epochalypse int64 = time.Date(2038, 1, 19, 3, 14, 7, 0, time.UTC).Unix()

  // This value is eventually passed to a function that does something with
  // timestamps, but requires a 32-bit signed integer.
  saveTime(int32(epochalypse))

  // When the time passes the epochalypse, this overflows.
  saveTime(int32(epochalypse + 1))
}
