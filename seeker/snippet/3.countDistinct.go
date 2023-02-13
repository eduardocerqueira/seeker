//date: 2023-02-13T17:06:45Z
//url: https://api.github.com/gists/d83deb7b3527b6008e415f2db4daea33
//owner: https://api.github.com/users/mokhlesurr031

package main

import "fmt"

func main() {
	arr := []int{2, 2, 2}

	reversedArr := []int{}
	for _, val := range arr {
		reversedInt := 0
		for val != 0 {
			reversedInt *= 10
			reversedInt += val % 10
			val /= 10
		}
		reversedArr = append(reversedArr, reversedInt)
	}

	combinedArr := append(arr, reversedArr...)

	numMap := map[int]int{}

	for _, val := range combinedArr {
		numMap[val]++
	}

	fmt.Println("Distinct Num: ", len(numMap))

}
