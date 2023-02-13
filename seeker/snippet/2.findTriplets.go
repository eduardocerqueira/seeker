//date: 2023-02-13T17:06:45Z
//url: https://api.github.com/gists/d83deb7b3527b6008e415f2db4daea33
//owner: https://api.github.com/users/mokhlesurr031

package main

import "fmt"

func main() {
	numArr := []int{-1, 0, 1, 2, -1, -4}

	numLen := len(numArr)

	resMap := make(map[[3]int]bool)


	intArr := []int{}

	for i := 0; i < numLen; i++ {
		for j := 1; j < numLen; j++ {
			for k := 2; k < numLen; k++ {
				if numArr[i]+numArr[j]+numArr[k] == 0 {
					intArr = append(intArr, numArr[i])
					intArr = append(intArr, numArr[j])
					intArr = append(intArr, numArr[k])
					resMap[intArr]=true
					
				}
			}
		}
	}

}

// [[-1,-1,2],[-1,0,1]]
