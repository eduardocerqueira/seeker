//date: 2021-12-31T17:06:51Z
//url: https://api.github.com/gists/c74642f110afbeab09450bdad48ef01d
//owner: https://api.github.com/users/sharathbj

package main

import (
	"fmt"
	"math"
)

/*
	gym = [1,2]
	school = [0,2,3,4]
	store = [4]

	[
		1   4,
        1 1 2 3    3
		2,
		2 1  1,
		3 2
	]

*/
func main() {
	//clemen keerti mock interview question
	blocks := []map[string]bool{
		{"gym": false, "school": true, "store": false},
		{"gym": true, "school": false, "store": false},
		{"gym": true, "school": true, "store": false},
		{"gym": false, "school": true, "store": false},
		{"gym": false, "school": true, "store": true},
	}
	fmt.Println(" outputIndex expected 3 got:", findShortestDur(blocks, []string{"gym", "school", "store"}))
	fmt.Println(" outputIndex expected 4 got:", findShortestDur(blocks, []string{"school", "store"}))
	fmt.Println(" outputIndex expected 3 got:", findShortestDur(blocks, []string{"gym", "store"}))
	fmt.Println(" outputIndex expected 2 got:", findShortestDur(blocks, []string{"gym", "school"}))
}

func findShortestDur(blocks []map[string]bool, reqs []string) (outputIndex int) {

	//storing each building location, so that overall search of building when building is false
	dpMap := make(map[string][]int, 0)
	for blockIndex, blocks := range blocks {
		for block, exist := range blocks {
			if exist {
				dpMap[block] = append(dpMap[block], blockIndex)
			}
		}
	}
	//fmt.Println("dpMap", dpMap)

	blocksDistanceCountList := make([]int, len(blocks))
	for blockIndex, block := range blocks {
		//for building, exist := range block {
		for _, building := range reqs {
			exist := block[building]
			finalMinDistance := math.MaxInt32
			if !exist {
				for _, availableBlock := range dpMap[building] {
					distance := blockIndex - availableBlock
					if distance < 0 {
						distance *= -1
					}
					// out of all blocks consider the one which is near
					if distance < finalMinDistance {
						finalMinDistance = distance
					}
				}
				//consider the finalMinDistance which is far, bcz that we have to avoid
				if finalMinDistance > blocksDistanceCountList[blockIndex] {
					blocksDistanceCountList[blockIndex] = finalMinDistance
				}
			}
		}
	}
	//fmt.Print("blocksDistanceCountList", blocksDistanceCountList)
	shortStep := blocksDistanceCountList[0]
	for index, o := range blocksDistanceCountList[1:] {
		if o < shortStep {
			shortStep = o
			outputIndex = index + 1 // since i am starting array frm index 1
		}
	}
	return outputIndex
}
