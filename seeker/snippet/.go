//date: 2024-08-22T17:06:39Z
//url: https://api.github.com/gists/adeca6f4a4529ea196d5c8b0003220ed
//owner: https://api.github.com/users/vasiliyantufev

package main

import (
	"fmt"
	"math"
	//"strings"
)

type node struct {
	val int
	lv  *node
	rv  *node
}

func buildTree(depth int, value int, num int) *node {
	if depth == 0 || value > num {
		return nil
	}

	fmt.Print(depth)
	fmt.Print("|")
	fmt.Print(value)
	fmt.Print("|")

	node := node{val: value}
	node.lv = buildTree(depth-1, value*2, num)
	node.rv = buildTree(depth-1, value*2+1, num)

	return &node
}

func main() {  
	numVertices := 10
	depth := int(math.Log2(float64(numVertices)) + 1) 
	
	//changeVertice := "5 7 4 7 8 7"
	//changeCount := 6
	
	root := buildTree(depth, 1, numVertices)

	fmt.Println("Pre-order traversal:")
	preOrderTraversal(root)

}

func preOrderTraversal(root *node) {
	if root == nil {
		return
	}
	fmt.Println(root.val)
	preOrderTraversal(root.lv)
	preOrderTraversal(root.rv)
}
