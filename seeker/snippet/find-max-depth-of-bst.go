//date: 2022-10-05T17:19:15Z
//url: https://api.github.com/gists/9b2750c6d3bc9e701c605cbd5d2882dd
//owner: https://api.github.com/users/dinukasaminda

package main

import (
	"fmt"
)


type Node struct{
	Value int
	Left *Node
	Right *Node
}
type BinarySearchTree struct{
	Root *Node
}
func main(){
	tree := BinarySearchTree{Root: nil}
	Insert(&tree,46)
	Insert(&tree,20)
	Insert(&tree,9)
	Insert(&tree,22)
	Insert(&tree,50)
	Insert(&tree,47)
	Insert(&tree,60)
	Insert(&tree,49)
	
	PrintTree(&tree)
	// fmt.Println(getMin(&tree))
	// fmt.Println(getMax(&tree))
	fmt.Println("max depth:",maxDepth(tree.Root))
}
func PrintNode(n *Node,depth int){
	fmt.Print("depth:",depth," value:",n.Value," ")
	if n.Left!=nil{
		PrintNode(n.Left,depth +1)
	} 
	if n.Right!=nil{
		PrintNode(n.Right,depth +1)
	}
	fmt.Println()
}
func PrintTree(tree *BinarySearchTree){
	PrintNode(tree.Root,0)
}
func Insert(t *BinarySearchTree,value int){
	new_node := Node{Value: value}
	temp := t.Root
	if temp ==nil{
		t.Root = &new_node
	}else{
		for{
			if temp.Value > value{
				if temp.Left == nil{
					temp.Left=&new_node
					break
				}else{
					temp = temp.Left
				}
			}else {
				if temp.Right ==nil{
					temp.Right = &new_node
					break
				}else{
					temp = temp.Right
				}
			}
			
		}
	}
}
func maxDepth(node *Node )int{
	if node == nil{
		return -1
	}
	leftHeight := maxDepth(node.Left)
	rightHeight := maxDepth(node.Right)

	if leftHeight > rightHeight{
		return leftHeight + 1
	}

	return rightHeight + 1
}
func getMin(t *BinarySearchTree)*Node{
	temp := t.Root
	if temp==nil{
		return nil
	}
	for {
		if temp.Left ==nil{
			return temp
		} 
		temp = temp.Left
			
	}
} 