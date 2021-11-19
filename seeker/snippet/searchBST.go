//date: 2021-11-19T16:55:29Z
//url: https://api.github.com/gists/089f70d66139bbbfa0c31a3c711da435
//owner: https://api.github.com/users/muhfaa

package main

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func searchBST(root *TreeNode, val int) bool {
	if root == nil {
		return false
	}
	if val < root.Val {
		return searchBST(root.Left, val)
	}
	if val > root.Val {
		return searchBST(root.Right, val)
	}
	return true
}