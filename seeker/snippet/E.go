//date: 2024-05-16T17:00:06Z
//url: https://api.github.com/gists/14db7c6c48e361ae43dea7c719ef06fb
//owner: https://api.github.com/users/schk

package main

import (
	"strconv"
	"strings"
	"os"
	"fmt"
	"bufio"
)

type TrieNode struct {
	letters []*TrieNode
	parentNode *TrieNode
	indexOfMaxWord int
	isEnd bool
}

// root node
func NewTrieNode(parent *TrieNode, index, weight int) *TrieNode {
	return &TrieNode{
		make([]*TrieNode, 26),
		parent,
		index,
		false,
	}
}

func (t *TrieNode) Insert(word string, index, weight int) {
	node := t	
	if weight > weights[node.indexOfMaxWord] {
		node.indexOfMaxWord = index
	}
	for _, c := range word {
		idx := c - 'a'
		// print("IDX", idx, "C: ", c, "\n")
		if node.letters[idx] == nil {
			node.letters[idx] = NewTrieNode(node, index, weight)
		} else if weight > weights[node.letters[idx].indexOfMaxWord] {
			node.letters[idx].indexOfMaxWord = index
		}
		node = node.letters[idx]
	}
	node.isEnd = true
}

func (t *TrieNode) ParentNode(node *TrieNode) *TrieNode {
	return node.parentNode
}

func (t *TrieNode) ChildNode(node *TrieNode, ch string) *TrieNode {
	c := []byte(ch)[0]
	idx := c - 'a'
	child := node.letters[idx]

	return child 
}

type Record struct {
	Index int
	Weight int
}

var weights map[int]int
var words map[string]Record

func main() {
	scanner := bufio.NewScanner(os.Stdin)
	// n, q
	scanner.Scan()
	s := scanner.Text()
	nq := strings.Split(s, " ")
	n, _ := strconv.Atoi(nq[0])
	q, _ := strconv.Atoi(nq[1])
	// fmt.Println("n and q", n, q)

	// cnt - number of letters after correct prefix
	// length of the incorrect part
	cnt := 0
	weights = make(map[int]int)
	words = make(map[string]Record)
	dummy := &TrieNode{}
	trie := NewTrieNode(dummy, 0, -1)
	for i := 1; i <= n; i++ {
		scanner.Scan()
		s = scanner.Text()
		wordweight := strings.Split(s, " ")
		word := wordweight[0]
		weight, _ := strconv.Atoi(wordweight[1])
		weights[i] = weight
		words[word] = Record{i, weight}
		// fmt.Printf("inserting: %s\n", word)
		trie.Insert(word, i, weight)
	}
	// fmt.Println(words)
	// fmt.Println(trie.indexOfMaxWord)
	for i := 0; i < q; i++ {
		scanner.Scan()
		s = scanner.Text()
		if len(s) == 1 {
			if cnt > 0 {
				if cnt == 1 {
					fmt.Println(trie.indexOfMaxWord)
				} else {
					fmt.Println(-1)
				}
				cnt--
			} else {
				trie = trie.ParentNode(trie)
				fmt.Println(trie.indexOfMaxWord)
			}
		} else {
			if cnt == 0 {
				signch := strings.Split(s, " ")
				temp := trie.ChildNode(trie, signch[1])
				// if we don't have such prefix
				if temp == nil {
					cnt++
					fmt.Println(-1)
				} else {
					trie = temp
					fmt.Println(trie.indexOfMaxWord)
				}
			} else {
				cnt++
				fmt.Println(-1)
			}
		}
		// fmt.Println(trie.indexOfMaxWord)
		// fmt.Println(prefix)
	}
	/////////
	for scanner.Scan() {
		fmt.Println(scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintln(os.Stderr, "reading standard input:", err)
	}
}
