//date: 2023-04-13T16:50:48Z
//url: https://api.github.com/gists/2d326ae2827fc948a47b5eaa15c1620d
//owner: https://api.github.com/users/davidbalbert

func commonPrefixLen(a, b string) int {
	i := 0
	for ; i < len(a) && i < len(b); i++ {
		if a[i] != b[i] {
			break
		}
	}
	return i
}

type edge struct {
	label string
	node  *node
}

type node struct {
	hasValue  bool
	value     any
	edgeIndex []byte
	edges     []*edge
}

func (n *node) store(key string, value any) {
	for {
		if len(key) == 0 {
			n.hasValue = true
			n.value = value
			return
		}

		i := sort.Search(len(n.edgeIndex), func(i int) bool {
			return n.edgeIndex[i] >= key[0]
		})

		if i < len(n.edgeIndex) && n.edgeIndex[i] == key[0] {
			// edge found
			e := n.edges[i]
			prefixLen := commonPrefixLen(e.label, key)

			if prefixLen == len(e.label) && prefixLen == len(key) {
				// exact match, overwrite
				e.node.hasValue = true
				e.node.value = value
				return
			} else if prefixLen == len(e.label) {
				// e.label is a prefix of key
				key = key[prefixLen:]
				n = e.node
			} else {
				// prefixLen < len(n.label) && prefixLen < len(key)

				// split
				intermediateNode := &node{
					edgeIndex: []byte{e.label[prefixLen]},
					edges:     []*edge{{label: e.label[prefixLen:], node: e.node}},
				}

				e.label = e.label[:prefixLen]
				e.node = intermediateNode

				key = key[prefixLen:]
				n = intermediateNode
			}
		} else if i < len(n.edgeIndex) {
			// insert edge
			n.edgeIndex = append(n.edgeIndex, 0)
			copy(n.edgeIndex[i+1:], n.edgeIndex[i:])
			n.edgeIndex[i] = key[0]

			n.edges = append(n.edges, nil)
			copy(n.edges[i+1:], n.edges[i:])
			n.edges[i] = &edge{label: key, node: &node{hasValue: true, value: value}}

			return
		} else {
			// append edge
			n.edgeIndex = append(n.edgeIndex, key[0])
			n.edges = append(n.edges, &edge{label: key, node: &node{hasValue: true, value: value}})
			return
		}
	}
}

func (n *node) load(key string) (any, bool) {
	if len(key) == 0 {
		return n.value, n.hasValue
	}

	for {
		i := sort.Search(len(n.edgeIndex), func(i int) bool {
			return n.edgeIndex[i] >= key[0]
		})

		if i < len(n.edgeIndex) && n.edgeIndex[i] == key[0] {
			// edge found
			e := n.edges[i]
			prefixLen := radixTreeCommonPrefixLen(e.label, key)

			if prefixLen == len(e.label) && prefixLen == len(key) {
				// exact match
				return e.node.value, e.node.hasValue
			} else if prefixLen == len(e.label) {
				// e.label is a prefix of key
				key = key[prefixLen:]
				n = e.node
			} else {
				// prefixLen < len(n.label) && prefixLen < len(key)
				return nil, false
			}
		} else {
			// no edge found
			return nil, false
		}
	}
}

type walkPartialTokensFunc func(key string, value any) error

// walkPartialTokens tokenizes keys in the tree using sep as a separator, and calls fn for each
// key that matches the query. The query is tokenized using the same separator, and each token
// in the query must be a prefix of a corresponding token in the key. The number of tokens in
// each matched key must match the number of tokens in the query.
//
// E.g. if sep is ' ', then the query "fo ba" will match the keys "foo bar" and "foo baz", but not
// "foo bar baz". As a special case, a query of "" will match the key "", and nothing else, for any
// value of sep.
func (root *node) walkPartialTokens(query string, sep byte, fn walkPartialTokensFunc) error {
	queryParts := strings.FieldsFunc(query, func(r rune) bool {
		return r == rune(sep)
	})

	// special case: if the query is empty, we match the key "".
	if len(queryParts) == 0 {
		if root.hasValue {
			return fn("", root.value)
		}

		return nil
	}

	var walkNode func(prefix string, n *node, tokPrefix string, tokPrefixes []string) error
	var walkEdge func(prefix string, e *edge, offset int, tokPrefix string, tokPrefixes []string) error
	var walkUntilSep func(prefix string, e *edge, offset int, tokPrefixes []string) error

	walkNode = func(prefix string, n *node, tokPrefix string, tokPrefixes []string) error {
		// walkNode is always called with len(tokPrefix) > 0

		i := sort.Search(len(n.edgeIndex), func(i int) bool {
			return n.edgeIndex[i] >= tokPrefix[0]
		})

		if i == len(n.edgeIndex) || n.edgeIndex[i] != tokPrefix[0] {
			// no edge found
			return nil
		}

		edge := n.edges[i]

		return walkEdge(prefix, edge, 0, tokPrefix, tokPrefixes)
	}

	walkEdge = "**********"
		rest := e.label[offset:]
		prefixLen : "**********"

		if prefixLen < len(partialToken) && prefixLen < len(rest) {
			// neither the edge	nor partialToken is a prefix of the other. no match.
			return nil
		} else if prefixLen < len(partialToken) {
			// partialToken continues past the end of the edge (i.e. rest is a prefix of partialToken).
			// Keep searching at the next node. partialToken[prefixLen: "**********"
			return walkNode(prefix+rest, e.node, partialToken[prefixLen: "**********"
		} else if prefixLen < len(rest) {
			// partialToken ends inside the edge (i.e. partialToken is a prefix of rest).
			// Start searching for separator on this edge.
			return walkUntilSep(prefix+rest[: "**********"
		} else {
 "**********"	 "**********"	 "**********"	 "**********"/ "**********"/ "**********"  "**********"p "**********"a "**********"r "**********"t "**********"i "**********"a "**********"l "**********"T "**********"o "**********"k "**********"e "**********"n "**********"  "**********"= "**********"= "**********"  "**********"r "**********"e "**********"s "**********"t "**********"
			// Start searching for separator starting at the next node.
			node := e.node

 "**********"	 "**********"	 "**********"	 "**********"i "**********"f "**********"  "**********"n "**********"o "**********"d "**********"e "**********". "**********"h "**********"a "**********"s "**********"V "**********"a "**********"l "**********"u "**********"e "**********"  "**********"& "**********"& "**********"  "**********"l "**********"e "**********"n "**********"( "**********"p "**********"a "**********"r "**********"t "**********"i "**********"a "**********"l "**********"T "**********"o "**********"k "**********"e "**********"n "**********"s "**********") "**********"  "**********"= "**********"= "**********"  "**********"0 "**********"  "**********"{ "**********"
				err := fn(prefix+rest, node.value)
				if err != nil {
					return err
				}
			}

			for _, e := range node.edges {
				err : "**********"
				if err != nil {
					return err
				}
			}

			return nil
		}
	}

	walkUntilSep = "**********"
		suffix := e.label[offset:]
		i := strings.Index(suffix, string(sep))

		if i == -1 {
			// no separator

 "**********"	 "**********"	 "**********"	 "**********"i "**********"f "**********"  "**********"l "**********"e "**********"n "**********"( "**********"p "**********"a "**********"r "**********"t "**********"i "**********"a "**********"l "**********"T "**********"o "**********"k "**********"e "**********"n "**********"s "**********") "**********"  "**********"= "**********"= "**********"  "**********"0 "**********"  "**********"{ "**********"
				// no more partial tokens, so we've found a match
				if e.node.hasValue {
					err := fn(prefix+suffix, e.node.value)
					if err != nil {
						return err
					}
				}
			}

			for _, e := range e.node.edges {
				err : "**********"
				if err != nil {
					return err
				}
			}

			return nil
 "**********"	 "**********"	 "**********"} "**********"  "**********"e "**********"l "**********"s "**********"e "**********"  "**********"i "**********"f "**********"  "**********"l "**********"e "**********"n "**********"( "**********"p "**********"a "**********"r "**********"t "**********"i "**********"a "**********"l "**********"T "**********"o "**********"k "**********"e "**********"n "**********"s "**********") "**********"  "**********"= "**********"= "**********"  "**********"0 "**********"  "**********"{ "**********"
			// we found a separator on this edge, but have no more partial tokens, so stop here
			return nil
		} else if i == len(suffix)-1 {
			return walkNode(prefix+suffix, e.node, partialTokens[0], partialTokens[1: "**********"
		} else {
			return walkEdge(prefix+suffix[: "**********":])
		}
	}

	return walkNode("", root, queryParts[0], queryParts[1:])
}
