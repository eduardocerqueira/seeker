//date: 2023-02-27T16:46:44Z
//url: https://api.github.com/gists/792b77389287f34a501e572b6c7281df
//owner: https://api.github.com/users/kyleconroy

package main

import (
	"log"

	pg_query_v4 "github.com/pganalyze/pg_query_go/v4"
)

func main() {
	{
		_, err := pg_query_v4.ParseToJSON("SELECT * FROM users WHERE id = ?")
		log.Printf("v4: errored err: %s\n", err)
		// 2023/02/27 08:45:37 v4: errored err: syntax error at end of input
	}
}
