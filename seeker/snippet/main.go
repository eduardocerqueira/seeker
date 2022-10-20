//date: 2022-10-20T17:25:09Z
//url: https://api.github.com/gists/073df3b1d94cef5f5c0d3b806666e3a0
//owner: https://api.github.com/users/carlosljr

package main

import (
	"github.com/PicPay/poc-api/book"
	"github.com/PicPay/poc-api/book/postgresql"
	ginHandle "github.com/PicPay/poc-api/internal/http/gin"
  	"github.com/gin-gonic/gin"
)

func main() {
	s := book.NewService(postgresql.New())
  	r := gin.Default()
	ginHandle.Handlers(s, r)
	
	if err := r.Run(); err != nil {
		l.Fatal("error running api", err)
	}
}