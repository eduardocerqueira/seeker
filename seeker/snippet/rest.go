//date: 2022-01-07T17:03:22Z
//url: https://api.github.com/gists/b79d31d15d36c9bcd57b5620b904d0d2
//owner: https://api.github.com/users/peterbradford

package main

import (
	"fmt"
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
	"net/http"
)

var db map[string]string

func runServer(addr string) {
	srv := &http.Server{Addr: addr}

	srv.Handler = handlers()

	go func() {
		var err error
		err = srv.ListenAndServe()

		if err != nil && err.Error() != "http: Server closed" {
			log.Err(err).Msgf("server failed")
		}
	}()

	//shutdown stuff can go here

	log.Info().Msgf("api running on: %v", addr)

}

func handlers() http.Handler {
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(cors.Default(), gin.Recovery(), gin.Logger()) //rscors.AllowAll()

	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, "ok")
	})

	r.POST("", new)
	r.GET("", get)
	r.PUT("", update)
	r.DELETE("", del)

	//someGroup := r.Group("pathName")
	//someGroup.POST("", new)
	//someGroup.GET("", get)
	//someGroup.PUT("", update)
	//someGroup.DELETE("", delete)

	return r
}

func new(c *gin.Context) {
	//if err := json.NewDecoder(c.Request.Body).Decode(someStruct); err != nil {
	//	log.Err(err).Msg("received bad payload")
	//	c.JSON(http.StatusBadRequest, ErrorMessage{ErrorMessages: []string{"unable to parse body into struct", err.Error()}})
	//	return
	//}

	// ^ is how you'd read in the req body to be used for the new db obj

	if _, ok := db["new"]; ok {
		c.JSON(http.StatusBadRequest, fmt.Errorf("'new' key already exists"))
	} else {
		db["new"] = "new"
		c.JSON(http.StatusOK, db)
	}
}

func get(c *gin.Context) {
	c.JSON(http.StatusOK, db["new"])
}

func update(c *gin.Context) {
	db["new"] = "even newer"
	c.JSON(http.StatusOK, db)
}

func del(c *gin.Context) {
	delete(db, "new")
	c.JSON(http.StatusOK, db)
}
