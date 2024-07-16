//date: 2024-07-16T16:45:33Z
//url: https://api.github.com/gists/3ee51664f79b21e06fec83818856da26
//owner: https://api.github.com/users/BK1031

package api

import (
	"bookstore/config"
	"net/http"

	"github.com/gin-gonic/gin"
)

func StartServer() {
	r := gin.Default()
	RegisterRoutes(r)
	r.Run(":" + config.Port)
}

func RegisterRoutes(r *gin.Engine) {
	r.GET("/ping", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"message": "pong",
		})
	})
}
