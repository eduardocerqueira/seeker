//date: 2024-09-19T16:41:24Z
//url: https://api.github.com/gists/5598eded0409e47acaf5cdcf7855f591
//owner: https://api.github.com/users/loderunner

package main

import (
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

type DataPoint struct {
	Timestamp   time.Time `json:"timestamp"`
	Temperature float32   `json:"temperature"`
}

var upgrader websocket.Upgrader

func data(ctx *gin.Context) {
	c, err := upgrader.Upgrade(ctx.Writer, ctx.Request, nil)
	if err != nil {
		ctx.AbortWithError(http.StatusInternalServerError, err)
		return
	}
	defer c.Close()

	var dataPoint DataPoint
	for err := c.ReadJSON(&dataPoint); err == nil; err = c.ReadJSON(&dataPoint) {
		fmt.Printf(
			"[%s] %.2fºC\n",
			dataPoint.Timestamp.Format(time.RFC3339),
			dataPoint.Temperature,
		)
	}

	if err != nil {
		ctx.AbortWithError(http.StatusInternalServerError, err)
		return
	}

	ctx.Status(http.StatusOK)
}

func main() {
	r := gin.Default()

	r.GET("/data", data)

	err := r.Run(":8888")
	if !errors.Is(err, http.ErrServerClosed) {
		panic(err.Error())
	}
}
