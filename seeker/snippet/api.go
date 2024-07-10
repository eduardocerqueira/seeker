//date: 2024-07-10T16:55:53Z
//url: https://api.github.com/gists/3ee12b530dc49c2dc093fc53e89d4427
//owner: https://api.github.com/users/BK1031

package main

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

var Port = "9000"

func StartServer() {
	r := gin.Default()
	r.GET("/ecu", GetAllECUs)
	r.GET("/battery", GetAllBatteries)
	r.Run(":" + Port)
}

func GetAllECUs(c *gin.Context) {
	var ecus []ECU
	DB.Find(&ecus)
	c.JSON(http.StatusOK, ecus)
}

func GetAllBatteries(c *gin.Context) {
	var batteries []Battery
	DB.Find(&batteries)
	c.JSON(http.StatusOK, batteries)
}