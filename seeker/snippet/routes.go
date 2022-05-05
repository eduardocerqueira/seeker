//date: 2022-05-05T17:16:42Z
//url: https://api.github.com/gists/dd200fb6a0f75ffdd3059f7fc7f66cfc
//owner: https://api.github.com/users/jfjensen

package routes

import (
	"github.com/gin-gonic/gin"

	controllers "gin_session_auth/controllers"
)

func PublicRoutes(g *gin.RouterGroup) {

	g.GET("/login", controllers.LoginGetHandler())
	g.POST("/login", controllers.LoginPostHandler())
	g.GET("/logout", controllers.LogoutGetHandler())
	g.GET("/", controllers.IndexGetHandler())

}

func PrivateRoutes(g *gin.RouterGroup) {

	g.GET("/dashboard", controllers.DashboardGetHandler())

}
