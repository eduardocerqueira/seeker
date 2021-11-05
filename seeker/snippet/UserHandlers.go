//date: 2021-11-05T17:07:29Z
//url: https://api.github.com/gists/1bb3ec0c62154df99348752ea00b8adf
//owner: https://api.github.com/users/jainam240101

package users

import (
	"github.com/gin-gonic/gin"
	domain "github.com/jainam240101/todo/server/domain/Users"
	service "github.com/jainam240101/todo/server/services"
)

type Userhandlers struct {
	Service service.UserService
}

func (uh *Userhandlers) CreateUser(c *gin.Context) {
	u := domain.UserModel{}
	if err := c.BindJSON(&u); err != nil {
		c.JSON(406, gin.H{
			"message": "Body Not Proper",
		})
		return
	}
	user, err := uh.Service.CreateUser(u)
	if err != nil {
		c.JSON(406, gin.H{
			"message": "Body Not Proper",
		})
		return
	}
	c.JSON(200, gin.H{
		"data": user,
	})
}

func (uh *Userhandlers) FindUserById(c *gin.Context) {
	username := c.Request.URL.Query().Get("username")
	if username == "" {
		c.JSON(406, gin.H{
			"message": "Query Not Proper",
		})
	}
	user, err := uh.Service.FindUserByUsername(username)
	if err != nil {
		c.JSON(406, gin.H{
			"message": "Something went wrong",
		})
		return
	}
	c.JSON(200, gin.H{
		"data": user,
	})
}

func (uh *Userhandlers) UpdateUser(c *gin.Context) {
	userId := "57dffe4c-3205-4201-86b3-544331323fd3" // For Now we are hardcoding it but we will change this field when we introducr Middleware
	u := domain.UserModel{}
	if err := c.BindJSON(&u); err != nil {
		c.JSON(406, gin.H{
			"message": "Body has Parameters missing",
		})
		return
	}
	data, err := uh.Service.UpdateUser(u, userId)
	if err != nil {
		c.JSON(406, gin.H{
			"message": "Something went wrong",
		})
		return
	}
	c.JSON(200, gin.H{
		"data": data,
	})
}

func (uh *Userhandlers) DeleteUser(c *gin.Context) {
	userId := "57dffe4c-3205-4201-86b3-544331323fd3" // For Now we are hardcoding it but we will change this field when we introducr Middleware

	deleteErr := uh.Service.DeleteUser(userId)
	if deleteErr != nil {
		c.JSON(406, gin.H{
			"message": "Something went wrong",
		})
		return
	}
	c.JSON(200, gin.H{
		"data": "User Deleted",
	})
}
