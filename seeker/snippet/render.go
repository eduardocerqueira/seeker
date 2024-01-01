//date: 2024-01-01T16:55:14Z
//url: https://api.github.com/gists/4cb0ee8ade7f73e545d36124c580dcb4
//owner: https://api.github.com/users/mhrlife

func Render(c echo.Context, comp templ.Component) error {
	c.Response().Header().Set(echo.HeaderContentType, echo.MIMETextHTML)
	return comp.Render(c.Request().Context(), c.Response().Writer)
}