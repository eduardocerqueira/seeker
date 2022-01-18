//date: 2022-01-18T17:19:49Z
//url: https://api.github.com/gists/c532dd285d3f985fd3027dbbcfff7d56
//owner: https://api.github.com/users/fulviodenza

func main() {

	// Setup The AWS Region and AWS session
	conf := &aws.Config{Region: aws.String("eu-west-1")}
	mySession := session.Must(session.NewSession(conf))

	// Fill App structure with environment keys and session generated
	a := App{
		CognitoClient:   cognito.New(mySession),
		UserPoolID:      os.Getenv("COGNITO_USER_POOL_ID"),
		AppClientID:     os.Getenv("COGNITO_APP_CLIENT_ID"),
		AppClientSecret: os.Getenv("COGNITO_APP_CLIENT_SECRET"),
	}

	// Echo stuff
	e := echo.New()
	validate := &CustomValidator{validator: validator.New()}
	validate.validator.RegisterStructValidation(validateUser, User{})

	e.Validator = validate

	e.Use(middleware.Logger())
	e.Use(middleware.Recover())
	e.Use(middleware.CORSWithConfig(middleware.CORSConfig{
		AllowOrigins: []string{"*"},
		AllowHeaders: []string{echo.HeaderOrigin, echo.HeaderContentType, echo.HeaderAccept},
	}))

	registerFunc := func(c echo.Context) error {
		return a.Register(c, *validate.validator)
	}
	e.POST("/auth/register", registerFunc)
	e.POST("/auth/login", a.Login)
	e.POST("/auth/otp", a.OTP)
	e.GET("/auth/forgot", a.ForgotPassword)

	confirmForgotPasswordFunc := func(c echo.Context) error {
		return a.ConfirmForgotPassword(c, *validate.validator)
	}
	e.POST("/auth/confirmforgot", confirmForgotPasswordFunc)
	e.Logger.Fatal(e.Start(":1323"))
}