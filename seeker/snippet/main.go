//date: 2022-05-02T17:18:14Z
//url: https://api.github.com/gists/db6b1a65d9ff220cbb276a5220236ebe
//owner: https://api.github.com/users/Maixy

package main

import (
 ...
)

var logger *zap.SugaredLogger

func init() {
	//always initialize random uniquely at startup
	rand.Seed(time.Now().UnixNano())
	logger = logging.GetLogger()
}

func main() {
	initGinRouter()
}

var tracer = otel.Tracer("main-server")

func initGinRouter() {
	//Set to ReleaseMode unless debug flag has been enabled
	if !config.GetBool("debug") {
		gin.SetMode(gin.ReleaseMode)
	}

	r := gin.Default()

	//Add logging middleware
	r.Use(logging.GetRequestLogger())
	r.Use(logging.GetRecoveryLogger())

	if tools.OtelConfigured() {
		tools.InitOtelTracer()
		tools.AddOtelMiddlware(r)
	}

	r.GET("/otelx", func(c *gin.Context) {
		_, span := tracer.Start(c.Request.Context(), "otelx", oteltrace.WithAttributes(
			attribute.String("someKey", "someVal")))
		defer span.End()
		
		c.String(200, "result data")
	})

	port := config.GetInt("server.port")
	portString := fmt.Sprintf(`:%d`, port)
	logger.Debugf(`Running on %s`, portString)
	r.Run(portString)
}
