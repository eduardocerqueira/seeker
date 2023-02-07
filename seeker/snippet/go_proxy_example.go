//date: 2023-02-07T16:49:12Z
//url: https://api.github.com/gists/219e0f0dc8409270dc97055f7343fc04
//owner: https://api.github.com/users/LtHummus

// the "real" frontend gets embedded here

//go:embed frontend/build
var frontend embed.FS


func setupReverseProxyForTest(target string) *httputil.ReverseProxy {
	testServer, err := url.Parse(target)
	if err != nil {
		log.Fatal().Str("backend_target", target).Err(err).Msg("could not parse target URL")
	}

	return httputil.NewSingleHostReverseProxy(testServer)
}
  
func main() {
  
	/* bunch of setup removed */
  
	var noRouteHandler http.Handler
	if debugFrontendTarget := os.Getenv("DEBUG_FRONTEND_TARGET"); debugFrontendTarget != "" {
		// we're in dev mode, just proxy everything we don't have to our react server
		noRouteHandler = setupReverseProxyForTest(debugFrontendTarget)
		log.Warn().Str("frontend_target", debugFrontendTarget).Msg("building reverse proxy to react dev server")
	} else {
		// we're in prod mode, use the embedded frontend
		realFrontend, err := fs.Sub(frontend, "frontend/build")
		if err != nil {
			log.Fatal().Err(err).Msg("could not load frontend")
		}
		noRouteHandler = http.FileServer(http.FS(realFrontend))
	}

	r.NoRoute(gin.WrapH(noRouteHandler))
 
	/* more stuff eliminated here */
}