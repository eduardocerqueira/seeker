//date: 2023-03-27T16:48:01Z
//url: https://api.github.com/gists/0170a77a31fb47ba50a3383afbc05e98
//owner: https://api.github.com/users/nouhoum

r := chi.NewRouter()
// Declare your routes

// Show all routes declared to go chi (in gin style). Very useful in dev
chi.Walk(r, func(method string, route string, handler http.Handler, middlewares ...func(http.Handler) http.Handler) error {
    sep := strings.Repeat(" ", 10-len(method))
    cleanRoute, _ := strings.CutSuffix(route, "/")
    fmt.Printf("%s%s%s\n", method, sep, cleanRoute)
    return nil
})