//date: 2025-06-17T16:56:39Z
//url: https://api.github.com/gists/774ff158e363771ece706dd0125d5f64
//owner: https://api.github.com/users/blinkinglight

router.Get("/stream/{id}", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		id := chi.URLParam(r, "id")
		sse := datastar.NewSSE(w, r)
		_ = sse

		ctx := bee.WithJetStream(r.Context(), js)
		ctx = bee.WithNats(ctx, nc)

		agg := &Aggregate{}
		updates := bee.ReplayAndSubscribe(ctx, users.Aggregate, id, agg)
		go func() {
			for {
				select {
				case <-r.Context().Done():
					return
				case update := <-updates:
					sse.MergeFragmentTempl(partials.History(update.History))
				}
			}
		}()
		<-r.Context().Done()
	})