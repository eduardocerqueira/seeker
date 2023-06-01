//date: 2023-06-01T16:41:18Z
//url: https://api.github.com/gists/8dc4189ee43a09020b202fcd95017026
//owner: https://api.github.com/users/rkuprov

func NewWaitPool(size int) *WaitPool {
	if size < 1 {
		size = 1
	}

	return &WaitPool{
		queue: make(chan struct{}, size),
	}
}

type WaitPool struct {
	queue chan struct{}
	wg    sync.WaitGroup
}

func (w *WaitPool) Run(ctx context.Context, f func()) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	// if we can't add to the queue, we're full and should wait
	case w.queue <- struct{}{}:
		w.wg.Add(1)
		go func() {
			f()
			w.wg.Done()
			// remove from the queue to open a space for another
			<-w.queue
		}()

		return nil
	}
}

func (w *WaitPool) Wait(ctx context.Context) error {
	wp := make(chan struct{})

	go func() {
		w.wg.Wait()
		close(wp)
	}()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-wp:
		return nil
	}
}