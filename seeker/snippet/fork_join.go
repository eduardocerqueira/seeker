//date: 2023-06-20T17:08:09Z
//url: https://api.github.com/gists/d7b918128f08420178133280a80b1af3
//owner: https://api.github.com/users/Luiz-Monad


func DoWorkPooled[T any](items []T, poolSize int, task func(T) (*T, error)) ([]T, error) {
	if poolSize == 0 { // 0 means sequential
		output := []T{}
		for _, item := range items {
			if out, err := task(item); err != nil {
				return output, err
			} else {
				if out != nil {
					output = append(output, *out)
				}
			}
		}
		return output, nil
	}

	numOfItems := len(items)
	input := make(chan T, numOfItems)
	output := make(chan T, numOfItems)
	errout := make(chan error, 1)
	var wg sync.WaitGroup
	wg.Add(numOfItems)
	for _, item := range items {
		input <- item
	}
	close(input)

	ctx, cancel := context.WithCancel(context.Background())

	for i := 0; i < poolSize; i++ {
		go func() {
			for {
				select {
				case item, ok := <-input:
					if !ok {
						// Input is empty
						return
					}
					if out, err := task(item); err != nil {
						errout <- err
						cancel()
					} else {
						if out != nil {
							output <- *out
						}
						wg.Done()
					}
				case <-ctx.Done():
					// Context cancelled, exit early
					return
				}
			}
		}()
	}

	wg.Wait()
	cancel()
	close(output)
	close(errout)

	itemsout := []T{}
	for out := range output {
		itemsout = append(itemsout, out)
	}

	err, haserr := <-errout
	if haserr {
		return itemsout, err
	}
	return itemsout, nil
}
