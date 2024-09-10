//date: 2024-09-10T17:10:57Z
//url: https://api.github.com/gists/1ca01da35b1115bf90f0484c4562d401
//owner: https://api.github.com/users/timmattison

func DumbGenerateRandomNumber(length int) string {
	output := ""

	var channels []chan struct{}
	var results []int
	ctx, cancelFunc := context.WithCancelCause(context.Background())

	for range 10 {
		channels = append(channels, make(chan struct{}, 100))

		for range 100 {
			channels[len(channels)-1] <- struct{}{}
		}
	}

	for range 10 {
		results = append(results, 0)
	}

	outputDigitsChannel := make(chan rune)

	for i := range 10 {
		go func() {
			defer close(channels[i])
			select {
			case <-ctx.Done():
				return
			case channels[i] <- struct{}{}:
			}
		}()
	}

	go func() {
		defer close(outputDigitsChannel)

		for {
			select {
			case <-ctx.Done():
				return
			case <-channels[0]:
				outputDigitsChannel <- '0'
			case <-channels[1]:
				outputDigitsChannel <- '1'
			case <-channels[2]:
				outputDigitsChannel <- '2'
			case <-channels[3]:
				outputDigitsChannel <- '3'
			case <-channels[4]:
				outputDigitsChannel <- '4'
			case <-channels[5]:
				outputDigitsChannel <- '5'
			case <-channels[6]:
				outputDigitsChannel <- '6'
			case <-channels[7]:
				outputDigitsChannel <- '7'
			case <-channels[8]:
				outputDigitsChannel <- '8'
			case <-channels[9]:
				outputDigitsChannel <- '9'
			}
		}
	}()

	for outputDigit := range outputDigitsChannel {
		results[outputDigit-'0']++
		output += string(outputDigit)

		if len(output) == length {
			cancelFunc(nil)
			break
		}
	}

	return output
}