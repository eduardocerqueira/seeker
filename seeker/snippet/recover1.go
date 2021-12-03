//date: 2021-12-03T17:16:02Z
//url: https://api.github.com/gists/0cb4b26ec24301f75c86e219775cb5e9
//owner: https://api.github.com/users/tirmizee

const limit = 10

func process(i int) (int, error) {

	if i < 0 {
		return i, errors.New("number must not be less than zero")
	}

	i++
	if i >= limit {
		return i, errors.New("number out of bound")
	}
	return i, nil
}

func main() {

	defer func() {
		if err := recover(); err != nil {
			fmt.Println("panic occurred:", err)
		}
	}()

	fmt.Println("process init")

	result1, err := process(10)
	if err != nil {
		panic(err.Error())
	}

	fmt.Println("process result", result1)

}