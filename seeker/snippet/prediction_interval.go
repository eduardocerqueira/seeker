//date: 2024-03-07T17:01:47Z
//url: https://api.github.com/gists/c2d534c03d852bd8849c3246383082ab
//owner: https://api.github.com/users/ivan3bx

package main

func predictionInterval() {
	// Given values
	values := []float64{
		21.500000,
		23.250000,
		22.000000,
		23.700000,
		26.000000,
		25.000000,
		20.000000,
		28.200000,

		15.000000,
	}
	n := len(values) // Sample size

	// Calculate sample mean and standard deviation
	mean, stdDev := stat.MeanStdDev(values, nil)

	// Set the desired confidence level (e.g., 95%)
	confidenceLevel := 0.95

	// Calculate the critical value from the t-distribution
	degreesOfFreedom := float64(n - 1)
	tDist := distuv.StudentsT{
		Mu:    0,
		Sigma: 1,
		Nu:    degreesOfFreedom,
	}

	tCritical := tDist.Quantile(1 - (1-confidenceLevel)/2)

	// Calculate the margin of error
	marginOfError := tCritical * stdDev * math.Sqrt(1+1/float64(n))

	// Calculate the prediction interval
	lowerBound := mean - marginOfError
	upperBound := mean + marginOfError

	fmt.Println("Prediction Interval:")
	fmt.Println("Lower Bound:", lowerBound)
	fmt.Println("Upper Bound:", upperBound)

	fmt.Println("Mean: ", mean)
	fmt.Println("Stddev:", stdDev)
	fmt.Println("Margin of error:", marginOfError)

}