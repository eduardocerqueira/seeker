//date: 2024-03-07T17:00:30Z
//url: https://api.github.com/gists/7e87584f36bd1730d39243adf270ad81
//owner: https://api.github.com/users/ivan3bx

package main

import (
	"fmt"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distuv"
	"slices"

	"math"
	"sort"
)

func main() {
	oatmealCookieData := []Record{
		Record{Distance: 0.221699, CaloriesMin: 105.000000},
		Record{Distance: 0.246969, CaloriesMin: 151.000000},
		Record{Distance: 0.257615, CaloriesMin: 129.000000},
		Record{Distance: 0.265802, CaloriesMin: 220.000000},
		//		Record{Distance: 0.273367, CaloriesMin: 96.000000},
		Record{Distance: 0.278475, CaloriesMin: 166.000000},
		Record{Distance: 0.281798, CaloriesMin: 102.000000},
		Record{Distance: 0.286099, CaloriesMin: 149.000000},
		Record{Distance: 0.298152, CaloriesMin: 131.706000},
		Record{Distance: 0.300421, CaloriesMin: 229.000000},
		Record{Distance: 0.302333, CaloriesMin: 108.000000},
		//		Record{Distance: 0.306329, CaloriesMin: 232.000000},
		Record{Distance: 0.311352, CaloriesMin: 126.924000},
		Record{Distance: 0.316033, CaloriesMin: 137.778000},
		Record{Distance: 0.318632, CaloriesMin: 139.287000},
		Record{Distance: 0.319882, CaloriesMin: 105.000000},
		Record{Distance: 0.338914, CaloriesMin: 139.287000},
		Record{Distance: 0.352538, CaloriesMin: 126.300000},
		Record{Distance: 0.352664, CaloriesMin: 139.287000},
		Record{Distance: 0.355051, CaloriesMin: 144.828000},
	}

	lasagnaData := []Record{
		Record{Distance: 0.308568, CaloriesMin: 361.770000},
		Record{Distance: 0.317278, CaloriesMin: 352.110000},
		Record{Distance: 0.340925, CaloriesMin: 394.740000},
		//		Record{Distance: 0.345523, CaloriesMin: 164.070000},
		Record{Distance: 0.361125, CaloriesMin: 438.000000},
		Record{Distance: 0.364716, CaloriesMin: 3346.000000},
		Record{Distance: 0.365076, CaloriesMin: 357.000000},
		Record{Distance: 0.368206, CaloriesMin: 187.000000},
		Record{Distance: 0.368369, CaloriesMin: 231.000000},
		Record{Distance: 0.375945, CaloriesMin: 371.430000},
		Record{Distance: 0.380823, CaloriesMin: 450.000000},
		Record{Distance: 0.381618, CaloriesMin: 1108.920000},
		Record{Distance: 0.382454, CaloriesMin: 399.000000},
		Record{Distance: 0.388987, CaloriesMin: 976.000000},
		Record{Distance: 0.397948, CaloriesMin: 820.000000},
		Record{Distance: 0.403988, CaloriesMin: 540.000000},
		Record{Distance: 0.404686, CaloriesMin: 641.000000},
		//		Record{Distance: 0.409581, CaloriesMin: 1942.000000},
		Record{Distance: 0.410584, CaloriesMin: 975.000000},
		Record{Distance: 0.414556, CaloriesMin: 286.000000},
	}

	fmt.Println("LASAGNA")
	generateBins(lasagnaData)
	calculateHist(lasagnaData)

	fmt.Println("OATMEAL COOKIE")
	generateBins(oatmealCookieData)
	calculateHist(oatmealCookieData)
}

type Record struct {
	Distance    float64
	CaloriesMin float64
}

// simple binning, creating equally sized bins of distance 0.1
func calculateHist(data []Record) {
	var distances, calories []float64

	slices.SortFunc(data, func(a, b Record) int {
		if a.Distance < b.Distance {
			return -1
		} else {
			return 1
		}
	})

	for _, r := range data {
		distances = append(distances, r.Distance)
		calories = append(calories, r.CaloriesMin)
	}

	dst := make([]float64, 6)
	floats.Span(dst, 0.0, 0.5)

	counts := stat.Histogram(nil, dst, distances, nil)
	values := stat.Histogram(nil, dst, distances, calories)

	for idx, count := range counts {
		if count == 0 {
			continue
		}
		values[idx] = values[idx] / count
	}

	fmt.Printf("Values: %+v\n", values)

	weightedAverage := weightedAverage(values)
	fmt.Println("Weighted Average: ", weightedAverage)
}

func generateBins(records []Record) {
	var distances []float64
	for _, r := range records {
		distances = append(distances, r.Distance)
	}

	width := calculateScaledFreedmanDiaconisBinWidth(distances)
	fmt.Println("Width:", width)

	start := 0.0
	for start <= distances[len(distances)-1] {
		fmt.Println("Bucket: ", start, start+width)
		start = start + width
	}
}

// calculate # bins with sturges' rule (log2(n) + 1)
func calculateSturgesRule(data []float64) float64 {
	// Determine the number of data points
	n := float64(len(data))

	// Calculate the number of bins using Sturges' Rule
	numBins := math.Ceil(math.Log2(n) + 1)

	sort.Float64s(data)
	smallest := data[0]
	largest := data[len(data)-1]

	fmt.Println("BINS: ", numBins, " / Smallest:", smallest, " / Biggest:", largest)
	return largest / numBins
}

// Function to calculate Scott's rule bin width based on varianace (std dev across the data set)
func calculateScottsRule(data []float64) float64 {
	// Calculate the standard deviation of the data
	stdDev := standardDeviation(data)

	// Determine the number of data points
	n := float64(len(data))

	// Calculate the bin width using Scott's rule
	binWidth := 3.5 * stdDev / math.Pow(n, 1.0/3.0)
	return binWidth
}

// Function to calculate the standard deviation of a dataset
func standardDeviation(data []float64) float64 {
	// Calculate the mean of the data
	mean := mean(data)

	// Calculate the sum of squares of differences from the mean
	sumSqDiff := 0.0
	for _, value := range data {
		diff := value - mean
		sumSqDiff += diff * diff
	}

	// Calculate the variance
	variance := sumSqDiff / float64(len(data))

	// Calculate the standard deviation
	stdDev := math.Sqrt(variance)

	return stdDev
}

// Function to calculate the mean of a dataset
func mean(data []float64) float64 {
	sum := 0.0
	for _, value := range data {
		sum += value
	}
	return sum / float64(len(data))
}

// Freedman-Diaconis rule looks at the interquartile range of data between 0.25 and 0.75 percentile)
// https://en.wikipedia.org/wiki/Freedman–Diaconis_rule
func calculateScaledFreedmanDiaconisBinWidth(data []float64) float64 {
	// applying a scaling factor to reduce number of bins overall (-ivan)
	const scaleFactor = 2.0
	sort.Float64s(data)

	// Step 2: Calculate the interquartile range (IQR)
	q1 := percentile(data, 0.25)
	q3 := percentile(data, 0.75)
	iqr := q3 - q1

	// Step 3: Determine the number of data points
	n := float64(len(data))

	// Step 4: Calculate the bin width using the Freedman-Diaconis rule
	binWidth := scaleFactor * 2.0 * iqr * math.Pow(n, (-1.0/3.0))

	fmt.Printf("Interquartile range (IQR): %.6f\n", iqr)
	//	fmt.Printf("Number of data points: %d\n", int(n))
	fmt.Printf("Bin width (Freedman-Diaconis rule): %.6f\n", binWidth)
	return binWidth
}

// Function to calculate the percentile of a sorted dataset
func percentile(data []float64, percentile float64) float64 {
	k := (float64(len(data)) - 1) * percentile
	f := int(k)
	c := k - float64(f)

	if f+1 < len(data) {
		return data[f] + c*(data[f+1]-data[f])
	}
	return data[f]
}

func weightedAverage(values []float64) float64 {
	var totalWeight float64
	n := len(values)

	//	// Calculate the total weight
	//	for i := 0; i < n; i++ {
	//		totalWeight += i
	//	}

	weightedSum := 0.0
	for i := 0; i < n; i++ {
		weight := math.Pow(float64(n-i), 2)
		if values[i] > 0 {
			fmt.Println("Adding value: ", values[i], " weight: ", weight)
			weightedSum += values[i] * weight
			totalWeight += weight
		}
	}
	fmt.Println("Total weight:", totalWeight)
	return weightedSum / float64(totalWeight)
}
