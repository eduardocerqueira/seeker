//date: 2023-12-18T16:50:37Z
//url: https://api.github.com/gists/90cae857a6c7f470000e09b8d5f96a40
//owner: https://api.github.com/users/AndHov09

public static int nonuniformRandomNumberGeneration(List<Integer> values, List<Double> probabilities) 
{
    List<Double> prefixSumOfProbabilities = new ArrayList<>();
    prefixSumOfProbabilities.add(0.0); // Initial value for the prefix sum

    // Creating the endpoints for the intervals corresponding to the probabilities
    for (double p : probabilities) 
    {
        double previousSum = prefixSumOfProbabilities.get(prefixSumOfProbabilities.size() - 1);
        prefixSumOfProbabilities.add(previousSum + p);
    }

    Random random = new Random();
    // Get a random number in [0.0, 1.0)
    final double uniformRandom = random.nextDouble();

    // Find the index of the interval that uniformRandom lies in
    int intervalIndex = Collections.binarySearch(prefixSumOfProbabilities, uniformRandom);

    if (intervalIndex < 0) 
    {
        // Adjust index if the value is not present in the array
        int adjustedIndex = Math.abs(intervalIndex) - 1 - 1;
        return values.get(adjustedIndex);
    } else {
        // Return the value corresponding to the matched interval
        return values.get(intervalIndex);
    }
}
