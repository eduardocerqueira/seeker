//date: 2021-12-30T17:12:48Z
//url: https://api.github.com/gists/a271442b769ebd7663624c86365d2ea4
//owner: https://api.github.com/users/ShubhraSuman

public BubbleSortAlgorithm implements SortAlgorithm{
  public int[] sort(int[] numbers){
    int i, j;
    int n = numbers.size();
    for (i = 0; i < n-1; i++)    
    // Last i elements are already in place
    for (j = 0; j < n-i-1; j++)
        if (numbers[j] > numbers[j+1])
            swap(&numbers[j], &numbers[j+1]);
    return numbers;
  }
}