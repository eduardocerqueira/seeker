//date: 2022-05-11T17:15:31Z
//url: https://api.github.com/gists/f2063edc2323e1274f59600f17f782d6
//owner: https://api.github.com/users/MorrisLaw

func maxProfit(prices []int) int {
    var maxP int
    left, right := 0, 1
    
    for right < len(prices) {
        if prices[left] < prices[right] {
            profit := prices[right] - prices[left]
            maxP = max(profit, maxP)
        } else {
            left = right
        }
        right++
    }
    return maxP
}

func max(a, b int) int {
    return int(math.Max(float64(a), float64(b)))
}