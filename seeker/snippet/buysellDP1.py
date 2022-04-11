#date: 2022-04-11T17:16:48Z
#url: https://api.github.com/gists/1c3d0c58a65a4971fbeff42693b64ce2
#owner: https://api.github.com/users/PrasadGanesh

# --------------- Brute force approach ---------
def BuySellStockBrute(prices):
    '''
                This function calculates the maximum benefit after buying and selling
                a stock.
                Input: it is the list of prices of a stock
                Return: The maximum profit possible
                Time Comlexity: O(n*n)... Read code for why!
    '''

    # get the size of the prices list
    n = len(prices)
    # declare a variable to hold the maximum profit
    maximum = 0

    # traverse every possible pair
    for buyOn in range(n):  # <---- Complexity
        for sellOn in range(buyOn+1, n):  # <--- Complexity
            # calculate profit
            profit = prices[sellOn] - prices[buyOn]
            if(maximum < profit):
                maximum = profit

    # return the maximum profit
    return maximum


# -------------- Greedy approach --------------

def BuySellStockGreedy(prices):
    '''
        this function calculates the maximum profit on buying a stock
        and selling it in future.
        input: a list of prices
        output: maximum profit
        Time Complexity: O(n)... Read code for why!
    '''
    # variable to hold the maximum profit
    maximum_profit = 0
    # variable to hold the minimum price seen so far
    minimum_price = prices[0]

    # traverse the prices list to find the maximum profit possible
    for price in prices:
        # this is the greedy part, choose the minimum seen so far
        # as well as look for maximum profit
        # if the current price is minimum seen so far
        if minimum_price > price:
            # update price
            minimum_price = price
        elif (price - minimum_price) > maximum_profit:
            maximum_profit = price - minimum_price

    # return the maximum profit seen so far
    return maximum_profit


# -------------- Driver program -----------------


def main():
    # define a prices
    prices = [7, 1, 5, 3, 6, 4]
    print(BuySellStockBrute(prices))
    print(BuySellStockGreedy(prices))

    prices = [7, 6, 4, 3, 1]
    print(BuySellStockBrute(prices))
    print(BuySellStockGreedy(prices))


if __name__ == '__main__':
    main()
