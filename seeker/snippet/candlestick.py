#date: 2022-05-16T17:14:53Z
#url: https://api.github.com/gists/3064f73fd2d23b69ea1c3153048e26d9
#owner: https://api.github.com/users/theDestI

def plot_candlestick(prices):
    """
    Plots the candlestick of a pricing data.
    Credits: https://www.statology.org/matplotlib-python-candlestick-chart/ 
    """
    #create figure
    plt.figure()

    #define width of candlestick elements
    width = .4
    width2 = .05

    #define up and down prices
    up = prices[prices.Close>=prices.Open]
    down = prices[prices.Close<prices.Open]

    #plot up prices
    plt.bar(up.index,up.Close-up.Open,width,bottom=up.Open,color='green')
    plt.bar(up.index,up.High-up.Close,width2,bottom=up.Close,color='green')
    plt.bar(up.index,up.Low-up.Open,width2,bottom=up.Open,color='green')

    #plot down prices
    plt.bar(down.index,down.Close-down.Open,width,bottom=down.Open,color='red')
    plt.bar(down.index,down.High-down.Open,width2,bottom=down.Open,color='red')
    plt.bar(down.index,down.Low-down.Close,width2,bottom=down.Close,color='red')

    #rotate x-axis tick labels
    plt.xticks(rotation=45, ha='right')
    plt.show()