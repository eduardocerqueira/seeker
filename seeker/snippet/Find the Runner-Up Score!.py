#date: 2021-09-20T17:05:35Z
#url: https://api.github.com/gists/5b8c66070d17d4226d6dddf975d4fecd
#owner: https://api.github.com/users/pankajk1997

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    
    winner = arr[0]
    runner = arr[0]
    
    for i in arr:
        if i>winner:
            runner=winner
            winner=i
        elif i<winner:
            if winner == runner:
                runner = i
            elif i > runner:
                runner = i
    
    print(runner)
