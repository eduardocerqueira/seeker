#date: 2022-01-31T17:06:20Z
#url: https://api.github.com/gists/0f6ac24907bc246eebba3b0f700fcc34
#owner: https://api.github.com/users/benjaminhorvath

def uniquePaths(m, n):
    '''
        Starting from top left corner of matrix of m*n need to get to bottom roght corner 
        by taking 1 step at the time either to the right or to the bottom. Return how many unique steps are there.
    '''
    uniquePathsFromPositionCache = {(m,n):1}
    #question is how many paths are there from (1,1), remember not to count from (0,0)
    res = _getUniquePathsFromPoint(1, 1, uniquePathsFromPositionCache, m, n)
    return res

def _getUniquePathsFromPoint(M, N, cache, m, n):
    '''returns number of unique paths from point M,N in a matrix of size m,n'''
    if (M,N) in cache:
        return cache[(M,N)]
    if M > m or N > n:
        return 0
    maxUniquePaths = _getUniquePathsFromPoint(M+1, N, cache, m, n) + _getUniquePathsFromPoint(M, N+1, cache, m, n)
    cache[(M,N)] = maxUniquePaths
    return maxUniquePaths

if __name__ == '__main__':
    print(uniquePaths(3,7))