//date: 2025-08-18T17:01:46Z
//url: https://api.github.com/gists/257218602110c16c83ba232e674e93e2
//owner: https://api.github.com/users/serj09123

func getScore(gameStamps []ScoreStamp, offset int) Score {
    if offset < gameStamps[0].Offset {
        return Score{Home: 0, Away: 0}
    }

    left := 0
    right := len(gameStamps) - 1
    resultIndex := -1

    for left <= right {
        mid := left + (right-left)/2
        if gameStamps[mid].Offset == offset {
            return gameStamps[mid].Score
        } else if gameStamps[mid].Offset < offset {
            resultIndex = mid
            left = mid + 1
        } else {
            right = mid - 1
        }
    }

    return gameStamps[resultIndex].Score
}
