//date: 2025-08-18T17:03:34Z
//url: https://api.github.com/gists/c90cdd89fb398fba2debf4f8a7393e42
//owner: https://api.github.com/users/serj09123

package main

import (
    "testing"
)

func TestExactOffsetMatch(t *testing.T) {
    stamps := []ScoreStamp{
        {Offset: 5, Score: Score{Home: 1, Away: 0}},
        {Offset: 10, Score: Score{Home: 2, Away: 0}},
        {Offset: 15, Score: Score{Home: 2, Away: 1}},
    }

    result := getScore(stamps, 10)
    expected := Score{Home: 2, Away: 0}

    if result != expected {
        t.Errorf("Expected %v, got %v", expected, result)
    }
}

func TestOffsetBetweenStamps(t *testing.T) {
    stamps := []ScoreStamp{
        {Offset: 5, Score: Score{Home: 1, Away: 0}},
        {Offset: 10, Score: Score{Home: 2, Away: 0}},
        {Offset: 15, Score: Score{Home: 2, Away: 1}},
    }

    result := getScore(stamps, 12)
    expected := Score{Home: 2, Away: 0}

    if result != expected {
        t.Errorf("Expected %v, got %v", expected, result)
    }
}

func TestOffsetBeforeFirstStamp(t *testing.T) {
    stamps := []ScoreStamp{
        {Offset: 5, Score: Score{Home: 1, Away: 0}},
    }

    result := getScore(stamps, 3)
    expected := Score{Home: 0, Away: 0}

    if result != expected {
        t.Errorf("Expected %v, got %v", expected, result)
    }
}

func TestOffsetAfterLastStamp(t *testing.T) {
    stamps := []ScoreStamp{
        {Offset: 5, Score: Score{Home: 1, Away: 0}},
        {Offset: 10, Score: Score{Home: 2, Away: 1}},
    }

    result := getScore(stamps, 20)
    expected := Score{Home: 2, Away: 1}

    if result != expected {
        t.Errorf("Expected %v, got %v", expected, result)
    }
}

func TestSingleStamp(t *testing.T) {
    stamps := []ScoreStamp{
        {Offset: 7, Score: Score{Home: 1, Away: 1}},
    }

    result := getScore(stamps, 7)
    expected := Score{Home: 1, Away: 1}

    if result != expected {
        t.Errorf("Expected %v, got %v", expected, result)
    }
}

func TestOffsetBetweenSingleStampAndZero(t *testing.T) {
    stamps := []ScoreStamp{
        {Offset: 7, Score: Score{Home: 1, Away: 1}},
    }

    result := getScore(stamps, 3)
    expected := Score{Home: 0, Away: 0}

    if result != expected {
        t.Errorf("Expected %v, got %v", expected, result)
    }
}
