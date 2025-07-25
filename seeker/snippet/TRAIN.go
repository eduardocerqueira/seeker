//date: 2025-07-25T17:11:57Z
//url: https://api.github.com/gists/5af4cbd0f0151c40bd724ca63c91d81f
//owner: https://api.github.com/users/ProtMello

package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"net/url"
	"sync"
	"time"
)

// ugly json response structs
type UserResponse struct {
	Results []struct {
		Name Name `json:"name"`
	} `json:"results"`
}

type JokeResponse struct {
	Value string `json:"value"`
}

// pretty json structs
type Name struct {
	First string `json:"first"`
	Last  string `json:"last"`
}

type CombinedResponse struct {
	Name  Name   `json:"name"`
	Joke  string `json:"joke"`
	Combo string `json:"combo"`
}

func main() {
	fmt.Println("Server running on :5000")
	http.ListenAndServe(":5000", Core{})
}

type Core struct {
	Names  []Name
	Jokes  []string
	Combos []CombinedResponse
	mu     sync.RWMutex
	Client *http.Client
}

func (c Core) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if len(c.Combos) == 0 {
		http.Error(w, "No combos ready", http.StatusServiceUnavailable)
		return
	}

	i := len(c.Combos)
	rn := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(i)

	combo := c.Combos[rn]

	c.Combos[rn] = c.Combos[i-1]
	c.Combos = c.Combos[:i-1]

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(combo)
}

func (c Core) fetchJoke() (string, error) {
	resp, err := c.Client.Get("https://api.chucknorris.io/jokes/random")
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("unexpected status: %d", resp.StatusCode)
	}

	var j JokeResponse
	if err := json.NewDecoder(resp.Body).Decode(&j); err != nil {
		return "", err
	}
	return j.Value, nil
}

func (c Core) fetchNames(count int) ([]Name, error) {
	u, err := url.Parse("https://randomuser.me/api/")
	if err != nil {
		return nil, err
	}
	q := u.Query()
	q.Set("gender", "male")
	q.Set("nat", "US")
	q.Set("results", fmt.Sprintf("%d", count))
	q.Set("inc", "name")
	u.RawQuery = q.Encode()

	resp, err := c.Client.Get(u.String())
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status: %d", resp.StatusCode)
	}

	return parseUser(resp.Body)
}

func parseUser(r io.Reader) ([]Name, error) {
	var ur UserResponse
	if err := json.NewDecoder(r).Decode(&ur); err != nil {
		return nil, err
	}
	if len(ur.Results) == 0 {
		return nil, errors.New("no users returned")
	}
	names := make([]Name, 0, len(ur.Results))
	for _, r := range ur.Results {
		names = append(names, r.Name)
	}
	return names, nil
}
