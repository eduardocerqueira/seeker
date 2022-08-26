//date: 2022-08-26T17:00:19Z
//url: https://api.github.com/gists/00392cabe715b581be60069bde8b0ea5
//owner: https://api.github.com/users/zeroidentidad

package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
)

type GaiaTruncatedData struct {
	Datos []struct {
		CveAgee string `json:"cve_agee"`
		CveAgem string `json:"cve_agem"`
		NomLoc  string `json:"nom_loc"`
	} `json:"datos"`
}

//[GET]/localidades?state=id&text=locName
func localidades(w http.ResponseWriter, r *http.Request) {
	state, text := r.URL.Query().Get("state"), r.URL.Query().Get("text")

	//Validate arguments
	if state == "" || text == "" {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"error": 'Invalid arguments'}`)
		return
	}

	url := "https://gaia.inegi.org.mx/wscatgeo/localidades/buscar/" + text
	resp, _ := http.Get(url)
	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprint(w, `{"error": 'Can not read data'}`)
		return
	}

	var result GaiaTruncatedData
	if err := json.Unmarshal(data, &result); err != nil {
		w.WriteHeader(http.StatusUnprocessableEntity)
		fmt.Fprint(w, `{"error": 'Can not unmarshal JSON'}`)
		return
	}

	locs := []string{}
	for _, loc := range result.Datos {
		if loc.CveAgee == state {
			locs = append(locs, fmt.Sprintf("%s %s", loc.CveAgem, loc.NomLoc))
		}
	}

	payload, err := json.Marshal(locs)
	if err != nil {
		w.WriteHeader(http.StatusUnprocessableEntity)
		fmt.Fprint(w, `{"error": 'Can not unmarshal JSON'}`)
		return
	}

	//Final payload response
	w.WriteHeader(http.StatusOK)
	fmt.Fprint(w, string(payload))
}

func main() {
	http.HandleFunc("/localidades", localidades)
	http.ListenAndServe(":3000", nil)
}
