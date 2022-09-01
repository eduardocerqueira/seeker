//date: 2022-09-01T17:00:12Z
//url: https://api.github.com/gists/33ac87210fba30568500218438c854ff
//owner: https://api.github.com/users/Maksclub

type LoginCreds struct {
	Password string `json: "**********"
	Email    string `json:"email"`
}

func (h *Handler) UserLogin(w http.ResponseWriter, req *http.Request) {
	input := LoginCreds{}
	dec := json.NewDecoder(req.Body)
	if err := dec.Decode(&input); err != nil {
		http.Error(w, "email and password not satisfied", http.StatusBadRequest)
		return
	}

	hashedPass : "**********"
	u := h.users.GetByInputCredentials(input.Email, hashedPass)
	if u == nil {
		w.WriteHeader(http.StatusUnauthorized)
		http.Error(w, "user not found", http.StatusUnauthorized)
		return
	}

	if !hasher.EqualPasswords(hashedPass, u.PasswordHash()) {
		http.Error(w, "password bad", http.StatusUnauthorized)
		return
	}

	if !u.Activated() {
		http.Error(w, "user not activated", http.StatusUnauthorized)
		return
	}

	h.writePairTokenResponse(w, u)
}TokenResponse(w, u)
}