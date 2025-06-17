//date: 2025-06-17T16:56:39Z
//url: https://api.github.com/gists/774ff158e363771ece706dd0125d5f64
//owner: https://api.github.com/users/blinkinglight


type userModel struct {
	Name    string `json:"name"`
	Country string `json:"country"`
}

type Aggregate struct {
	History []string
}

func (a *Aggregate) ApplyEvent(event *gen.EventEnvelope) error {
	switch event.EventType {
	case "created":
		var user userModel
		if err := json.Unmarshal(event.Payload, &user); err != nil {
			return err
		}
		a.History = append(a.History, "User created: "+user.Name+" from "+user.Country)
	case "updated":
		var user userModel
		if err := json.Unmarshal(event.Payload, &user); err != nil {
			return err
		}
		a.History = append(a.History, "User updated: "+user.Name+" from "+user.Country)
	default:
		return nil // Ignore other event types
	}
	return nil
}
