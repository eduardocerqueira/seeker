//date: 2024-08-16T17:08:59Z
//url: https://api.github.com/gists/db97c2388e335f6f542a874fba4745b8
//owner: https://api.github.com/users/hariso

func (p *Instance) SetStatus(s Status) {
	p.status.Store(&s)
}

func (p *Instance) GetStatus() Status {
	return *p.status.Load()
}