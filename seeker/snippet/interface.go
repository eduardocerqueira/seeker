//date: 2022-12-30T16:30:47Z
//url: https://api.github.com/gists/556f6c3dcbc6e34807229bc7f29a8149
//owner: https://api.github.com/users/TripleDogDare

package serrah

type ReeError interface {
	Code() string
	Message() string
	Details() map[string]string
	Cause() error
	Error() string
}

type TheError struct {
	TheCode string
}

func (e *TheError) Code() string               { return e.TheCode }
func (e *TheError) Message() string            { return e.TheCode }
func (e *TheError) Details() map[string]string { return nil }
func (e *TheError) Cause() error               { return nil }
func (e *TheError) Error() string              { return e.Message() }
