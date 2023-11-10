//date: 2023-11-10T16:42:33Z
//url: https://api.github.com/gists/b6864f1c84a57e4abd9c70f2f7065839
//owner: https://api.github.com/users/readpe

// NetworkElement represents a power system network element.
type NetworkElement interface {
	// SetModel assigns the NetworkElement model.
	SetModel(m *Model)

	// Name returns the unique name of the NetworkElement. Uniqueness is per type.
	Name() string

	// CalcYPrim calculates the YPrim matrix and returns a copy.
	CalcYPrim() matrix.CMatrix

	// ProcessBusDefinitions processes the bus definitions and applies it to the model..
	ProcessBusDefinitions(m *Model) error

	// Terms returns the element terminals.
	Terms() []Term

	// NodeRefs returns the global node references for all terminals.
	NodeRefs() []NodeRef

	// Currents returns the terminal currents into the circuit element.
	Currents() []complex128
}

// PCElement represents a Power Conversion Element.
type PCElement interface {
	// Implements NetworkElement.
	NetworkElement

	// InjCurrents returns the power conversion Element injection currents.
	InjCurrents() []complex128
}