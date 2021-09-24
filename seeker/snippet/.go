//date: 2021-09-24T17:09:46Z
//url: https://api.github.com/gists/533ad93138aa4927d3cfbdcb4102fe3e
//owner: https://api.github.com/users/mariostyvenv

package main

import "fmt"

type IConveyorFactory interface {
	crearGuia()
	anularGuia()
	solicitarRecogida()
	estadoGuia()
}

// Conveyor: Envía
type Envia struct {
}

func (e Envia) crearGuia() {
	fmt.Println("Creando guia con Envia...")
}
func (e Envia) anularGuia() {

}
func (e Envia) solicitarRecogida() {

}
func (e Envia) estadoGuia() {

}

//Conveyor: TCC
type TCC struct {
}

func (s TCC) crearGuia() {
	fmt.Println("Creando guia con TCC...")
}
func (s TCC) anularGuia() {

}
func (s TCC) solicitarRecogida() {

}
func (s TCC) estadoGuia() {

}

//Conveyor: Servientrega
type Servientrega struct {
}

func (s Servientrega) crearGuia() {
	fmt.Println("Creando guia con Servientrega...")
}
func (s Servientrega) anularGuia() {

}
func (s Servientrega) solicitarRecogida() {

}
func (s Servientrega) estadoGuia() {

}

//Conveyor: Saferbo
type Saferbo struct {
}

func (s Saferbo) crearGuia() {
	fmt.Println("Creando guia con Saferbo...")
}
func (s Saferbo) anularGuia() {

}
func (s Saferbo) solicitarRecogida() {

}
func (s Saferbo) estadoGuia() {

}

//Get ConveyorFactory

func getConveyorFactory(conveyor string) (IConveyorFactory, error) {
	switch conveyor {
	case "ENVIA":
		return &Envia{}, nil
	case "TCC":
		return &TCC{}, nil
	case "SERVIENTREGA":
		return &Servientrega{}, nil
	case "SAFERBO":
		return &Saferbo{}, nil
	}
	return nil, fmt.Errorf("No existe la transportadora asociada")
}

func crearGuia(c IConveyorFactory) {
	c.crearGuia()
}

func anularGuia(c IConveyorFactory) {
	c.anularGuia()
}

func solicitarRecogida(c IConveyorFactory) {
	c.solicitarRecogida()
}

func estadoGuia(c IConveyorFactory) {
	c.estadoGuia()
}

func main() {
	valueEnvia, _ := getConveyorFactory("ENVIA")
	valueTCC, _ := getConveyorFactory("TCC")
	valueSERVIENTREGA, _ := getConveyorFactory("SERVIENTREGA")
	valueSAFERBO, _ := getConveyorFactory("SAFERBO")

	crearGuia(valueEnvia)
	crearGuia(valueTCC)
	crearGuia(valueSERVIENTREGA)
	crearGuia(valueSAFERBO)

}
