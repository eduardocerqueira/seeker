//date: 2025-03-24T17:10:41Z
//url: https://api.github.com/gists/1c9ea54e8f8e0849a75510603b23504f
//owner: https://api.github.com/users/fulldump

package main

import (
	"bufio"
	"fmt"
	"log"
	"net"
	"strings"

	"github.com/fulldump/goconfig"
)

type Config struct {
	Addr   string `json:"addr" usage:"Address"`
	Domain string `json:"domain" usage:"Server domain"`
}

type Server struct {
	Config  Config
	Handler Handler
}

type Handler interface {
	HandleSMTP(email Email) error
}

type Email = string

func NewServer(c Config) *Server {
	return &Server{
		Config: c,
	}
}

func (s *Server) Serve(l net.Listener) error {
	for {
		// Acepta nuevas conexiones
		conn, err := l.Accept()
		if err != nil {
			log.Printf("Error al aceptar conexión: %v\n", err)
			continue
		}
		go s.handleConnection(conn)
	}

	return nil
}

func (s *Server) ListenAndServe() error {
	ln, err := net.Listen("tcp", s.Config.Addr)
	if err != nil {
		return err
	}
	return s.Serve(ln)
}

func main() {

	c := Config{
		Addr:   ":2525",
		Domain: "testmail.hola.cloud",
	}

	goconfig.Read(&c)

	s := NewServer(c)

	fmt.Println("Listening on", s.Config.Addr)

	err := s.ListenAndServe()
	if err != nil {
		fmt.Println("ERROR", err.Error())
	}

}

func (s *Server) handleConnection(conn net.Conn) {
	defer conn.Close()

	// Saludo inicial al cliente
	fmt.Fprintf(conn, "220 "+s.Config.Domain+" ESMTP Servidor SMTP\r\n")

	scanner := bufio.NewScanner(conn)
	for scanner.Scan() {
		line := scanner.Text()
		log.Printf("<%s", line)

		// Divide el comando y sus argumentos
		commandParts := strings.SplitN(line, " ", 2)
		// command := strings.ToUpper(commandParts[0])

		var out []byte

		switch {
		case strings.HasPrefix(line, "HELO"), strings.HasPrefix(line, "EHLO"):
			out = []byte(fmt.Sprintf("250 Hello %s, I am glad to meet you\r\n", commandParts[1]))
		case strings.HasPrefix(line, "MAIL FROM:"):
			out = []byte("250 OK\r\n")
		case strings.HasPrefix(line, "RCPT TO:"):
			out = []byte("250 OK\r\n")
		case strings.HasPrefix(line, "DATA"):
			out = []byte("354 End data with <CR><LF>.<CR><LF>\r\n")
			log.Printf(">%s", string(out))
			conn.Write(out)

			message := strings.Builder{}
			for limit := 0; limit < 200000; limit++ {
				if !scanner.Scan() {
					log.Println("Unexpected end of message")
				}
				l := scanner.Text()
				message.WriteString(l)
				message.WriteString("\r\n")
				if l == "." {
					break
				}
			}

			log.Println("Message received:\n", message.String())

			out = []byte("250 OK: queued as 12345")
		case strings.EqualFold(line, "."):
			out = []byte("250 OK: mensaje recibido\r\n")
		case strings.HasPrefix(line, "QUIT"):
			out = []byte("221 Have a nice day\r\n")
			log.Printf(">%s", string(out))
			conn.Write(out)
			return
		default:
			out = []byte("502 Comando no implementado\r\n")
		}

		if out != nil {
			log.Printf(">%s", string(out))
			conn.Write(out)
		}

	}

	if err := scanner.Err(); err != nil {
		log.Printf("Error al leer: %v\n", err)
	}
}
