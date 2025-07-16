//date: 2025-07-16T17:10:07Z
//url: https://api.github.com/gists/bde3751b1563e79b6ac4359ec51554b7
//owner: https://api.github.com/users/WilliamSampaio

package main

import (
    "log"
    "os"
    "os/signal"
    "golang.org/x/crypto/ssh"
    "golang.org/x/term"
)

func main() {
    // Configurações do cliente SSH
    config := &ssh.ClientConfig{
        User: "server", // substitua pelo seu usuário
        Auth: []ssh.AuthMethod{
            ssh.Password("password"), // ou use chave privada
        },
        HostKeyCallback: ssh.InsecureIgnoreHostKey(), // cuidado: desativa verificação da chave do host
    }

    // Conecta ao servidor SSH
    client, err := ssh.Dial("tcp", "host:22", config)
    if err != nil {
        log.Fatalf("Falha ao conectar: %v", err)
    }
    defer client.Close()

    // Cria nova sessão SSH
    session, err := client.NewSession()
    if err != nil {
        log.Fatalf("Falha ao criar sessão: %v", err)
    }
    defer session.Close()

    // Prepara o terminal local para modo raw (interativo)
    oldState, err := term.MakeRaw(int(os.Stdin.Fd()))
    if err != nil {
        log.Fatalf("Falha ao entrar em modo raw: %v", err)
    }
    defer term.Restore(int(os.Stdin.Fd()), oldState)

    // Captura Ctrl+C para restaurar terminal
    c := make(chan os.Signal, 1)
    signal.Notify(c, os.Interrupt)
    go func() {
        <-c
        term.Restore(int(os.Stdin.Fd()), oldState)
        os.Exit(0)
    }()

    // Redireciona IO (entrada/saída padrão)
    session.Stdin = os.Stdin
    session.Stdout = os.Stdout
    session.Stderr = os.Stderr

    // Solicita TTY (modo interativo)
    modes := ssh.TerminalModes{
        ssh.ECHO:          1,
        ssh.TTY_OP_ISPEED: 14400,
        ssh.TTY_OP_OSPEED: 14400,
    }

    if err := session.RequestPty("xterm", 80, 40, modes); err != nil {
        log.Fatalf("Falha ao solicitar TTY: %v", err)
    }

    // Inicia shell interativo
    if err := session.Shell(); err != nil {
        log.Fatalf("Falha ao iniciar shell: %v", err)
    }

    // Espera até a sessão terminar
    session.Wait()
}
