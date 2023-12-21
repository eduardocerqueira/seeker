//date: 2023-12-21T17:08:46Z
//url: https://api.github.com/gists/bf8a2c4cc3a8a54a54941d4fa16f3fa6
//owner: https://api.github.com/users/mwpcheung

package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"simulator/gapstone"
	"testing"
)

func TestAMD64Asm(t *testing.T) {
	/*
		mov rax,xxxxxxxx    ;48 B8 F0 DE BC 9A 78 56 34 12
		call rax			;ff d0
		nop					;90
		nop					;90
		nop					;90
		nop					;90
	*/
	addr := uint64(0x123456789abcdef0)
	buf := new(bytes.Buffer)
	binary.Write(buf, binary.BigEndian, uint16(0x48b8))
	binary.Write(buf, binary.LittleEndian, addr)
	binary.Write(buf, binary.BigEndian, uint16(0xffd0))
	binary.Write(buf, binary.BigEndian, uint32(0x90909090))

	engine, _ := gapstone.New(gapstone.CS_ARCH_X86, gapstone.CS_MODE_64)
	inss, err := engine.Disasm(buf.Bytes(), 0x20000000, 0)
	if err != nil {
		fmt.Printf("%s", err.Error())
	}
	for _, ins := range inss {
		fmt.Printf("%x %s %s\n", ins.Address, ins.Mnemonic, ins.OpStr)
	}
}
