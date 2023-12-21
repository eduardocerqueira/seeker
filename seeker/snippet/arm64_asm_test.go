//date: 2023-12-21T17:03:16Z
//url: https://api.github.com/gists/9beee6f52f45a22c4f8dde55d0af2fbc
//owner: https://api.github.com/users/mwpcheung

package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"simulator/gapstone"
	"testing"
)

func TestARM64Asm(t *testing.T) {
	// arm64 shellcode 呼叫指定地址
	// con      reserved    I
	// 1101     00          1    0111 0001001000110100 10000   //0x1234
	// 1111     00          1    0110 0101011001111000 10000   //0x5678
	// 1111     00          1    0101 1001101010111100 10000   //0x9abc
	// 1111     00          1    0100 1101111011110000 10000   //0xdef0
	// 0:  90 46 e2 d2   mov     x16, #0x1234,lsl #48
	// 4:  10 cf ca f2   movk    x16, #0x5678, lsl #32
	// 8:  90 57 b3 f2   movk    x16, #0x9abc, lsl #16
	// c:  10 de 9b f2   movk    x16, #0xdef0
	// 10: 00 02 3f d6   blr     x16
	// 14: 1f 20 03 d5   nop
	// 18: 1f 20 03 d5   nop
	// 1c: 1f 20 03 d5   nop
	// 20: 1f 20 03 d5   nop
	// 90 46 e2 d2
	// 10 cf ca f2
	// 90 57 b3 f2
	// 10 de 9b f2
	value := 0x123456789abcdef0
	buf := new(bytes.Buffer)
	subvalue := uint32(value >> 48)
	op1 := uint32(0b11010010111)<<21 | uint32(subvalue<<5) | 0b10000
	binary.Write(buf, binary.LittleEndian, uint32(op1))

	subvalue = uint32(value>>32) & 0xffff
	op1 = uint32(0b11110010110)<<21 | uint32(subvalue<<5) | 0b10000
	binary.Write(buf, binary.LittleEndian, uint32(op1))

	subvalue = uint32(value>>16) & 0xffff
	op1 = uint32(0b11110010101)<<21 | uint32(subvalue<<5) | 0b10000
	binary.Write(buf, binary.LittleEndian, uint32(op1))

	subvalue = uint32(value & 0xffff)
	op1 = uint32(0b11110010100)<<21 | uint32(subvalue<<5) | 0b10000
	binary.Write(buf, binary.LittleEndian, uint32(op1))

	binary.Write(buf, binary.BigEndian, uint32(0x00023fd6))
	binary.Write(buf, binary.BigEndian, uint32(0x1f2003d5))
	binary.Write(buf, binary.BigEndian, uint32(0x1f2003d5))
	binary.Write(buf, binary.BigEndian, uint32(0x1f2003d5))
	binary.Write(buf, binary.BigEndian, uint32(0x1f2003d5))
	fmt.Printf("instructions %x\n", buf.Bytes())
	engine, _ := gapstone.New(gapstone.CS_ARCH_ARM64, gapstone.CS_MODE_ARM)
	inss, err := engine.Disasm(buf.Bytes(), 0x20000000, 0)
	if err != nil {
		fmt.Printf("%s", err.Error())
	}
	for _, ins := range inss {
		fmt.Printf("%x %s %s\n", ins.Address, ins.Mnemonic, ins.OpStr)
	}
}