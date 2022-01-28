//date: 2022-01-28T17:07:07Z
//url: https://api.github.com/gists/9b9f5c417499fd1d1e951bc702d6c898
//owner: https://api.github.com/users/FZFalzar

// M8C Register Renaming
//@author Falzar#4798
//@category _NEW_
//@keybinding 
//@menupath 
//@toolbar 
import java.util.Map;
import java.util.HashMap;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.util.*;
import ghidra.program.model.reloc.*;
import ghidra.program.model.data.*;
import ghidra.program.model.block.*;
import ghidra.program.model.symbol.*;
import ghidra.program.model.scalar.*;
import ghidra.program.model.mem.*;
import ghidra.program.model.listing.*;
import ghidra.program.model.lang.*;
import ghidra.program.model.pcode.*;
import ghidra.program.model.address.*;

public class m8c_renameregisters extends GhidraScript {

    // see page 101-102 of PSoC TRM 001-48461

    // bank 0: user space registers
    Map<String, String> bank0_regs = new HashMap<String, String>();
    // bank 1: configuration registers
    Map<String, String> bank1_regs = new HashMap<String, String>();

    private void init() {
        // === BANK0 REGS ===
        // GPIO REGS
        bank0_regs.put("00", "PRT0DR");
        bank0_regs.put("01", "PRT0IE");
        bank0_regs.put("02", "PRT0GS");
        bank0_regs.put("03", "PRT0DM2");

        bank0_regs.put("04", "PRT1DR");
        bank0_regs.put("05", "PRT1IE");
        bank0_regs.put("06", "PRT1GS");
        bank0_regs.put("07", "PRT1DM2");

        bank0_regs.put("08", "PRT2DR");
        bank0_regs.put("09", "PRT2IE");
        bank0_regs.put("0a", "PRT2GS");
        bank0_regs.put("0b", "PRT2DM2");

        bank0_regs.put("0c", "PRT3DR");
        bank0_regs.put("0d", "PRT3IE");
        bank0_regs.put("0e", "PRT3GS");
        bank0_regs.put("0f", "PRT3DM2");

        bank0_regs.put("10", "PRT4DR");
        bank0_regs.put("11", "PRT4IE");
        bank0_regs.put("12", "PRT4GS");
        bank0_regs.put("13", "PRT4DM2");

        // DBC
        bank0_regs.put("20", "DBC00DR0");
        bank0_regs.put("21", "DBC00DR1");
        bank0_regs.put("22", "DBC00DR2");
        bank0_regs.put("23", "DBC00CR0");

        bank0_regs.put("24", "DBC01DR0");
        bank0_regs.put("25", "DBC01DR1");
        bank0_regs.put("26", "DBC01DR2");
        bank0_regs.put("27", "DBC01CR0");

        bank0_regs.put("30", "DBC10DR0");
        bank0_regs.put("31", "DBC10DR1");
        bank0_regs.put("32", "DBC10DR2");
        bank0_regs.put("33", "DBC10CR0");

        bank0_regs.put("34", "DBC11DR0");
        bank0_regs.put("35", "DBC11DR1");
        bank0_regs.put("36", "DBC11DR2");
        bank0_regs.put("37", "DBC11CR0");


        // DCC
        bank0_regs.put("28", "DCC02DR0");
        bank0_regs.put("29", "DCC02DR1");
        bank0_regs.put("2a", "DCC02DR2");
        bank0_regs.put("2b", "DCC02CR0");

        bank0_regs.put("2c", "DCC03DR0");
        bank0_regs.put("2d", "DCC03DR1");
        bank0_regs.put("2e", "DCC03DR2");
        bank0_regs.put("2f", "DCC03CR0");

        bank0_regs.put("38", "DCC12DR0");
        bank0_regs.put("39", "DCC12DR1");
        bank0_regs.put("3a", "DCC12DR2");
        bank0_regs.put("3b", "DCC12CR0");

        bank0_regs.put("3c", "DCC13DR0");
        bank0_regs.put("3d", "DCC13DR1");
        bank0_regs.put("3e", "DCC13DR2");
        bank0_regs.put("3f", "DCC13CR0");

        // CSD Registers
        bank0_regs.put("50", "CSD0_DR0_L");
        bank0_regs.put("51", "CSD0_DR1_L");
        bank0_regs.put("52", "CSD0_CNT_L");
        bank0_regs.put("53", "CSD0_CR0");
        bank0_regs.put("54", "CSD0_DR0_H");
        bank0_regs.put("55", "CSD0_DR1_H");
        bank0_regs.put("56", "CSD0_CNT_H");
        bank0_regs.put("57", "CSD0_CR1");

        bank0_regs.put("58", "CSD1_DR0_L");
        bank0_regs.put("59", "CSD1_DR1_L");
        bank0_regs.put("5a", "CSD1_CNT_L");
        bank0_regs.put("5b", "CSD1_CR0");
        bank0_regs.put("5c", "CSD1_DR0_H");
        bank0_regs.put("5d", "CSD1_DR1_H");
        bank0_regs.put("5e", "CSD1_CNT_H");
        bank0_regs.put("5f", "CSD1_CR1");

        // Analog MUX
        bank0_regs.put("60", "AMX_IN");
        bank0_regs.put("61", "AMUX_CFG");

        // PWM Control
        bank0_regs.put("62", "PWM_CR");
        bank0_regs.put("63", "ARF_CR");

        // Analog Comparator Bus
        bank0_regs.put("64", "CMP_CR0");
        bank0_regs.put("66", "CMP_CR1");

        // ADC
        bank0_regs.put("68", "ADC0_CR");
        bank0_regs.put("69", "ADC1_CR");
        bank0_regs.put("6a", "SADC_DH");
        bank0_regs.put("6b", "SADC_DL");

        // Temporary Data Registers
        bank0_regs.put("6c", "TMP_DR0");
        bank0_regs.put("6d", "TMP_DR1");
        bank0_regs.put("6e", "TMP_DR2");
        bank0_regs.put("6f", "TMP_DR3");

        // Analog Continuous Type E 
        bank0_regs.put("72", "ACE00CR1");
        bank0_regs.put("73", "ACE00CR2");
        bank0_regs.put("76", "ACE01CR1");
        bank0_regs.put("77", "ACE01CR2");

        // ASE (Analog Switch Capacitor Type E) Control Registers
        bank0_regs.put("80", "ASE10CR0");
        bank0_regs.put("84", "ASE11CR0");

        // RDI (Row Digital Interconnect)
        bank0_regs.put("b0", "RDI0RI");
        bank0_regs.put("b1", "RDI0SYN");
        bank0_regs.put("b2", "RDI0IS");
        bank0_regs.put("b3", "RDI0LT0");
        bank0_regs.put("b4", "RDI0LT1");
        bank0_regs.put("b5", "RDI0RO0");
        bank0_regs.put("b6", "RDI0RO1");
        bank0_regs.put("b7", "RDI0DSM");

        bank0_regs.put("b8", "RDI1RI");
        bank0_regs.put("b9", "RDI1SYN");
        bank0_regs.put("ba", "RDI1IS");
        bank0_regs.put("bb", "RDI1LT0");
        bank0_regs.put("bc", "RDI1LT1");
        bank0_regs.put("bd", "RDI1RO0");
        bank0_regs.put("be", "RDI1RO1");
        bank0_regs.put("bf", "RDI1DSM");

        // Peripherals
        bank0_regs.put("c8", "PWMVREF0");
        bank0_regs.put("c9", "PWMVREF1");
        bank0_regs.put("ca", "IDAC_MODE");
        bank0_regs.put("cb", "PWM_SRC");

        // Peripherals - Trigger Source
        bank0_regs.put("cc", "TS_CR0");
        bank0_regs.put("cd", "TS_CMPH");
        bank0_regs.put("ce", "TS_CMPL");
        bank0_regs.put("cf", "TS_CR1");

        // RAM Paging / Page Pointer Registers
        bank0_regs.put("d0", "CUR_PP");
        bank0_regs.put("d1", "STK_PP");
        bank0_regs.put("d3", "IDX_PP");
        bank0_regs.put("d4", "MVR_PP");
        bank0_regs.put("d5", "MVW_PP");

        // I2C
        bank0_regs.put("d6", "I2C0_CFG");
        bank0_regs.put("d7", "I2C0_SCR");
        bank0_regs.put("d8", "I2C0_DR");
        bank0_regs.put("d9", "I2C0_MSCR");

        // Interrupts
        bank0_regs.put("da", "INT_CLR0");
        bank0_regs.put("db", "INT_CLR1");
        bank0_regs.put("dc", "INT_CLR2");
        bank0_regs.put("dd", "INT_CLR3");
        bank0_regs.put("de", "INT_MSK3");
        bank0_regs.put("df", "INT_MSK2");
        bank0_regs.put("e0", "INT_MSK0");
        bank0_regs.put("e1", "INT_MSK1");
        bank0_regs.put("e2", "INT_VC");

        // Watchdog
        bank0_regs.put("e3", "RES_WDT");

        // ADC DEC (Decimator) Control
        bank0_regs.put("e6", "DEC_CR0");
        bank0_regs.put("e7", "DEC_CR1");

        // Multiply Control
        bank0_regs.put("e8", "MUL0_X");
        bank0_regs.put("e9", "MUL0_Y");
        bank0_regs.put("ea", "MUL0_DH");
        bank0_regs.put("eb", "MUL0_DL");

        // Accumulator Control
        bank0_regs.put("ec", "ACC0_DR1");
        bank0_regs.put("ed", "ACC0_DR0");
        bank0_regs.put("ee", "ACC0_DR3");
        bank0_regs.put("ef", "ACC0_DR2");

        // M8C CPU Flag Register
        bank0_regs.put("f7", "CPU_F");

        // Analog MUX Left/Right DAC Control Registers
        bank0_regs.put("fc", "IDACR_D");
        bank0_regs.put("fd", "IDACL_D");

        // System Status / Control Registers
        bank0_regs.put("fe", "CPU_SCR1");
        bank0_regs.put("ff", "CPU_SCR0");
        
        // ==============================================
        // === BANK1 REGS ===
        bank1_regs.put("00", "PRT0DM0");
        bank1_regs.put("01", "PRT0DM1");
        bank1_regs.put("02", "PRT0IC0");
        bank1_regs.put("03", "PRT0IC1");

        bank1_regs.put("04", "PRT1DM0");
        bank1_regs.put("05", "PRT1DM1");
        bank1_regs.put("06", "PRT1IC0");
        bank1_regs.put("07", "PRT1IC1");

        bank1_regs.put("08", "PRT2DM0");
        bank1_regs.put("09", "PRT2DM1");
        bank1_regs.put("0a", "PRT2IC0");
        bank1_regs.put("0b", "PRT2IC1");

        bank1_regs.put("0c", "PRT3DM0");
        bank1_regs.put("0d", "PRT3DM1");
        bank1_regs.put("0e", "PRT3IC0");
        bank1_regs.put("0f", "PRT3IC1");

        bank1_regs.put("10", "PRT4DM0");
        bank1_regs.put("11", "PRT4DM1");
        bank1_regs.put("12", "PRT4IC0");
        bank1_regs.put("13", "PRT4IC1");
        
        // DBC
        bank0_regs.put("20", "DBC00FN");
        bank0_regs.put("21", "DBC00IN");
        bank0_regs.put("22", "DBC00OU");
        bank0_regs.put("23", "DBC00CR1");

        bank0_regs.put("24", "DBC01FN");
        bank0_regs.put("25", "DBC01IN");
        bank0_regs.put("26", "DBC01OU");
        bank0_regs.put("27", "DBC01CR1");

        bank0_regs.put("30", "DBC10FN");
        bank0_regs.put("31", "DBC10IN");
        bank0_regs.put("32", "DBC10OU");
        bank0_regs.put("33", "DBC10CR1");

        bank0_regs.put("34", "DBC11FN");
        bank0_regs.put("35", "DBC11IN");
        bank0_regs.put("36", "DBC11OU");
        bank0_regs.put("37", "DBC11CR1");


        // DCC
        bank0_regs.put("28", "DCC02FN");
        bank0_regs.put("29", "DCC02IN");
        bank0_regs.put("2a", "DCC02OU");
        bank0_regs.put("2b", "DCC02CR1");

        bank0_regs.put("2c", "DCC03FN");
        bank0_regs.put("2d", "DCC03IN");
        bank0_regs.put("2e", "DCC03OU");
        bank0_regs.put("2f", "DCC03CR1");

        bank0_regs.put("38", "DCC12FN");
        bank0_regs.put("39", "DCC12IN");
        bank0_regs.put("3a", "DCC12OU");
        bank0_regs.put("3b", "DCC12CR1");

        bank0_regs.put("3c", "DCC13FN");
        bank0_regs.put("3d", "DCC13IN");
        bank0_regs.put("3e", "DCC13OU");
        bank0_regs.put("3f", "DCC13CR1");
        
        // CMP
        bank1_regs.put("50", "CMP0CR1");
        bank1_regs.put("51", "CMP0CR2");
        
        bank1_regs.put("53", "VDAC50CR0");
        bank1_regs.put("54", "CMP1CR1");
        bank1_regs.put("55", "CMP1CR2");
        
        bank1_regs.put("57", "VDAC51CR0");
        bank1_regs.put("58", "CSCMPCR0");
        bank1_regs.put("59", "CSCMPGOEN");
        bank1_regs.put("5a", "CSLUTCR0");
        bank1_regs.put("5b", "CMPCOLMUX");
        bank1_regs.put("5c", "CMPPWMCR");
        bank1_regs.put("5d", "CMPFLTCR");
        bank1_regs.put("5e", "CMPCLK1");
        bank1_regs.put("5f", "CMPCLK0");
        
        bank1_regs.put("60", "CLK_CR0");
        bank1_regs.put("61", "CLK_CR1");
        bank1_regs.put("62", "ABF_CR0");
        bank1_regs.put("63", "AMD_CR0");
        bank1_regs.put("64", "CMP_GO_EN");
        
        bank1_regs.put("66", "AMD_CR1");
        bank1_regs.put("67", "ALT_CR0");
        
        bank1_regs.put("6a", "AMUX_CFG1");
        bank1_regs.put("6b", "CLK_CR3");
        bank1_regs.put("6c", "TMP_DR0");
        bank1_regs.put("6d", "TMP_DR1");
        bank1_regs.put("6e", "TMP_DR2");
        bank1_regs.put("6f", "TMP_DR3");
        
        bank1_regs.put("72", "ACE00CR1");
        bank1_regs.put("73", "ACE00CR2");
        
        bank1_regs.put("76", "ACE01CR1");
        bank1_regs.put("77", "ACE01CR2");
        
        bank1_regs.put("80", "ASE10CR0");
        
        bank1_regs.put("84", "ASE11CR0");
        
        bank1_regs.put("a0", "GDI_O_IN_CR");
        bank1_regs.put("a1", "GDI_E_IN_CR");
        bank1_regs.put("a2", "GDI_O_OU_CR");
        bank1_regs.put("a3", "GDI_E_OU_CR");
        bank1_regs.put("a4", "RTC_H");
        bank1_regs.put("a5", "RTC_M");
        bank1_regs.put("a6", "RTC_S");
        bank1_regs.put("a7", "RTC_CR");
        bank1_regs.put("a8", "SADC_CR0");
        bank1_regs.put("a9", "SADC_CR1");
        bank1_regs.put("aa", "SADC_CR2");
        bank1_regs.put("ab", "SADC_CR3TRIM");
        bank1_regs.put("ac", "SADC_CR4");
        bank1_regs.put("ad", "I2C0_ADDR");
        
        bank1_regs.put("b0", "RDI0RI");
        bank1_regs.put("b1", "RDI0SYN");
        bank1_regs.put("b2", "RDI0IS");
        bank1_regs.put("b3", "RDI0LT0");
        bank1_regs.put("b4", "RDI0LT1");
        bank1_regs.put("b5", "RDI0RO0");
        bank1_regs.put("b6", "RDI0RO1");
        bank1_regs.put("b7", "RDI0DSM");
        
        bank1_regs.put("b8", "RDI1RI");
        bank1_regs.put("b9", "RDI1SYN");
        bank1_regs.put("ba", "RDI1IS");
        bank1_regs.put("bb", "RDI1LT0");
        bank1_regs.put("bc", "RDI1LT1");
        bank1_regs.put("bd", "RTI1RO0");
        bank1_regs.put("be", "RTI1RO1");
        bank1_regs.put("bf", "RDI1DSM");
        
        bank1_regs.put("d0", "GDI_O_IN");
        bank1_regs.put("d1", "GDI_E_IN");
        bank1_regs.put("d2", "GDI_O_OU");
        bank1_regs.put("d3", "GDI_E_OU");
        
        bank1_regs.put("d8", "MUX_CR0");
        bank1_regs.put("d9", "MUX_CR1");
        bank1_regs.put("da", "MUX_CR2");
        bank1_regs.put("db", "MUX_CR3");
        bank1_regs.put("dc", "IDAC_CR1");
        bank1_regs.put("dd", "OSC_GO_EN");
        bank1_regs.put("de", "OSC_CR4");
        bank1_regs.put("df", "OSC_CR3");
        bank1_regs.put("e0", "OSC_CR0");
        bank1_regs.put("e1", "OSC_CR1");
        bank1_regs.put("e2", "OSC_CR2");
        bank1_regs.put("e3", "VLT_CR");
        bank1_regs.put("e4", "VLT_CMP");
        bank1_regs.put("e5", "ADC0_TR");
        bank1_regs.put("e6", "ADC1_TR");
        bank1_regs.put("e7", "V2BG_TR");
        bank1_regs.put("e8", "IMO_TR");
        bank1_regs.put("e9", "ILO_TR");
        bank1_regs.put("ea", "BDG_TR");
        bank1_regs.put("eb", "ECO_TR");
        bank1_regs.put("ec", "MUX_CR4");
        
        bank1_regs.put("f7", "CPU_F");
        
        bank1_regs.put("fa", "FLS_PR1");
        
        bank1_regs.put("fd", "IDAC_CR0");
        bank1_regs.put("fe", "CPU_SCR1");
        bank1_regs.put("ff", "CPU_SCR0");
    }


    public void run() throws Exception {
        init();

        //TODO Add User Code Here
        Listing listing = currentProgram.getListing();
        DataIterator it = listing.getData(false);
        while(it.hasNext()) {
            Data d = it.next();
            if(d.getPathName().startsWith("DAT_BANK")) {
                // get address
                Address a = d.getAddress();

                // get register bank number
                // BANK#:XX
                String[] tmp = a.toString().split(":");
                String bankNumber = tmp[0].substring(4);
                String regName = "";
                String regAddr = tmp[1];
                
                // get name from respective bank
                if(bankNumber == "0") regName = bank0_regs.get(regAddr);
                else regName = bank1_regs.get(regAddr);
                
                if(regName == "") regName = "UNK_" + regAddr;

                // set label at address
                println(d.getLabel() + " -> " + regName);
                getSymbolAt(a).setName("B"+bankNumber+"_"+regName, SourceType.USER_DEFINED);
                //println(getSymbolAt(a).getName());
            }
        }
    }

}
