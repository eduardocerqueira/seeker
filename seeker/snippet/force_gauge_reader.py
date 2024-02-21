#date: 2024-02-21T16:57:15Z
#url: https://api.github.com/gists/c75b28edd5c364da57f40ed97d6485b4
#owner: https://api.github.com/users/jglatts

"""
    Simple class for serial communication between Arduino and Windows machines 
    Saves test data to an output file after completion

    BAUD 9600 FOR XMOONFREE

    Copyright: Z-Axis Connector Company
    Date: 2/9/23
    Author: John Glatts
"""
import serial
import datetime
import time
from time import sleep

class WinSer():

    """
        WinSer Constructor
        
        Args:
        write_out (bool, optional): flag for writing to output file. Defaults to True.
        port (int, optional): COM port of the device. Defaults to 5.
        baud (int, optional): baud rate of the device. Defaults to 9600.
    """
    def __init__(self, write_out=False, port=4, baud=115200):
        self.port = port
        self.bauds = [9600, 115200]
        self.board = None
        self.out_file = open("testData.txt", "a")
        self.write_out = write_out

    """
        Main test method for the LBCC tester
        Will setup comm. with the MCU and then begin the test

        Returns:
            none
    """
    def main(self):
        self.get_board_with_baud()
        self.run()

    def get_board_with_baud(self):
        count = 0
        while (1):
            self.baud = self.bauds[count]
            print("baud = " + str(self.baud))
            if not self.get_board():
                print('\nNo Board Connected!\n')
                count += 1
                if (count > 1):
                    count = 0
            else:
                sleep(1)
                self.board.write(b'hello')
                self.board.flush()
                print('sent data')
                return

    """
        Begin the comm. with the MCU

        Returns:
            bool: True if the device is found, False otherwise
    """
    def get_board(self):
        com_port = 'COM' + str(self.port)
        print("trying " + com_port)
        try: 
            self.board = serial.Serial(port=com_port, baudrate=self.baud, timeout=.2)
            #self.board = serial.Serial(port=com_port, baudrate=self.baud, timeout=.2, dsrdtr=None)
            #self.board.setRTS(False)
            #self.board.setDTR(False)
            return True
        except:
            return False

    """
        Run the approriate number of tests 

        Returns:
            none
    """
    def run(self):
        print('GETTING DATA\n')
        dateInfo = datetime.datetime.now()
        if (self.write_out):
            data = str(dateInfo.month) + "-" + str(dateInfo.day) + "-" + str(dateInfo.year) + "-" + str(dateInfo.hour) + "-" + str(dateInfo.minute)
            self.write_to_file("\n\n" + "Test Data: " + data + "\n\n");

        while (1):
            uart_str = self.get_uart_str()
            print(uart_str)
            if (self.write_out):
                self.write_to_file(uart_str)
    
    """
        Get data from the MCU and write 
        
        Returns:
            string: message from mcu
    """
    def get_uart_str(self):
        print(self.board.read())
        ret = self.board.read_until().decode()
        return ret

    """
        Write data to file
        
        Returns:
            none
    """
    def write_to_file(self, data_str):
        self.out_file.write(data_str)    


if __name__ == '__main__':
    WinSer().main()
