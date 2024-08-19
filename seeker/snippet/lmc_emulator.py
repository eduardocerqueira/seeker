#date: 2024-08-19T17:03:32Z
#url: https://api.github.com/gists/5d11916083b009976906447b9186c5c2
#owner: https://api.github.com/users/michirakara

import time
import re

class LMC:
    def __init__(self):
        self.memory=[0]*100
        self.register=0
    def print_ram(self,now):
        print("\033[0;0f\033[J",end="")
        for i in range(10):
            for j in range(10):
                if now==i*10+j:
                    print(f"\033[44;30m {self.memory[i*10+j]:>3} ",end="")
                else:
                    print(f"\033[47;30m {self.memory[i*10+j]:>3} ",end="")
            print("\033[0m")
        print(f"\033[44;30mregister: {self.register}\033[0m")
    def execute(self):
        now=0
        while self.memory[now]!=0:
            self.print_ram(now)
            instruction=self.memory[now]
            if 100<=instruction<200:
                self.register+=self.memory[instruction%100]
                now+=1
            elif 200<=instruction<300:
                self.register-=self.memory[instruction%100]
                now+=1
            elif 300<=instruction<400:
                self.memory[instruction%100]=self.register
                now+=1
            elif 500<=instruction<600:
                self.register=self.memory[instruction%100]
                now+=1
            elif 600<=instruction<700:
                now=instruction%100
            elif 700<=instruction<800:
                if self.register==0:
                    now=instruction%100
                else:
                    now+=1
            elif 800<=instruction<900:
                if self.register>0:
                    now=instruction%100
                else:
                    now+=1
            elif instruction==901:
                self.register=int(input("input a number: "))
                now+=1
            elif instruction==902:
                print("output:",self.register)
                now+=1
            elif instruction==0:
                break
            else:
                print("error!")
            time.sleep(0.3)

src=input("source file:")

computer=LMC()

with open(src) as f:
    lines=f.readlines()
    for i in range(len(lines)):
        if re.fullmatch(r'(@[0-9][0-9]\s*[0-9]+)?\s*(//.*)?\s*',lines[i]):
            line=re.match(r'@[0-9][0-9]\s*[0-9]+',lines[i])
            if line:
                addr,op=line.group(0).split()
                computer.memory[int(addr[1:])]=int(op) 
        else:
            print(f"invalid instruction at line {i+1}")
computer.execute()
