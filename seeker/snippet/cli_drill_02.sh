#date: 2024-09-23T17:06:44Z
#url: https://api.github.com/gists/fe80a7dcd99d5face1312153207766aa
#owner: https://api.github.com/users/Akash-Pramod

// CLI drill - 02 :



There are many commands which I have used to complete this CLI drill : 

1. curl -o : curl command is used to extract the information using url
1. head -n : it is used to get the top 3 line from the book
3. tail -n 10 : to get the last 10 line of the book
4. grep : it is used to search a specific pattern of a text in a file. it checks line by line and print that matches the pattern.
5. grep -i : ignore the text case
6. grep -o : print only the matched word not the entire line.
7. wc : it is used to count the lines, words , bytes and characters in a files.
8. wc -l : here -l counts only line
9. awk : it is a text editor  which is used to manipulate the text and analyze the data. it is a text processing tool.
10. I have used of loop, if condition and variable as same a we do in any programming language.
11. gsub : It is global substiturte. I used gsub in awk to replace the word or char that is not alnum by white space.
12. I used length of each item and check that it is a word or white space if it is a word then increase the count var by 1.
13. then finally print the count.  
14. ps : used to display the current running processes of the system
15. ps -e : e display all the running process
16. ps -f : used to display all runing processes in more detail
17. pkill process_name : using pkill we can terminate any process by it's name
18. top : it is used to display all current running processes in real time. it refresh by its self in every 3 seconds.
19. netstat : stands for network statistics. it is used to display all network connections in detail.







// Pipes : 


Q1. Download the contents of "Harry Potter and the Goblet of fire" using the command line from url
Ans : curl -O url

Q2. Print the first three lines in the book
Ans : head -n 3 harry_potter.txt

Q3. Print the last 10 lines in the book
Ans : tail -n 10 harry_potter.txt

Q4. How many times do the following words occur in the book?
Ans : 
* Harry : grep -io "harry" harry_potter.txt | wc -l
* Ron : grep -io "ron" harry_potter.txt | wc -l
* Hermione : grep -io "hermione" harry_potter.txt | wc -l
* Dumbledore : grep -io "dumbledore" harry_potter.txt | wc -l

Q5. Print lines from 100 through 200 in the book
Ans : head -n 200 harry_potter.txt | tail -n 101
Ans : cat harry_potter.txt | head -n 200 | tail -n 101

Q6. How many unique words are present in the book?
Ans : awk '{for(i=1;i<=NF;i++) {gsub(/[^[:alnum:]]/, "", $i); if(length($i) > 0) words[$i]++ } } END{ print length(words) }' harry_potter.txt






// Processes and Ports :

Q1. List your browser's process ids (pid) and parent process ids(ppid)
Ans : ps -e -o pid,ppid,comm | grep chrome



Q2. Stop the browser application from the command line.
Ans : By terminating using pkill in terminal. pkill : it allows us to terminate processes in place of PID.
Ans : pkill chrome


Q3. List the top 3 processes by CPU usage.
Ans : firstly run top command then press P to sort by cpu then press n with 3 to get top 3 processes by cpu.


Q4. List the top 3 processes by memory usage.
Ans : firstly run top command then press M to sort by memory then press n with 3 to get top 3 processes by memory.


Q5. Start a Python HTTP server on port 8000
Ans : python3 -m http.server 8000


Q6. Open another tab. Stop the process you started in the previous step
Ans : ps -ef | grep python3 to check that python process is running or not then using command (pkill python) terminate the stop the process. another way : (kill PID)


Q7. Start a Python HTTP server on port 90
Ans : python3 -m http.server 90


Q8. Display all active connections and the corresponding TCP / UDP ports.
Ans : netstat -tuln or netstat -tu


Q9. Find the pid of the process that is listening on port 5432
Ans : netstat -tuln | grep 5432.




MISC : 

Q1. What is your local IP address ?
Ans : run ifconfig command. we will get the output. if enp4s0 is avialable with IP and my system is connect to internet by cabkle then it's IP address can be my local IP address but here my system is connected by wifi and wlxc4e90a0592ab is for wifi network with IP address present in it so it's IP address is my local IP address. 
IP Address : 192.168.74.249


Q2. Find the IP address of google.com
Ans : host google.com then will get output like this : 
google.com has address 142.250.183.238
google.com has IPv6 address 2404:6800:4009:824::200e
google.com mail is handled by 10 smtp.google.com.

Here , 142.250.183.238 is the google IP address



Q3. How to check if Internet is working using CLI?
Ans : ping -c 4 google.com, run this command if it gives the output then network is connected.


Q4. Where is the node command located? What about code?
Ans : we can find the location of node using any one of below command : which node, whereis node, type -a node
 