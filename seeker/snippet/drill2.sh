#date: 2022-11-25T17:00:30Z
#url: https://api.github.com/gists/4d4e6559e6974bdb2b78e7bb3abc3f0f
#owner: https://api.github.com/users/vamsi220-ai


	### 1. Pipes
	
	
	#    First download the Text Book from the following link (https://raw.githubusercontent.com/bobdeng/owlreader/master/ERead/assets/books/Harry%20Potter%20and%20the%20Goblet%20of%20Fire.txt) by using the command "wget -O filename link". and saved file name as vamsi.
wget -O vamsi https://raw.githubusercontent.com/bobdeng/owlreader/master/ERead/assets/books/Harry%20Potter%20and%20the%20Goblet%20of%20Fire.txt
cat vamsi           #   i checked whether the file is downloaded or not

head -3 vamsi       #   print first three lines from the vamsi file using head command 
tail -10 vamsi      #   print last ten lines from the vamsi file using tail command

grep -o -i Harry 'vamsi' | wc -w       # count the word 'Harry' occure in vamsi file using command grep -o -i 'word' 'file name' 'pipe' 'word count(wc)' -w(word)
grep -o -i Ron 'vamsi' | wc -w         # count the word Ron in vamsi  using this command
grep -o -i Hermione 'vamsi' | wc -w    # count the word Hermione in vamsi  
grep -o -i Dumbledore 'vamsi' | wc -w  # count the word Dumbledore 

sed -n '100,200p' vamsi                # print the lines 100 through 200 using sed command

tr ' ' '\n' < vamsi | sort | uniq -u | wc -w # print the unique words are present in the vamsi using tr ' ' '\n' <vamsi | sort | uniq -u | wc -c command.


      ### 2. Processes, ports
      
      ps -ef | grep firefox            # our browser's process ids (pid) and parent process ids(ppid) using command ps -ef | grep firefox
      killall firefox                  # stop the applicatin browser from command line
      ps aux --sort -%cpu | head -4    # top 3 processes by CPU usage.
      ps aux --sort -%mem | head -4    # the top 3 processes by memory usage.
      
      python3 -m http.server 8000      #  using this i Started a Python HTTP server on port 8000

      sudo python3 -m http.server 90   #  using this i am started a python http server on port 90
      gnome-terminal                   #  open new termina using this command
      
      netstat -at -au                  #  using this i am active connections and the corresponding TCP / UDP ports.
      
      lsof -i :5432                    #  in this command pid of the process that is listening on port 5432
      
   ### 3.Managing software
   
   sudo apt update                     # using theese two commands
   sudo apt install htop               #                          i am installing htop
   htop                                #  to check whether the htop is installed or not
   
   sudo apt-get update                 #  installing vim using 
   sudo apt-get install vim            #		      these two commands
   vim -v			       #  checking vim is installed or not

   sudo apt-get update                 #  i am installed nginx 
   sudo apt-get install nginx          #		      using these two commands
   
   sudo systemctl status nginx         #  Removing nginx 
   sudo apt-get remove nginx           #		using this command
      
### 4. Misc

ifconfig                               #  using this commands 
dig google.com                         #   		     i find my ip address






