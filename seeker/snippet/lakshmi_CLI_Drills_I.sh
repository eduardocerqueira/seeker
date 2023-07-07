#date: 2023-07-07T16:49:14Z
#url: https://api.github.com/gists/3dd5302bd412071b46de19d98be17545
#owner: https://api.github.com/users/nlprasanna5

#1
mkdir hello   # created the directory of hello
cd hello      # navigating to the hello directory
mkdir five one  # created the directories of five and one
cd five       # navigating to the directory of five
mkdir six     #created the directory of six which is inside of five
cd six       #moved to the directory of six
touch c.txt   #created c.txt file which is inside of six directory
mkdir seven    #In the directory of six,created a directory of "seven"
cd seven      #navigating to the directory of seven
touch error.log     #Inside of "seven" directory,Created a file named error.log
cd ../../../one      #going back to the directory of one
touch a.txt b.txt   #created two files named a.txt and b.txt in the directory of one
mkdir two     #created a directory of two in the current directory
cd two        #navigating to the directory of two
touch d.txt    #created a file named d.txt inside of the directory of two
mkdir three    #created a directory of three 
cd three     #moving to the directory of three
touch e.txt    #created a file named e.txt in the directory of three
mkdir four      #created a directory folder named four
cd four        #navigating to the directory of four 
touch access.log       #created a file named access.log in the directory of four
cd ../../../../../      #going back to the parent directory
tree              #view directory contents


#2
cd hello/five/six/seven/    #navigating to the directory of seven
rm error.log          #removing a file named error.log by using rm command
cd ../../../one/two/three/four/      #navigating to the directory of four
rm access.log        #removed a file named access.log
cd ../../../../../    #going back to the main directory
tree 



#3
cd hello/one/   #navigating to the directory of one
nano a.txt      #edited a.txt file with the given content

#4
cd ../        # navigating to the hello directory
rm -R five     # removed the directory of five
mv one uno     # changed the name of one to uno
cd uno         #moving to the directory of uno
mv a.txt two    # moved a.txt file to the directory of two
cd ../../      # going back to the main directory
tree 

cd hello/uno/two/     #navigating to the folder of two
cat a.txt          # read a.txt file



#mkdir -> used to create a directory or folder
#touch ->used to create a file directory
#cd  -> used to move to the different directories
#rm  -> used to remove the directory
#nano  -> used to edit a file
#cat -> used to read the content 
#ls -> lists all files and folders directories