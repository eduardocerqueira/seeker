#date: 2023-03-09T16:47:27Z
#url: https://api.github.com/gists/9d3ddbe0f8db7e68dae46ee9eb85ce03
#owner: https://api.github.com/users/TommyPKeane

# NOTE:
# -------------------------------------------------------------------------------------------------
# The bash (Bourne Again SHell) language commands use whitespace instead of commas and parentheses.
#
# All arguments are separated by whitespace, flags are preceeded by dashes (-), and separate 
# commands are delineated by semicolons (;).
#
# For flags, typically, single dashes refer to a single character flag (e.g., rm -r) which makes it
# easy to combine multiple single-character flags with only a single dash (e.g., rm -r), and thus
# two-dashes (--) tends to refer to multi-character (word-like) flags (e.g., pip --force-reinstall)
# that may even contain dashes.
#
# Makefiles are written in the bash scripting language, thus they too are whitespace dependent.
# -------------------------------------------------------------------------------------------------


# COMMAND HELP BLURB ([cmd] IS THE COMMAND)
[cmd] --help;

# COMMAND DOCUMENTATION MANUAL ([cmd] IS THE COMMAND)
man [cmd];

# EXAMPLE WITH CHANGE DIRECTORY (cd)
cd --help;
man cd;

# QUIT MANPAGE (man) DOCUMENTATION:
#	PRESS [q]

# NAVIGATE MANPAGE (man) DOCUMENTATION:
#	[Up]/[Down] CHANGE LINE
#	[Enter] CHANGES LINE
#	[Space] JUMPS BY PAGE


# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


# FORCE COMMAND TO RUN AS SUPER USER (su), THAT IS TO SAY: "super user do [this command]", THUS (sudo)
sudo rm -rf /Users/UserName/Desktop/NewFolder;


# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


# CHANGE DIRECTORY
cd /Users/UserName/Desktop/;

# CREATE DIRECTORY (NEW DESKTOP FOLDER)
mkdir /Users/UserName/Desktop/NewFolderName;

# DELETE DIRECTORY (AND CONTENTS)
rm -r /Users/UserName/Desktop/NewFolderName;

# FORCE DELETE DIRECTORY (AND CONTENTS)
rm -rf /Users/UserName/Desktop/NewFolderName;


# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


# SEE CURRENTLY RUNNING PROCESSES
ps;

# END RUNNING PROCESS
kill [process];

# END ALL INSTANCES OF A RUNNING PROCESS
killall [process];

# FIND IF A COMMAND EXISTS AND WHERE IT IS LOCATED
which cd;


# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


# LIST CONTENTS OF A DIRECTORY (NO ARGUMENT MEANS USE THE CURRENT DIRECTORY)
ls /Users/UserName/Desktop/NewFolderName;

# LIST CONTENTS OF A DIRECTORY WITH HIDDEN FILES AND FOLDERS
ls -a /Users/UserName/Desktop/NewFolderName;

# LIST CONTENTS OF A DIRECTORY WITH HIDDEN FILES AND FOLDERS BUT NOT THE SYMBOLIC LINKS
ls -A /Users/UserName/Desktop/NewFolderName;

# LIST CONTENTS OF A DIRECTORY IN A LIST WITH USER/GROUP/OWNER INFO
ls -l /Users/UserName/Desktop/NewFolderName;

# LIST CONTENTS OF A DIRECTORY IN A "ONE ENTRY PER LINE" OUTPUT
ls -1 /Users/UserName/Desktop/NewFolderName;


# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


# SEND THE OUTPUT OF A COMMAND TO A FILE (>)
ls -a ./ > /Users/UserName/Desktop/CurrentDirectoryContents.txt;


# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


# EXTRACT CONTENTS BASED ON MATCHING A REGULAR EXPRESSION, THAT IS TO SAY "Get Regular Expression Pattern" (grep)
grep -i ^[a-z] /Users/UserName/Desktop/CurrentDirectoryContents.txt;

# The above example will return all the lines in the file whose first character starts with a letter
# between 'a' and 'z' (that is, all of them) and it is case-insensitive because of the '-i' flag.
#
# Lookup regular expressions: http://www.robelle.com/smugbook/regexpr.html


# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


# PASS OUTPUT OF A COMMAND TO THE INPUT OF ANOTHER COMMAND (PIPE: |)
ls -a ./ | grep -i ^[a-z];


# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


# BASIC MATHEMATICS (SPACE BETWEEN EACH ARGUMENT) (MATHEMATICAL exprESSION)
expr 3 + 5;

# STORE MATHEMATIC OUTPUT AS VARIABLE
a=$(expr 3 + 5);

# PRINT THE VALUE OF A VARIABLE
echo $a;


# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------