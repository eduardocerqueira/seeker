#date: 2025-03-04T17:07:36Z
#url: https://api.github.com/gists/49091f26ae3ee944087c47777b67023e
#owner: https://api.github.com/users/fenugrec

#!/bin/bash
#
# grab ownership of ghidra project specified in arg1 or current dir if absent
#
#	- looks for the first <projectname>.gpr file
#	- modifies owner name <projectname>/project.prp
#
# example : ./ghrab.sh ~/RE/coolstuff


# Discussed in https://github.com/NationalSecurityAgency/ghidra/issues/5507

# " if you create a project under your current username then try to give it to someone else,
# or otherwise just copy it to another machine or VM with a different user name you will get an error:
# "..Failure to open project ... NoOwnerException: Project is not owned by xxx.."
#
# obvious caveat : this is a bad idea if permissions are not appropriate on some of the project files.



u=`whoami`

# use arg1 if defined, else pwd
base_dir="${1:-$(pwd)}"


gpr_file=$(find "$base_dir" -name *.gpr -printf %f -quit)
if [ -z "$gpr_file" ]; then
	echo "no .gpr file found !"
	exit 1
fi

prj_name=${gpr_file%.gpr}
prp_file="$base_dir/${prj_name}.rep/project.prp"
if ! [ -f "$prp_file" ]; then
	echo "$prp_file" not found !
	exit 1
fi

echo Found ghidra project: "$prj_name"
sed "$prp_file" -i -e "/OWNER/s/VALUE=\"[^\"]\+/VALUE=\"$u/"
